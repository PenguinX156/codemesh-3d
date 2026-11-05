"""Offline tests of the pipeline registry and stage adapters on the synthetic scene (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from codemesh.engines.base import ViewGeometry
from codemesh.jobs.context import StageContext
from codemesh.jobs.models import Job
from codemesh.jobs.paths import job_paths
from codemesh.jobs.store import JobStore
from codemesh.meta import sam
from codemesh.pipeline import ACTIONS, geometry, reconstruct_settings, segment
from codemesh.pipeline.cameras import load_cameras
from synthetic import make_scene


@pytest.fixture
def setup(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(Job(mode="walkaround", settings={"preset": "standard"}))
    paths = job_paths(tmp_path, job.id)
    paths.create()
    make_scene(paths.root)
    yield store, job, paths
    store.close()


def run_stage(store: JobStore, job: Job, paths, action: str, name: str, settings: dict) -> dict:
    store.update(job.id, lambda j: j.settings.update(settings))
    stages = ACTIONS[action].stages
    i = [s.name for s in stages].index(name)
    ctx = StageContext(store, job.id, paths, store.get(job.id).settings, name, i, len(stages))
    return stages[i].run(ctx)


def test_registry_has_all_stages() -> None:
    assert [s.name for s in ACTIONS["segment"].stages] == ["segment"]
    assert [s.name for s in ACTIONS["reconstruct"].stages] == ["geometry", "fuse", "mesh", "texture", "export",
                                                               "evaluate", "agent", "quality"]
    flat = reconstruct_settings({"preset": "standard", "selection": {}, "reconstruct": {"preset": "draft",
                                                                                       "known_dimension": None}})
    assert flat == {"preset": "draft"}


def test_segment_photo_job_writes_masks_and_drops_bad_frames(setup, monkeypatch) -> None:
    store, job, paths = setup
    frames = json.loads((paths.frames / "frames.json").read_text())
    for e in frames["frames"]:
        e["source_index"] = e["index"]
    (paths.frames / "frames.json").write_text(json.dumps({"kind": "images", **frames}))
    gt = {i: cv2.imread(str(paths.masks / f"{i:04d}.png"), 0) >= 128 for i in range(len(frames["frames"]))}
    gt[3] = np.ones_like(gt[3])  # whole image = background picked -> border rule drops it
    gt[5] = np.zeros_like(gt[5])  # nothing found
    calls = []

    def fake_segment_video(video, *, points=(), text=None, prompt_frame=0, frame_indices=None, size=None, client=None):
        with av.open(str(video)) as c:
            n = sum(1 for _ in c.decode(video=0))
        calls.append((n, size, bool(points), text, prompt_frame, list(frame_indices)))
        res = sam.SegResult(size=size, selected_id=1)
        for i in range(n):  # padded frames beyond the real ones must be ignored
            m = cv2.resize(gt[i % len(gt)].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                res.masks[i] = sam.ObjectMask(1, i, m, (0, 0, 1, 1), 0.9)
        return res

    monkeypatch.setattr(sam, "segment_video", fake_segment_video)
    m = run_stage(store, job, paths, "segment", "segment",
                  {"selection": {"frame_index": 2, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "text": None}})
    w, h = frames["frames"][0]["width"], frames["frames"][0]["height"]
    assert calls == [(segment.SAM_MIN_FRAMES, (w // 2 * 2, h // 2 * 2), True, None, 2, list(range(len(gt))))]
    assert m["clip_size"] == [w // 2 * 2, h // 2 * 2]
    valid = json.loads((paths.masks / "valid.json").read_text())
    assert 3 not in valid and 5 not in valid and len(valid) == len(gt) - 2
    assert m["frames_empty"] == [5] and m["frames_border"] == [3]
    assert (cv2.imread(str(paths.masks / "0000.png"), 0) >= 128).sum() == gt[0].sum()
    assert store.get(job.id).artifacts["masks_preview"] == "masks/preview.jpg"
    assert (paths.masks / "preview.jpg").stat().st_size > 1000


class FakeEngine:
    """Returns the synthetic scene's engine outputs for the views the geometry stage picks."""

    name = "fake"

    def __init__(self, views_npz: Path, indices: list[int]) -> None:
        with np.load(views_npz) as z:
            self.views = dict(z)
        self.indices = indices
        self.images: list[np.ndarray] = []

    def info(self) -> dict:
        return {"engine": "fake", "device": "cpu"}

    def infer(self, images, masks=None) -> ViewGeometry:
        assert len(images) == len(self.indices) and masks is not None and len(masks) == len(images)
        self.images = images
        v, sel = self.views, self.indices
        return ViewGeometry(v["intrinsics"][sel], v["extrinsics"][sel], v["depth"][sel], v["depth_conf"][sel],
                            v["pixel_to_original"][sel], metric_scale=True)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("mode", ["walkaround", "turntable"])
def test_geometry_caps_views_and_writes_contract(setup, monkeypatch, mode: str) -> None:
    store, job, paths = setup
    store.update(job.id, lambda j: setattr(j, "mode", mode))
    n = len(json.loads((paths.frames / "frames.json").read_text())["frames"])
    (paths.masks / "valid.json").write_text(json.dumps(list(range(n))))
    picked = geometry.pick_views(list(range(n)), geometry.PRESET_VIEWS["draft"])
    assert len(picked) == 24 and picked[0] == 0 and picked[-1] == n - 1
    engine = FakeEngine(paths.geometry / "views.npz", picked)
    monkeypatch.setattr(geometry, "get_engine", lambda name=None: engine)
    m = run_stage(store, job, paths, "reconstruct", "geometry", {"reconstruct": {"preset": "draft"}})
    assert m["views"] == 24 and m["engine"] == "fake"
    with np.load(paths.geometry / "views.npz") as z:
        assert z["frame_indices"].tolist() == picked and z["depth"].shape[0] == 24
    cams, meta = load_cameras(paths.geometry / "cameras.json")
    assert [c.index for c in cams] == picked and meta["metric_scale"] is True
    outside = cv2.imread(str(paths.masks / f"{picked[0]:04d}.png"), 0) == 0
    blacked = (engine.images[0][outside] == 0).all(axis=-1).mean()
    assert blacked > 0.9 if mode == "turntable" else blacked < 0.1
    assert store.get(job.id).metrics["geometry_views"] == 24


def test_geometry_needs_eight_views(setup) -> None:
    store, job, paths = setup
    (paths.masks / "valid.json").write_text(json.dumps([0, 1, 2]))
    with pytest.raises(RuntimeError, match="need at least 8"):
        run_stage(store, job, paths, "reconstruct", "geometry", {})


def test_fuse_mesh_adapters_record_metrics(setup) -> None:
    store, job, paths = setup
    run_stage(store, job, paths, "reconstruct", "fuse", {"reconstruct": {"preset": "draft"}})
    m = run_stage(store, job, paths, "reconstruct", "mesh", {})
    metrics = store.get(job.id).metrics
    assert metrics["tris_low"] == m["tris_low"] > 0 and metrics["fuse_points"] > 1000
    assert set(metrics["stage_seconds"]) == {"fuse", "mesh"}
    assert (paths.mesh / "low.ply").is_file()


def test_engine_factory_resolves_auto(monkeypatch) -> None:
    from codemesh.engines import factory

    from codemesh.engines import rocm_engine

    monkeypatch.setattr(rocm_engine, "ROCM_PYTHON", Path("does/not/exist"))
    monkeypatch.setattr(factory, "modal_available", lambda timeout=0: False)
    assert factory.resolve_name("auto") == "torch"
    monkeypatch.setattr(factory, "modal_available", lambda timeout=0: True)
    assert factory.resolve_name("auto") == "modal"
    monkeypatch.setattr(rocm_engine, "ROCM_PYTHON", Path(__file__))  # any existing file
    assert factory.resolve_name("auto") == "rocm" and factory.resolve_name("ROCM") == "rocm"
    with pytest.raises(ValueError):
        factory.resolve_name("ggml")


def test_view_count_uses_fixed_counts_only() -> None:
    assert [geometry.view_count(n, 32) for n in (8, 15, 16, 23, 24, 31, 40)] == [8, 8, 16, 16, 24, 24, 32]
    assert geometry.view_count(100, 16) == 16 and geometry.view_count(20, 24) == 16
    assert len(geometry.pick_views(list(range(20)), 24)) == 16
    assert len(geometry.pick_views(list(range(9)), 16)) == 8


def test_upscale_mask_smooths_to_full_size() -> None:
    small = np.zeros((20, 20), bool)
    small[5:15, 5:15] = True
    big = segment.upscale_mask(small, (100, 80))
    assert big.shape == (80, 100) and big.dtype == bool and big[40, 50] and not big[2, 2]
    assert segment.upscale_mask(small, (20, 20)) is small
