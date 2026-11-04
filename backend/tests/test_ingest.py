"""Ingest stage tests on small synthetic videos/photos generated here (no external files)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
from PIL import Image

from codemesh.jobs.paths import JobPaths
from codemesh.pipeline import ingest

W, H = 320, 240


def _texture(seed: int = 0) -> np.ndarray:
    """Large smooth features (so thumbnails differ when shifted) plus fine detail (sharpness)."""
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8), (W * 2, H), interpolation=cv2.INTER_CUBIC)
    checker = ((np.indices((H, W * 2)).sum(0) // 2) % 2 * 60).astype(np.int16)[..., None]
    return np.clip(coarse.astype(np.int16) + checker - 30, 0, 255).astype(np.uint8)


def make_video(path: Path, n: int = 60, sharp_every: int | None = 10, static: bool = False) -> Path:
    """mp4 with `n` frames; if `sharp_every`, only frames i % sharp_every == 3 are sharp."""
    tex = _texture()
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"qscale": "2"}
        for i in range(n):
            shift = 0 if static else 5 * i
            img = np.ascontiguousarray(tex[:, shift % W : shift % W + W])
            if sharp_every and i % sharp_every != 3:
                img = cv2.GaussianBlur(img, (0, 0), 3)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def make_images(folder: Path, n: int = 3, orientation: int | None = None) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        im = Image.fromarray(np.ascontiguousarray(_texture(i)[:, :W]))
        exif = Image.Exif()
        if orientation:
            exif[0x0112] = orientation
        p = folder / f"img_{i}.jpg"
        im.save(p, quality=95, exif=exif)
        paths.append(p)
    return paths


def test_video_picks_sharpest_frame_per_window(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.mp4", n=60, sharp_every=10)
    data = ingest.ingest_video(video, tmp_path / "frames", k=6)
    assert [f["source_index"] for f in data["frames"]] == [3, 13, 23, 33, 43, 53]
    assert data["source_frames"] == 60 and data["dropped_duplicates"] == 0
    first = data["frames"][0]
    assert (first["width"], first["height"]) == (W, H)
    full = cv2.imread(str(tmp_path / "frames" / first["file"]))
    thumb = cv2.imread(str(tmp_path / "frames" / first["thumb"]))
    assert full.shape == (H, W, 3) and thumb.shape == (H, W, 3)  # never upscaled


def test_static_video_drops_duplicates(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.mp4", n=30, sharp_every=None, static=True)
    data = ingest.ingest_video(video, tmp_path / "frames", k=6)
    assert len(data["frames"]) == 1 and data["dropped_duplicates"] == 5


def test_window_selector_keeps_all_when_fewer_items_than_k() -> None:
    picked = []
    sel = ingest.WindowSelector(total=3, k=10, emit=lambda c: picked.append(c.index))
    for i in range(3):
        sel.feed(i, 1.0, np.full((32, 32), i * 50, np.float32), None)
    sel.finish()
    assert picked == [0, 1, 2]


def test_rotate_upright() -> None:
    img = np.zeros((2, 4, 3), np.uint8)
    img[0, 0] = 255  # top-left marker
    rotated = ingest.rotate_upright(img, -90)  # display matrix -90 = rotate clockwise for display
    assert rotated.shape == (4, 2, 3) and rotated[0, 1, 0] == 255
    assert ingest.rotate_upright(img, 0) is img


def test_images_apply_exif_orientation(tmp_path: Path) -> None:
    paths = make_images(tmp_path / "in", n=3, orientation=6)  # 6 = rotate 90 CW to display
    data = ingest.ingest_images(paths, tmp_path / "frames", k=40)
    assert len(data["frames"]) == 3
    assert (data["frames"][0]["width"], data["frames"][0]["height"]) == (H, W)


def _ctx(root: Path, preset: str = "draft") -> SimpleNamespace:
    paths = JobPaths(root)
    paths.create()
    artifacts: dict[str, str] = {}
    return SimpleNamespace(paths=paths, settings={"preset": preset}, progress=lambda f, m=None: None,
                           set_artifact=artifacts.__setitem__, artifacts=artifacts)


def test_run_writes_frames_json_and_uses_cache(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path / "job")
    make_video(ctx.paths.input / "000_v.mp4", n=48, sharp_every=None)
    first = ingest.run(ctx)
    assert first["frames"] == 24 and "cached" not in first
    frames = json.loads((ctx.paths.frames / "frames.json").read_text())
    assert len(frames["frames"]) == 24 and ctx.artifacts["frames_json"] == "frames/frames.json"
    second = ingest.run(ctx)
    assert second == {**first, "cached": True}
    ctx.settings["preset"] = "standard"  # different K -> cache miss, recomputed
    assert ingest.run(ctx)["frames"] == 40
