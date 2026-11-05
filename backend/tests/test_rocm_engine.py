"""RocmEngine tests: skipped when the ROCm venv (bench/rocm/.venv) or the MapAnything weights are absent.

The full test starts the worker (model load + kernel warmup take minutes), so it is opt-in:
set CODEMESH_TEST_ROCM=1 to run it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from codemesh.config import MODELS_DIR, REPO_ROOT
from codemesh.engines.rocm_engine import RocmEngine, rocm_available

pytestmark = pytest.mark.skipif(not rocm_available(), reason="ROCm venv bench/rocm/.venv not installed")

IMG_DIR = REPO_ROOT / "third_party" / "vggt" / "examples" / "kitchen" / "images"
REF = REPO_ROOT / "bench" / "phase0" / "out" / "ref_mapanything_kitchen_8.npz"


def test_missing_worker_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("codemesh.engines.rocm_engine.WORKER", tmp_path / "nope.py")
    with pytest.raises(FileNotFoundError):
        RocmEngine()


@pytest.mark.skipif(
    os.environ.get("CODEMESH_TEST_ROCM") != "1" or not (MODELS_DIR / "map-anything-apache" / "config.json").exists()
    or not REF.exists(),
    reason="set CODEMESH_TEST_ROCM=1 and provide MapAnything weights + CPU reference to run the GPU test",
)
def test_rocm_matches_cpu_reference_and_survives_a_crash() -> None:
    from PIL import Image

    ref = np.load(REF)
    frames = [np.asarray(Image.open(IMG_DIR / n).convert("RGB")) for n in ref["frame_names"]]
    engine = RocmEngine(startup_timeout=900, infer_timeout=1200)
    try:
        geo = engine.infer(frames)
        assert geo.depth.shape == ref["depth"].shape and geo.metric_scale
        # bf16 vs fp32 CPU: poses and depth must stay close, not bit-identical
        cos = (np.trace(geo.extrinsics[:, :3, :3] @ ref["extrinsics"][:, :3, :3].transpose(0, 2, 1), axis1=1, axis2=2) - 1) / 2
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 5.0
        keep = ref["depth"] > 0
        assert np.median(np.abs(geo.depth - ref["depth"])[keep] / ref["depth"][keep]) < 0.1

        engine._proc.kill()  # simulate a worker crash: the next call must restart and still succeed
        engine._proc.wait()
        assert engine.infer(frames[:2]).depth.shape[0] == 2
    finally:
        engine.close()
