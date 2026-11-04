"""Contract tests for engines/preprocess.py and engines/torch_engine.py.

Fast tests check preprocessing against the official loaders and the pixel mapping.
`slow` tests need the weights in models/ (bench/phase0/reference.py documents the download).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codemesh.config import MODELS_DIR, THIRD_PARTY_DIR
from codemesh.engines.preprocess import preprocess_mapanything, preprocess_vggt, resize_crop_matrix

EXAMPLES = THIRD_PARTY_DIR / "vggt" / "examples"
needs_examples = pytest.mark.skipif(not EXAMPLES.exists(), reason="third_party/vggt examples not cloned")


def _paths(set_name: str, n: int = 2) -> list[Path]:
    return sorted((EXAMPLES / set_name / "images").iterdir())[:n]


def _load(paths: list[Path]) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


@needs_examples
@pytest.mark.parametrize("set_name", ["kitchen", "llff_flower"])
def test_preprocess_matches_official_loaders(set_name: str) -> None:
    pytest.importorskip("mapanything")
    pytest.importorskip("vggt")
    from mapanything.utils.image import load_images
    from vggt.utils.load_fn import load_and_preprocess_images

    paths = _paths(set_name)
    ours_ma, _ = preprocess_mapanything(_load(paths))
    official_ma = np.concatenate([v["img"].numpy() for v in load_images([str(p) for p in paths])])
    np.testing.assert_allclose(ours_ma, official_ma, atol=1e-5)

    ours_vggt, _ = preprocess_vggt(_load(paths))
    official_vggt = load_and_preprocess_images([str(p) for p in paths]).numpy()
    np.testing.assert_allclose(ours_vggt, official_vggt, atol=1e-6)


def test_pixel_mapping_keeps_centers_and_crop_offset() -> None:
    # Resize 1000x600 -> 500x300, no crop: engine center maps to original center.
    m = resize_crop_matrix((1000, 600), (500, 300), (0, 0))
    np.testing.assert_allclose(m @ [249.5, 149.5, 1], [499.5, 299.5, 1])
    # Pure crop (no resize): a pure shift by the crop offset.
    m = resize_crop_matrix((640, 480), (640, 480), (10, 20))
    np.testing.assert_allclose(m @ [0, 0, 1], [10, 20, 1])


def test_preprocess_shapes_and_mapping() -> None:
    img = np.zeros((520, 779, 3), np.uint8)  # kitchen size: 779x520
    ma, ma_map = preprocess_mapanything([img, img])
    assert ma.shape == (2, 3, 336, 518) and ma_map.shape == (2, 3, 3)
    vg, vg_map = preprocess_vggt([img])
    assert vg.shape == (1, 3, 350, 518)
    # Engine corners land inside the original image.
    for m, (h, w) in ((ma_map[0], ma.shape[-2:]), (vg_map[0], vg.shape[-2:])):
        u, v, _ = m @ [w - 1, h - 1, 1]
        assert 0 <= u < 779 and 0 <= v < 520
    with pytest.raises(ValueError):
        preprocess_vggt([img, np.zeros((600, 800, 3), np.uint8)])


@pytest.mark.slow
@needs_examples
@pytest.mark.parametrize("model,weights", [("mapanything", "map-anything-apache"), ("vggt", "VGGT-1B")])
def test_engine_contract(model: str, weights: str) -> None:
    if not (MODELS_DIR / weights / "model.safetensors").exists():
        pytest.skip(f"weights for {model} not downloaded")
    from codemesh.engines.torch_engine import TorchEngine

    images = _load(_paths("kitchen", 2))
    engine = TorchEngine(model, device="cpu")
    geo = engine.infer(images)
    engine.close()

    n, h, w = geo.depth.shape
    assert n == 2 and (h, w) == tuple(geo.info["engine_hw"])
    assert geo.intrinsics.shape == (2, 3, 3) and geo.extrinsics.shape == (2, 4, 4)
    assert geo.depth_conf.shape == (n, h, w) and geo.points.shape == (n, h, w, 3)
    assert geo.pixel_to_original.shape == (2, 3, 3)
    np.testing.assert_allclose(geo.extrinsics[:, 3], [[0, 0, 0, 1]] * 2)
    rot = geo.extrinsics[:, :3, :3]
    np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.tile(np.eye(3), (2, 1, 1)), atol=1e-4)
    assert (geo.depth > 0).mean() > 0.95
    assert geo.metric_scale == (model == "mapanything")

    # World points expressed in each camera have z close to the predicted depth.
    cam = np.einsum("nij,nhwj->nhwi", rot, geo.points) + geo.extrinsics[:, None, None, :3, 3]
    rel = np.abs(cam[..., 2] - geo.depth) / geo.depth
    assert np.median(rel) < (1e-4 if model == "mapanything" else 0.05)
