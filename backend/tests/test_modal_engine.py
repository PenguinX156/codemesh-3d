"""ModalEngine round trip against a fake remote worker (no Modal account or GPU needed)."""

from __future__ import annotations

import numpy as np

from codemesh.engines.base import ViewGeometry
from codemesh.engines.modal_engine import ModalEngine, decode_image, decode_mask, geometry_to_npz


class _FakeWorker:
    """Mimics `modal.Cls(...)().infer.remote`: decodes the payload and returns npz bytes."""

    def __init__(self) -> None:
        self.infer = self
        self.seen: tuple[list[np.ndarray], list[np.ndarray] | None] | None = None

    def remote(self, images: list[bytes], masks: list[bytes] | None) -> bytes:
        imgs = [decode_image(b) for b in images]
        ms = [decode_mask(b) for b in masks] if masks is not None else None
        self.seen = (imgs, ms)
        n, h, w = len(imgs), 4, 6
        geo = ViewGeometry(
            intrinsics=np.tile(np.eye(3), (n, 1, 1)),
            extrinsics=np.tile(np.eye(4), (n, 1, 1)),
            depth=np.ones((n, h, w), np.float32),
            depth_conf=np.full((n, h, w), 2.0, np.float32),
            pixel_to_original=np.tile(np.diag([10.0, 10.0, 1.0]), (n, 1, 1)),
            points=np.zeros((n, h, w, 3), np.float32),
            points_conf=None,
            metric_scale=True,
            info={"device": "cuda:0", "gpu": "NVIDIA L4", "infer_seconds": 1.5},
        )
        return geometry_to_npz(geo)


def test_modal_engine_round_trip() -> None:
    ramp = np.linspace(0, 255, 60, dtype=np.uint8)
    frames = [np.stack([np.tile(ramp, (40, 1))] * 3, axis=-1) for _ in range(2)]
    masks = [np.zeros((40, 60), bool) for _ in range(2)]
    masks[0][10:30, 20:40] = True
    worker = _FakeWorker()

    geo = ModalEngine(worker=worker).infer(frames, masks)

    sent_imgs, sent_masks = worker.seen
    assert [im.shape for im in sent_imgs] == [(40, 60, 3)] * 2  # full resolution, no downscale
    assert np.abs(sent_imgs[0].astype(int) - frames[0]).mean() < 3  # JPEG q95: lossy but close
    assert all((a == b).all() for a, b in zip(sent_masks, masks))  # masks are lossless
    assert geo.depth.shape == (2, 4, 6) and geo.depth.dtype == np.float32
    assert geo.extrinsics.dtype == np.float64 and geo.pixel_to_original[0, 0, 0] == 10.0
    assert geo.points is not None and geo.points_conf is None
    assert geo.metric_scale is True and geo.info["gpu"] == "NVIDIA L4"


def test_modal_engine_without_masks() -> None:
    worker = _FakeWorker()
    ModalEngine(worker=worker).infer([np.zeros((8, 8, 3), np.uint8)])
    assert worker.seen[1] is None
