"""GeometryEngine that runs TorchEngine on a Modal GPU (deploy/modal_app.py, class GeometryWorker).

Frames travel as JPEG q=95 at full resolution, masks as PNG; results come back as an .npz of
`ViewGeometry` fields. The codec helpers below are shared by the client and the remote worker.
"""

from __future__ import annotations

import io
import json
from dataclasses import fields

import numpy as np
from PIL import Image

from codemesh.engines.base import ViewGeometry

APP_NAME, WORKER_CLASS = "codemesh", "GeometryWorker"


def encode_image(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def encode_mask(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def decode_mask(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("L")) > 127


def geometry_to_npz(geo: ViewGeometry) -> bytes:
    arrays = {f.name: getattr(geo, f.name) for f in fields(geo) if isinstance(getattr(geo, f.name), np.ndarray)}
    buf = io.BytesIO()
    np.savez(buf, **arrays, metric_scale=np.bool_(geo.metric_scale), info=np.str_(json.dumps(geo.info)))
    return buf.getvalue()


def geometry_from_npz(data: bytes) -> ViewGeometry:
    with np.load(io.BytesIO(data)) as z:
        return ViewGeometry(
            intrinsics=z["intrinsics"],
            extrinsics=z["extrinsics"],
            depth=z["depth"],
            depth_conf=z["depth_conf"],
            pixel_to_original=z["pixel_to_original"],
            points=z["points"] if "points" in z else None,
            points_conf=z["points_conf"] if "points_conf" in z else None,
            metric_scale=bool(z["metric_scale"]),
            info=json.loads(str(z["info"])),
        )


class ModalEngine:
    """Calls the deployed GeometryWorker; the GPU container loads the model once and scales to zero."""

    name = "modal"

    def __init__(self, worker=None) -> None:
        if worker is None:
            import modal

            worker = modal.Cls.from_name(APP_NAME, WORKER_CLASS)()
        self.worker = worker
        self._info: dict = {"engine": self.name, "model": "mapanything", "device": "cuda (modal L4)"}

    def info(self) -> dict:
        return self._info

    def infer(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None) -> ViewGeometry:
        """Send full-res frames to the GPU worker and decode its ViewGeometry."""
        payload = [encode_image(im) for im in images]
        mask_payload = [encode_mask(m) for m in masks] if masks is not None else None
        geo = geometry_from_npz(self.worker.infer.remote(payload, mask_payload))
        self._info = {**self._info, **geo.info, "engine": self.name}
        return geo

    def close(self) -> None:
        pass
