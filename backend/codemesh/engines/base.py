"""Geometry engine interface.

Every multi-view geometry backend (ggml/Vulkan, PyTorch CPU/CUDA, ...) implements
`GeometryEngine`, so the pipeline never depends on a specific model or runtime.

Conventions: OpenCV cameras (x right, y down, z forward); extrinsics map world -> camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ViewGeometry:
    """Per-view outputs of a geometry engine, at the engine's working resolution.

    N = number of views, H x W = engine output resolution.
    """

    intrinsics: np.ndarray  # (N, 3, 3) float64, in engine pixel coordinates
    extrinsics: np.ndarray  # (N, 4, 4) float64, world -> camera
    depth: np.ndarray  # (N, H, W) float32, z-depth along the camera axis
    depth_conf: np.ndarray  # (N, H, W) float32, higher = more reliable
    # Maps engine pixel coords (u, v, 1) to original-image pixel coords. Captures any
    # resize/crop/pad the engine applied, so results can be used on full-res frames.
    pixel_to_original: np.ndarray  # (N, 3, 3) float64
    points: np.ndarray | None = None  # (N, H, W, 3) float32 world points, if the model predicts them
    points_conf: np.ndarray | None = None  # (N, H, W) float32
    metric_scale: bool = False  # True when units are meters
    info: dict = field(default_factory=dict)  # engine name, model, device, precision, timings


class GeometryEngine(Protocol):
    """A loaded model that can be called repeatedly (load weights once, infer many times)."""

    name: str

    def info(self) -> dict:
        """Return {engine, model, device, precision, ...} for /capabilities and benchmarks."""
        ...

    def infer(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None) -> ViewGeometry:
        """Run multi-view inference.

        images: full-resolution RGB uint8 arrays (H, W, 3); the engine resizes internally.
        masks: optional full-res bool masks; engines may use them or ignore them.
        """
        ...

    def close(self) -> None:
        """Free model memory."""
        ...
