"""Full-resolution camera set shared by the fuse, mesh, texture and evaluate stages.

Stored as `geometry/cameras.json` in the job dir. OpenCV convention: x right, y down,
z forward; `extrinsic` maps world -> camera; `K` is in ORIGINAL image pixels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from codemesh.engines.base import ViewGeometry


@dataclass
class Camera:
    index: int  # frame index (matches frames.json / masks/<index>.png)
    image: str  # path to the full-res frame, relative to the job dir
    width: int
    height: int
    K: list[list[float]]  # 3x3 intrinsics in original pixels
    extrinsic: list[list[float]]  # 4x4 world -> camera

    @property
    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=np.float64)

    @property
    def extrinsic_np(self) -> np.ndarray:
        return np.asarray(self.extrinsic, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        E = self.extrinsic_np
        return -E[:3, :3].T @ E[:3, 3]


def cameras_from_geometry(
    geo: ViewGeometry, frame_indices: list[int], image_paths: list[str], sizes: list[tuple[int, int]]
) -> list[Camera]:
    """Convert engine cameras to full-res cameras. sizes = [(width, height)] of the originals."""
    cams = []
    for i, (idx, path, (w, h)) in enumerate(zip(frame_indices, image_paths, sizes)):
        K_full = geo.pixel_to_original[i] @ geo.intrinsics[i]
        cams.append(Camera(idx, path, w, h, K_full.tolist(), geo.extrinsics[i].tolist()))
    return cams


def save_cameras(path: Path, cams: list[Camera], metric_scale: bool, scale_factor: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"metric_scale": metric_scale, "scale_factor": scale_factor, "cameras": [asdict(c) for c in cams]}
    path.write_text(json.dumps(data, indent=1))


def load_cameras(path: Path) -> tuple[list[Camera], dict]:
    """Return (cameras, meta) where meta has metric_scale and scale_factor."""
    data = json.loads(Path(path).read_text())
    cams = [Camera(**c) for c in data["cameras"]]
    return cams, {k: v for k, v in data.items() if k != "cameras"}
