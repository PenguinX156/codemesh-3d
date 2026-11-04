"""Mask payload decoding, kept import-light so process-pool workers start fast.

(Spawned workers re-import this module; `sam.py` pulls in cv2/av/PIL, which costs seconds each.)
"""

from __future__ import annotations

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster


def decode_one(args: tuple[str, str, int, int]) -> np.ndarray:
    """(encoding, payload, width, height) -> (height, width) bool raster."""
    encoding, payload, width, height = args
    raw = decode_mask_to_raster(SegmentationMask(encoding, payload, width, height))  # type: ignore[arg-type]
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).astype(bool)
