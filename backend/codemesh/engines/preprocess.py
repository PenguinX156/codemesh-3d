"""Model input preprocessing, reimplemented from the official repos on in-memory arrays.

Both functions reproduce the official loaders bit-for-bit (checked in
backend/tests/test_torch_engine.py) but take RGB uint8 arrays instead of file paths, and
also return the exact engine-pixel -> original-pixel mapping.

Pixel convention everywhere: integer coordinates are pixel centers (OpenCV), i.e. engine
pixel (u, v) is column u / row v of the engine-resolution depth map.

Only NumPy + Pillow are used so non-PyTorch runtimes (ggml, ONNX) can share this code.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PATCH = 14
VGGT_SIZE = 518
# MapAnything "fixed_mapping" table for resolution_set=518: aspect (W/H) -> (W, H).
# Copied from third_party/map-anything/mapanything/utils/image.py (RESOLUTION_MAPPINGS[518]).
MAPANYTHING_518 = {
    1.000: (518, 518),
    1.321: (518, 392),
    1.542: (518, 336),
    1.762: (518, 294),
    2.056: (518, 252),
    3.083: (518, 168),
    0.757: (392, 518),
    0.649: (336, 518),
    0.567: (294, 518),
    0.486: (252, 518),
}
DINOV2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINOV2_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_crop_matrix(orig_wh: tuple[int, int], resized_wh: tuple[int, int], crop_xy: tuple[int, int]) -> np.ndarray:
    """3x3 map from engine pixel (u, v, 1) to original pixel, for resize-then-crop.

    A resize from W to w maps pixel centers as x_orig = (x + 0.5) * W / w - 0.5; the crop
    shifts engine pixels by the crop offset first.
    """
    sx = orig_wh[0] / resized_wh[0]
    sy = orig_wh[1] / resized_wh[1]
    return np.array(
        [
            [sx, 0.0, (crop_xy[0] + 0.5) * sx - 0.5],
            [0.0, sy, (crop_xy[1] + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )


def _check(images: list[np.ndarray]) -> None:
    if not images:
        raise ValueError("At least one image is required")
    for im in images:
        if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 (H, W, 3), got {im.dtype} {im.shape}")


def preprocess_mapanything(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """MapAnything `load_images(resize_mode="fixed_mapping", resolution_set=518)`.

    One target size from the mean aspect ratio; each image is scaled so it covers the
    target (Lanczos down / bicubic up), then center-cropped.

    Returns (imgs (N, 3, H, W) float32 DINOv2-normalized, pixel_to_original (N, 3, 3)).
    """
    _check(images)
    mean_aspect = sum(im.shape[1] / im.shape[0] for im in images) / len(images)
    tw, th = MAPANYTHING_518[min(MAPANYTHING_518, key=lambda a: abs(a - mean_aspect))]
    out, maps = [], []
    for im in images:
        h0, w0 = im.shape[:2]
        scale = max(tw / w0, th / h0) + 1e-8
        rw, rh = int(np.floor(w0 * scale)), int(np.floor(h0 * scale))
        pil = Image.fromarray(im).resize((rw, rh), Image.LANCZOS if scale < 1 else Image.BICUBIC)
        left, top = (rw - tw) // 2, (rh - th) // 2
        crop = np.asarray(pil.crop((left, top, left + tw, top + th)), dtype=np.float32) / 255.0
        out.append(((crop - DINOV2_MEAN) / DINOV2_STD).transpose(2, 0, 1))
        maps.append(resize_crop_matrix((w0, h0), (rw, rh), (left, top)))
    return np.stack(out).astype(np.float32), np.stack(maps)


def preprocess_vggt(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """VGGT `load_and_preprocess_images(mode="crop")`.

    Width -> 518 (bicubic), height rounded to a multiple of 14, center-cropped to 518 if taller.

    Returns (imgs (N, 3, H, W) float32 in [0, 1], pixel_to_original (N, 3, 3)).
    """
    _check(images)
    out, maps = [], []
    for im in images:
        h0, w0 = im.shape[:2]
        nw, nh = VGGT_SIZE, round(h0 * (VGGT_SIZE / w0) / PATCH) * PATCH
        arr = np.asarray(Image.fromarray(im).resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        top = (nh - VGGT_SIZE) // 2 if nh > VGGT_SIZE else 0
        arr = arr[top : top + min(nh, VGGT_SIZE)]
        out.append(arr.transpose(2, 0, 1))
        maps.append(resize_crop_matrix((w0, h0), (nw, nh), (0, top)))
    if len({o.shape for o in out}) > 1:
        # shortcut: mixed sizes are rejected; official VGGT pads them white, add if photo sets need it.
        raise ValueError("VGGT preprocessing needs images of one size")
    return np.stack(out), np.stack(maps)
