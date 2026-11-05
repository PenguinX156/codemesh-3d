"""Geometry stage: valid masked frames -> engine cameras/depth (geometry/views.npz, cameras.json).

Views: masks/valid.json, capped per preset (draft 16, standard 24, high 32) and spread evenly;
fewer frames are all used, fewer than MIN_VIEWS fail. Turntable mode blacks out pixels outside
the dilated mask so the static background cannot pin the cameras. views.npz holds the
`ViewGeometry` arrays plus `frame_indices` (the contract read by fuse/mesh).
"""

from __future__ import annotations

import json
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from codemesh.engines.base import ViewGeometry
from codemesh.engines.factory import get_engine
from codemesh.jobs.cache import cached_metrics, invalidate, stage_key, write_manifest
from codemesh.jobs.context import StageContext
from codemesh.pipeline.cameras import cameras_from_geometry, save_cameras
from codemesh.pipeline.segment import MIN_VIEWS

PRESET_VIEWS = {"draft": 16, "standard": 24, "high": 32}
TURNTABLE_DILATE = 0.01  # of the long image side: keep a margin of background around the object


def pick_views(valid: list[int], cap: int) -> list[int]:
    """Up to `cap` indices from `valid`, evenly spread (all of them if there are fewer)."""
    if len(valid) <= cap:
        return list(valid)
    return [valid[i] for i in np.linspace(0, len(valid) - 1, cap).round().astype(int)]


def save_views(path, geo: ViewGeometry, frame_indices: list[int]) -> None:
    arrays = {k: getattr(geo, k) for k in ("intrinsics", "extrinsics", "depth", "depth_conf", "pixel_to_original",
                                           "points", "points_conf") if getattr(geo, k) is not None}
    np.savez(path, **arrays, frame_indices=np.asarray(frame_indices, np.int32))


def run(ctx: StageContext, mode: str = "walkaround") -> dict[str, Any]:
    """Infer geometry for the selected views. `mode` = walkaround | turntable. Returns metrics."""
    t0 = time.perf_counter()
    preset = ctx.settings.get("preset", "standard")
    if preset not in PRESET_VIEWS:
        raise ValueError(f"unknown preset {preset!r}")
    valid = json.loads((ctx.paths.masks / "valid.json").read_text(encoding="utf-8"))
    if len(valid) < MIN_VIEWS:
        raise RuntimeError(f"only {len(valid)} frames have a usable mask; need at least {MIN_VIEWS} views")
    entries = {e["index"]: e for e in json.loads((ctx.paths.frames / "frames.json").read_text("utf-8"))["frames"]}
    views = pick_views(valid, PRESET_VIEWS[preset])
    files = [ctx.paths.frames / entries[i]["file"] for i in views]
    mask_files = [ctx.paths.masks / f"{files[n].stem}.png" for n in range(len(views))]
    out = ctx.paths.geometry
    out.mkdir(parents=True, exist_ok=True)
    key = stage_key("geometry", files + mask_files, {"views": views, "mode": mode, "engine": ctx.settings.get("engine")})
    if (hit := cached_metrics(out, "geometry", key)) is not None:
        return {**hit, "cached": True}
    invalidate(out, "geometry")

    ctx.progress(0.05, f"loading {len(views)} views")
    images = [np.array(Image.open(f).convert("RGB")) for f in files]  # writable: turntable blacks out
    masks = [np.asarray(Image.open(f).convert("L")) >= 128 for f in mask_files]
    if mode == "turntable":
        for img, m in zip(images, masks):
            r = max(1, int(TURNTABLE_DILATE * max(m.shape)))
            keep = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
            img[keep == 0] = 0
    ctx.progress(0.1, "loading geometry engine")
    engine = get_engine(ctx.settings.get("engine"))
    ctx.progress(0.2, f"inferring geometry ({engine.name})")
    t_inf = time.perf_counter()
    geo = engine.infer(images, masks)
    t_inf = time.perf_counter() - t_inf

    save_views(out / "views.npz", geo, views)
    cams = cameras_from_geometry(geo, views, [f"frames/{f.name}" for f in files],
                                 [(int(entries[i]["width"]), int(entries[i]["height"])) for i in views])
    save_cameras(out / "cameras.json", cams, geo.metric_scale)
    info = engine.info()
    metrics = {"views": len(views), "frame_indices": views, "engine": info.get("engine", engine.name),
               "model": info.get("model"), "device": info.get("device"), "metric_scale": geo.metric_scale,
               "engine_resolution": list(geo.depth.shape[1:]), "infer_seconds": round(t_inf, 2),
               "seconds": round(time.perf_counter() - t0, 2)}
    write_manifest(out, "geometry", key, [out / "views.npz", out / "cameras.json"], metrics)
    ctx.progress(1.0, f"geometry for {len(views)} views")
    return metrics
