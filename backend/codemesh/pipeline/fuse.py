"""Fuse stage: engine depth maps -> one clean, colored, oriented point cloud (fuse/points.ply).

Per view: map engine pixels to the original image with `pixel_to_original`, keep pixels inside
the (eroded) object mask and in the top `conf_keep` fraction of confidence, unproject them and
color them from the FULL-RES frame. Then: adaptive voxel downsampling, statistical + radius
outlier removal, and normals oriented toward the camera that observed each point.

Settings: source (depth|points, default depth), conf_keep (0.7), erode_px (2, engine pixels),
max_points (2_000_000), voxel (None = half the median engine pixel footprint).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from codemesh.pipeline.cameras import Camera, load_cameras

Progress = Callable[[float, str], None]


def find_mask(job_dir: Path, cam: Camera) -> Path:
    """masks/<index>.png; accepts the frame file's stem (e.g. 0003.png) or the bare index."""
    for name in (Path(cam.image).stem, f"{cam.index:04d}", str(cam.index)):
        p = job_dir / "masks" / f"{name}.png"
        if p.exists():
            return p
    raise FileNotFoundError(f"no mask for frame {cam.index} in {job_dir / 'masks'}")


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def view_points(job_dir: Path, cam: Camera, depth: np.ndarray, conf: np.ndarray, K: np.ndarray, E: np.ndarray,
                p2o: np.ndarray, pts: np.ndarray | None, conf_keep: float, erode_px: int
                ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """World points (M, 3) float32 and colors (M, 3) uint8 of one view, plus stats."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u, v, np.ones_like(u)], -1)
    orig = pix @ p2o.T
    mx, my = (orig[..., :2] / orig[..., 2:]).astype(np.float32).transpose(2, 0, 1)

    mask_full = cv2.imread(str(find_mask(job_dir, cam)), cv2.IMREAD_GRAYSCALE)
    mask = cv2.remap(mask_full, mx, my, cv2.INTER_LINEAR, borderValue=0) >= 128
    if erode_px > 0:  # drop silhouette pixels: depth there mixes object and background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    mask &= np.isfinite(depth) & (depth > 0)
    stats = {"index": cam.index, "masked": int(mask.sum()), "kept": 0}
    if not mask.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), stats
    keep = mask & (conf >= np.quantile(conf[mask], 1 - conf_keep))

    if pts is None:
        cam_pts = (pix[keep] @ np.linalg.inv(K).T) * depth[keep, None]
        world = (cam_pts - E[:3, 3]) @ E[:3, :3]
    else:
        world = pts[keep]
    rgb = cv2.remap(read_rgb(job_dir / cam.image), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    stats["kept"] = int(keep.sum())
    return world.astype(np.float32), rgb[keep], stats


def run(job_dir: Path, settings: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    """Fuse all views into fuse/points.ply. Returns metrics."""
    t0 = time.perf_counter()
    report = progress or (lambda f, m: None)
    source = settings.get("source", "depth")
    conf_keep = float(settings.get("conf_keep", 0.7))
    erode_px = int(settings.get("erode_px", 2))
    max_points = int(settings.get("max_points", 2_000_000))
    if source not in ("depth", "points"):
        raise ValueError(f"fuse source must be depth|points, got {source!r}")

    with np.load(job_dir / "geometry" / "views.npz") as npz:  # NpzFile re-reads an array on every access
        views = dict(npz)
    cams = {c.index: c for c in load_cameras(job_dir / "geometry" / "cameras.json")[0]}
    use_points = source == "points" and "points" in views
    conf_all = views["points_conf"] if use_points and "points_conf" in views else views["depth_conf"]
    indices = [int(i) for i in views["frame_indices"]]

    all_pts, all_rgb, all_view, per_view, footprints = [], [], [], [], []
    for i, idx in enumerate(indices):
        report(0.6 * i / len(indices), f"fusing view {i + 1}/{len(indices)}")
        K, E, depth = views["intrinsics"][i], views["extrinsics"][i], views["depth"][i]
        pts, rgb, stats = view_points(job_dir, cams[idx], depth, conf_all[i], K, E, views["pixel_to_original"][i],
                                      views["points"][i] if use_points else None, conf_keep, erode_px)
        all_pts.append(pts)
        all_rgb.append(rgb)
        all_view.append(np.full(len(pts), i, np.int32))
        per_view.append(stats)
        if stats["kept"]:
            footprints.append(np.median(depth[depth > 0]) / K[0, 0])
    points, colors, view_of = np.concatenate(all_pts), np.concatenate(all_rgb), np.concatenate(all_view)
    if len(points) < 100:
        raise RuntimeError(f"fuse: only {len(points)} points inside masks; check masks and cameras")
    n_raw = len(points)

    report(0.65, "downsampling")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255)
    voxel = float(settings.get("voxel") or 0.5 * np.median(footprints))
    down = pcd.voxel_down_sample(voxel)
    while len(down.points) > max_points:  # density scales ~ 1/voxel^2 on a surface
        voxel *= 1.05 * np.sqrt(len(down.points) / max_points)
        down = pcd.voxel_down_sample(voxel)
    n_voxel = len(down.points)

    report(0.75, "removing outliers")
    down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    down, _ = down.remove_radius_outlier(nb_points=6, radius=4 * voxel)
    if len(down.points) < 100:
        raise RuntimeError("fuse: outlier removal left fewer than 100 points")

    report(0.85, "estimating normals")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    kept = np.asarray(down.points)
    _, nearest = cKDTree(points).query(kept, workers=-1)
    centers = np.stack([cams[indices[v]].center for v in range(len(indices))])
    to_cam = centers[view_of[nearest]] - kept
    normals = np.asarray(down.normals)
    flip = (normals * to_cam).sum(1) < 0
    normals[flip] *= -1
    down.normals = o3d.utility.Vector3dVector(normals)

    out = job_dir / "fuse"
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / "points.ply"), down)
    metrics = {"points_raw": n_raw, "points_voxel": n_voxel, "points": len(down.points), "voxel": voxel,
               "source": "points" if use_points else "depth", "views": per_view,
               "seconds": round(time.perf_counter() - t0, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    report(1.0, f"fused {len(down.points)} points")
    return metrics
