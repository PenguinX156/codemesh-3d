"""Visual hull from full-res masks + cameras, refined by engine depth (free-space carving).

Why: small objects cover a few % of the frame, so at the engine's ~518 px input their depth is
sparse and noisy and Poisson on the fused points yields a fragment. Masks are full-res and
cameras come from background-rich full frames, so the silhouettes pin the shape far better.

Pipeline: triangulate the mask-centroid rays -> object center + radius (also a camera check),
coarse carve (64^3) -> tight box, fine carve (192-320^3 by preset), optional depth carve,
per-view IoU of the hull reprojection vs the masks. Surface: points on the 0.5 iso-level of the
blurred occupancy with gradient normals (fed to Poisson by mesh.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from codemesh.pipeline.cameras import Camera
from codemesh.pipeline.fuse import find_mask

PRESET_RES = {"draft": 192, "standard": 256, "high": 320}
CHUNK = 1_000_000


class View:
    """One camera with its (dilated) full-res mask and, optionally, engine depth/conf."""

    def __init__(self, cam: Camera, mask: np.ndarray, dilate_px: int):
        self.cam, self.mask = cam, mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px > 0 else None
        self.test = cv2.dilate(mask.astype(np.uint8), k).astype(bool) if k is not None else mask
        E = cam.extrinsic_np
        self.R, self.t, self.K = E[:3, :3].astype(np.float32), E[:3, 3].astype(np.float32), cam.K_np.astype(np.float32)
        self.depth: np.ndarray | None = None  # engine depth (min-filtered), conf-gated: 0 = no evidence
        self.o2p: np.ndarray | None = None

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> (u, v) int original pixels, z, valid (in front and inside the image)."""
        pc = pts @ self.R.T + self.t
        z = pc[:, 2]
        front = z > 1e-6
        uvw = pc @ self.K.T
        zs = np.where(front, uvw[:, 2], 1.0)
        u, v = np.floor(uvw[:, 0] / zs).astype(np.int64), np.floor(uvw[:, 1] / zs).astype(np.int64)
        h, w = self.mask.shape
        ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u, v, z, ok


def load_views(job_dir: Path, cams: list[Camera], dilate_px: int) -> list[View]:
    views = []
    for cam in cams:
        m = cv2.imread(str(find_mask(job_dir, cam)), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m >= 128).any():
            views.append(View(cam, m >= 128, dilate_px))
    if len(views) < 3:
        raise RuntimeError(f"visual hull needs >= 3 views with a non-empty mask, got {len(views)}")
    return views


def attach_depth(job_dir: Path, views: list[View], conf_quantile: float) -> int:
    """Attach engine depth (3x3 min-filtered, below-quantile confidence zeroed) to the views. Returns count."""
    path = job_dir / "geometry" / "views.npz"
    if not path.exists():
        return 0
    with np.load(path) as npz:
        d = {k: npz[k] for k in ("frame_indices", "depth", "depth_conf", "pixel_to_original")}
    by_index = {v.cam.index: v for v in views}
    kernel = np.ones((3, 3), np.uint8)  # silhouette pixels mixing in background depth never vote "free"
    n = 0
    for i, idx in enumerate(d["frame_indices"]):
        v = by_index.get(int(idx))
        if v is None:
            continue
        depth = cv2.erode(np.nan_to_num(d["depth"][i].astype(np.float32), nan=0, posinf=0, neginf=0), kernel)
        conf = d["depth_conf"][i]
        depth[conf < np.quantile(conf, conf_quantile)] = 0
        v.depth, v.o2p = depth, np.linalg.inv(d["pixel_to_original"][i]).astype(np.float32)
        n += 1
    return n


def triangulate(views: list[View]) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares point closest to all mask-centroid rays, the object radius estimate and
    each ray's miss distance (camera-consistency check: small misses = poses agree)."""
    A, b, rays, radii = np.zeros((3, 3)), np.zeros(3), [], []
    for v in views:
        ys, xs = np.nonzero(v.mask)
        K, R = v.cam.K_np, v.cam.extrinsic_np[:3, :3]
        d = R.T @ np.linalg.solve(K, [xs.mean() + 0.5, ys.mean() + 0.5, 1.0])
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ v.cam.center
        rays.append((v.cam.center, d))
        radii.append(0.5 * np.hypot(np.ptp(xs), np.ptp(ys)) / K[0, 0])  # angular radius (rad)
    c = np.linalg.solve(A, b)
    miss = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (c - o)) for o, d in rays])
    dist = np.array([np.linalg.norm(c - o) for o, _ in rays])
    return c, float(np.median(dist * np.array(radii))), miss


def carve_points(pts: np.ndarray, views: list[View], max_outside: int, min_seen: int,
                 depth_tol: float, depth_votes: int) -> np.ndarray:
    """Survivor mask for points (N, 3) float32: inside the dilated mask in all but `max_outside`
    of the views that see them (>= min_seen such views), and (depth_tol > 0) not seen as free
    space (> depth_tol * depth in front of the engine depth) in >= depth_votes views."""
    alive = np.ones(len(pts), bool)
    out = np.zeros(len(pts), np.int16)
    seen = np.zeros(len(pts), np.int16)
    free = np.zeros(len(pts), np.int16)
    for v in views:
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        u, vv, z, ok = v.project(pts[idx])
        seen[idx] += ok
        outside = ok.copy()
        outside[ok] = ~v.test[vv[ok], u[ok]]
        out[idx] += outside
        if depth_tol > 0 and v.depth is not None:
            p = np.stack([u + 0.5, vv + 0.5, np.ones_like(z)], 1).astype(np.float32) @ v.o2p.T
            eu, ev = np.floor(p[:, 0] / p[:, 2]).astype(np.int64), np.floor(p[:, 1] / p[:, 2]).astype(np.int64)
            eh, ew = v.depth.shape
            inside = ok & (eu >= 0) & (eu < ew) & (ev >= 0) & (ev < eh)
            d = np.zeros(len(idx), np.float32)
            d[inside] = v.depth[ev[inside], eu[inside]]
            free[idx] += (d > 0) & (z < d * (1 - depth_tol))
        alive[idx] = (out[idx] <= max_outside) & (free[idx] < max(depth_votes, 1) if depth_tol > 0 else True)
    return alive & (seen >= min_seen)


def grid_centers(lo: np.ndarray, voxel: float, shape: tuple[int, int, int]) -> np.ndarray:
    ii = [lo[a] + voxel * (np.arange(shape[a], dtype=np.float32) + 0.5) for a in range(3)]
    return np.stack(np.meshgrid(*ii, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


def carve_grid(lo: np.ndarray, voxel: float, shape: tuple[int, int, int], views: list[View],
               max_outside: int, min_seen: int, depth_tol: float = 0.0, depth_votes: int = 2) -> np.ndarray:
    """Occupancy (shape) bool of the box [lo, lo + voxel * shape], carved in chunks of z-slabs."""
    occ = np.zeros(shape, bool)
    per = max(1, CHUNK // (shape[1] * shape[2]))
    for x0 in range(0, shape[0], per):
        sub = (min(per, shape[0] - x0), shape[1], shape[2])
        pts = grid_centers(lo + np.array([x0 * voxel, 0, 0]), voxel, sub)
        occ[x0:x0 + sub[0]] = carve_points(pts, views, max_outside, min_seen, depth_tol, depth_votes).reshape(sub)
    return occ


def view_iou(occ: np.ndarray, lo: np.ndarray, voxel: float, views: list[View], scale: int = 8) -> list[float]:
    """Per-view IoU of the hull's reprojection (voxel splats, closed) vs the mask at 1/scale res."""
    pts = grid_centers(lo, voxel, occ.shape)[occ.ravel()]
    ious = []
    for v in views:
        h, w = v.mask.shape
        hs, ws = h // scale, w // scale
        m = cv2.resize(v.mask.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_AREA) > 0
        u, vv, _, ok = v.project(pts)
        img = np.zeros((hs, ws), np.uint8)
        img[np.minimum(vv[ok] // scale, hs - 1), np.minimum(u[ok] // scale, ws - 1)] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        union = (img | m).sum()
        ious.append(float((img & m).sum() / union) if union else 0.0)
    return ious


def visual_hull(job_dir: Path, cams: list[Camera], res: int, dilate_px: int = 4, max_outside: int = 1,
                depth_tol: float = 0.0, depth_votes: int = 2, conf_quantile: float = 0.3,
                min_view_iou: float = 0.75, floor: tuple[np.ndarray, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """-> (occupancy bool grid, box min corner, voxel size, stats). floor = (unit up, level):
    voxels below the support plane are cleared (no camera sees under the object, so the hull
    otherwise grows a wedge there); skipped if it would remove > 25% of the hull height."""
    views = load_views(job_dir, cams, dilate_px)
    c, r, miss = triangulate(views)
    min_seen = max(3, len(views) // 2)
    st: dict = {"views": len(views), "center_ray_miss_over_radius_median": round(float(np.median(miss) / r), 3),
                "center_ray_miss_over_radius_max": round(float(miss.max() / r), 3)}
    # Coarse pass in a generous cube around the triangulated center, then a tight box.
    half = 3.0 * r
    n0 = 64
    lo = (c - half).astype(np.float32)
    vox0 = 2 * half / n0
    occ0 = largest_blob(carve_grid(lo, vox0, (n0,) * 3, views, max_outside, min_seen))  # drops stray frustum overlaps
    ix = np.argwhere(occ0)
    blo = lo + (ix.min(0) - 2) * vox0
    bhi = lo + (ix.max(0) + 3) * vox0
    voxel = float((bhi - blo).max() / res)
    shape = tuple(int(np.ceil(e / voxel)) for e in (bhi - blo))
    occ = largest_blob(carve_grid(blo, voxel, shape, views, max_outside, min_seen))
    if floor is not None:
        h = grid_centers(blo, voxel, shape) @ floor[0].astype(np.float32)
        hh = h[occ.ravel()]
        cut = floor[1] - voxel
        if hh.min() < cut and (cut - hh.min()) < 0.25 * np.ptp(hh):
            occ = largest_blob(occ & (h >= cut).reshape(shape))
            st["floor_clipped"] = round(float(cut - hh.min()), 5)
    st["voxels_hull"] = int(occ.sum())
    ious = view_iou(occ, blo, voxel, views)
    st.update(grid=list(shape), voxel=voxel, iou_mean=round(float(np.mean(ious)), 4),
              iou_min=round(float(np.min(ious)), 4), iou_per_view={v.cam.index: round(i, 3) for v, i in zip(views, ious)})
    if depth_tol > 0:
        # Only views whose silhouette agrees with the hull may carve by depth: a bad pose would cut the object.
        good = [v for v, i in zip(views, ious) if i >= min_view_iou]
        st["depth_views"] = attach_depth(job_dir, good, conf_quantile)
        if st["depth_views"]:
            carved = largest_blob(carve_grid(blo, voxel, shape, good, max_outside, max(3, len(good) // 2),
                                             depth_tol, depth_votes) & occ)
            if carved.sum() >= 0.5 * occ.sum():  # depth disagreeing wholesale with the hull is noise, not shape
                occ = carved
        st["voxels_after_depth"] = int(occ.sum())
    return occ, blo, voxel, st


def largest_blob(occ: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(occ)
    if n > 1:
        occ = lab == 1 + int(np.argmax(ndimage.sum(occ, lab, range(1, n + 1))))
    if not occ.any():
        raise RuntimeError("visual hull empty after carving")
    return occ


def surface_points(occ: np.ndarray, lo: np.ndarray, voxel: float, sigma: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Points on the 0.5 iso-level of the blurred occupancy (both sides of the boundary) and
    outward normals (-gradient). Input for Poisson."""
    pad = np.pad(occ, 3)
    b = ndimage.gaussian_filter(pad.astype(np.float32), sigma)
    g = np.stack(np.gradient(b), -1)
    boundary = pad ^ ndimage.binary_erosion(pad)
    boundary |= ndimage.binary_dilation(pad) & ~pad
    ix = np.argwhere(boundary)
    gi = g[tuple(ix.T)]
    gn = np.linalg.norm(gi, axis=1)
    keep = gn > 1e-4
    ix, gi, gn = ix[keep], gi[keep], gn[keep]
    n = -gi / gn[:, None]
    s = np.clip((b[tuple(ix.T)] - 0.5) / gn, -1.0, 1.0)  # voxels along the normal to the iso-level
    pts = lo + voxel * (ix - 3 + 0.5 + s[:, None] * n)
    return pts.astype(np.float64), n.astype(np.float64)
