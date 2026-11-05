"""Texture stage: UV atlas + albedo projected from the full-res photos + tangent-space normal map.

Inputs (job dir): mesh/low.ply (game mesh), mesh/high.ply (dense source), mesh/transform.json
(world -> asset 4x4; identity if missing), geometry/cameras.json (world frame), the frames and
masks/<index>.png. Both meshes are in the asset frame, so cameras become E_asset = E_world @ inv(T).

Steps:
  1. xatlas UV unwrap of the low mesh (resolution = texture size, padding scales with it).
  2. Rasterize the atlas: per covered texel the triangle id + barycentrics -> 3D position, normal,
     tangent frame (flat arrays over covered texels; all NumPy, chunked).
  3. Normal bake: ray from each texel along +/- the low normal onto the high mesh; the high mesh's
     smooth normal at the hit is stored in the low mesh's tangent frame (glTF: +Y green). The hit
     point is also the position used for color projection (closer to what the photos saw).
  4. Per view: project into the full-res camera; visible if inside the image, inside the eroded
     mask, facing the camera and not occluded (z vs. a ray-cast depth map of the high mesh at
     reduced resolution). Weight = cos^p * (f/z)^2 * sharpness^2; keep the top-k views per
     texel, optionally solve per-view RGB gains (log least squares on texels seen twice) to hide
     exposure seams, then blend. Colors are bilinear samples of the original frames.
  5. Unseen texels and the atlas gutter are filled by push-pull, so mip levels do not bleed.

Outputs (texture/): albedo.png, normal.png (8-bit RGB, requested size), uv_mesh.npz (vertices,
faces, uv (v up, trimesh convention), normals, tangents (n, 4) with handedness in w).
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
import trimesh
import xatlas

from codemesh.pipeline.cameras import Camera, load_cameras

Progress = Callable[[float, str], None] | None

DEFAULTS: dict[str, Any] = {
    "texture_size": 2048,
    "top_k": 3,
    "angle_power": 4.0,  # grazing views smear the background into the texture
    "min_cos": 0.15,  # reject views more grazing than ~81 deg
    "exposure_normalize": True,
    "depth_res": 800,  # long side of the visibility depth maps
    "occlusion_tol": 0.005,  # fraction of the bbox diagonal
    "bake_max_dist": 0.03,  # fraction of the bbox diagonal (low <-> high search distance)
    "mask_erode": 0.004,  # fraction of the image diagonal: silhouette pixels are mixed with the background
}


def load_transform(job_dir: Path) -> np.ndarray:
    """World -> asset 4x4 from mesh/transform.json (key world_to_asset|matrix|transform)."""
    path = job_dir / "mesh" / "transform.json"
    if not path.exists():
        return np.eye(4)
    data = json.loads(path.read_text())
    for key in ("world_to_asset", "matrix", "transform"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(4, 4)
    raise ValueError(f"{path}: no world_to_asset/matrix/transform key")


def unwrap(vertices: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas atlas for a `size`^2 texture. Returns (vmapping, faces, uv) with uv in [0, 1], v up."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = max(2, size // 512)
    pack.bilinear = True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmapping, new_faces, uv = atlas[0]
    return vmapping.astype(np.int64), new_faces.astype(np.int64), uv.astype(np.float64)


def uv_to_pixel(uv: np.ndarray, size: int) -> np.ndarray:
    """UV (v up) -> texel coords (x, y) with texel centers at integers, row 0 = top (v = 1)."""
    return np.stack([uv[..., 0] * size - 0.5, (1.0 - uv[..., 1]) * size - 0.5], -1)


def rasterize_uv(uv: np.ndarray, faces: np.ndarray, size: int, chunk: int = 1 << 23) -> tuple[np.ndarray, np.ndarray]:
    """Per texel: triangle id (-1 = empty) and barycentrics (size, size, 3), via vectorized
    bounding-box enumeration in chunks of ~`chunk` candidate texels."""
    tri = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    p = uv_to_pixel(uv, size)[faces]  # (F, 3, 2)
    x0, y0, x1, y1, x2, y2 = (p[:, i, j] for i in range(3) for j in range(2))
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    lo = np.clip(np.ceil(p.min(1)), 0, size - 1).astype(np.int64)
    hi = np.clip(np.floor(p.max(1)), 0, size - 1).astype(np.int64)
    bw, bh = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    counts = np.where((np.abs(den) > 1e-12) & (bw > 0) & (bh > 0), bw * bh, 0)
    ends = np.cumsum(counts)
    start = 0
    while start < len(faces):
        stop = int(np.searchsorted(ends, (ends[start - 1] if start else 0) + chunk, "right"))
        stop = max(stop, start + 1)
        ids = np.arange(start, stop)
        c = counts[ids]
        t = np.repeat(ids, c)
        local = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        px, py = lo[t, 0] + local % bw[t], lo[t, 1] + local // bw[t]
        l0 = ((y1[t] - y2[t]) * (px - x2[t]) + (x2[t] - x1[t]) * (py - y2[t])) / den[t]
        l1 = ((y2[t] - y0[t]) * (px - x2[t]) + (x0[t] - x2[t]) * (py - y2[t])) / den[t]
        l2 = 1.0 - l0 - l1
        ok = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        tri[py[ok], px[ok]] = t[ok]
        bary[py[ok], px[ok]] = np.stack([l0[ok], l1[ok], l2[ok]], -1)
        start = stop
    return tri, bary


def vertex_tangents(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (n, 4) from UVs: per-triangle dP/du accumulation, Gram-Schmidt against
    the normal, handedness w = sign(dot(cross(N, T), dP/dv)). Bitangent = cross(N, T) * w."""
    p, t = vertices[faces], uv[faces]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    T, B = np.zeros_like(vertices), np.zeros_like(vertices)
    for k in range(3):
        np.add.at(T, faces[:, k], tan)
        np.add.at(B, faces[:, k], bit)
    T -= normals * (normals * T).sum(1, keepdims=True)
    bad = np.linalg.norm(T, axis=1) < 1e-12
    if bad.any():  # no UV gradient: any vector perpendicular to the normal
        alt = np.where(np.abs(normals[bad, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        T[bad] = np.cross(normals[bad], alt)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    w = np.where((np.cross(normals, T) * B).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([T, w[:, None]], 1)


def camera_rays(K: np.ndarray, E: np.ndarray, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    """(h, w, 6) float32 rays of a camera sampled at `scale` (pixel-center aligned). Directions
    have camera z = 1, so t_hit equals z-depth in camera units even if E contains a scale."""
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    u, v = np.meshgrid((np.arange(w) + 0.5) / scale - 0.5, (np.arange(h) + 0.5) / scale - 0.5)
    dirs_cam = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    Rinv = np.linalg.inv(E[:3, :3])
    dirs = dirs_cam @ Rinv.T
    origin = np.broadcast_to(-Rinv @ E[:3, 3], dirs.shape)
    return np.concatenate([origin, dirs], -1).astype(np.float32)


def raycaster(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    return scene


def cast(scene: o3d.t.geometry.RaycastingScene, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_hit, primitive id, barycentrics (.., 3) for the triangle's vertices 0, 1, 2)."""
    ans = scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))
    t = ans["t_hit"].numpy()
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    buv = ans["primitive_uvs"].numpy()
    prim[~np.isfinite(t)] = 0
    return t, prim, np.concatenate([1 - buv.sum(-1, keepdims=True), buv], -1)


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear lookup of `img` at float pixel coords (any count; cv2.remap needs < 32k per axis)."""
    n = len(x)
    rows = -(-n // width) if n else 0
    mx = np.zeros(rows * width, np.float32)
    my = np.zeros(rows * width, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(rows, width), my.reshape(rows, width), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE) if n else img[:0]
    return out.reshape(rows * width, *img.shape[2:])[:n]


def push_pull(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill pixels with weight 0 from a pyramid of weighted averages of the known pixels."""
    if min(img.shape[:2]) <= 1 or weight.min() >= 1:
        return img
    h, w = img.shape[:2]
    small = ((w + 1) // 2, (h + 1) // 2)
    wsum = cv2.resize(weight, small, interpolation=cv2.INTER_AREA)
    csum = cv2.resize(img * weight[..., None], small, interpolation=cv2.INTER_AREA)
    coarse = push_pull(csum / np.maximum(wsum, 1e-8)[..., None], np.minimum(1.0, 4 * wsum).astype(np.float32))
    up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    return img * weight[..., None] + up * (1 - weight[..., None])


def frame_sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 640 / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def load_mask(job_dir: Path, cam: Camera) -> np.ndarray | None:
    for name in (f"{cam.index:04d}.png", f"{cam.index}.png"):
        if (path := job_dir / "masks" / name).exists():
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None


def solve_gains(colors: np.ndarray, weights: np.ndarray, views: np.ndarray, n_views: int,
                max_samples: int = 300_000, seed: int = 0) -> np.ndarray:
    """Per-view RGB gains g (n_views, 3) so that g_a * c_a ~= g_b * c_b on texels seen by several
    views (log-domain least squares, weak prior g = 1)."""
    rng = np.random.default_rng(seed)
    multi = np.flatnonzero(weights[:, 1] > 0)
    if len(multi) == 0:
        return np.ones((n_views, 3))
    multi = rng.choice(multi, min(len(multi), max_samples), replace=False)
    L = np.zeros((n_views, n_views))
    b = np.zeros((n_views, 3))
    k = weights.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            sel = multi[(weights[multi, j] > 0) & (colors[multi, i].min(1) > 8) & (colors[multi, j].min(1) > 8)]
            a, c = views[sel, i], views[sel, j]
            diff = np.log(colors[sel, j]) - np.log(colors[sel, i])  # g_a - g_c = log c_c - log c_a
            np.add.at(L, (a, a), 1)
            np.add.at(L, (c, c), 1)
            np.add.at(L, (a, c), -1)
            np.add.at(L, (c, a), -1)
            np.add.at(b, a, diff)
            np.add.at(b, c, -diff)
    reg = 1e-3 * max(1.0, np.trace(L) / n_views)
    g = np.linalg.solve(L + reg * np.eye(n_views), b)
    return np.clip(np.exp(g - g.mean(0)), 0.5, 2.0)


def run(job_dir: Path, settings: dict, progress: Progress = None) -> dict[str, Any]:
    """Texture the low mesh. Settings: see DEFAULTS (texture_size 2048|4096, ...). Returns metrics."""
    t0 = time.perf_counter()
    cfg = {**DEFAULTS, **settings}
    size = int(cfg["texture_size"])
    report = progress or (lambda f, m: None)
    out = job_dir / "texture"
    out.mkdir(parents=True, exist_ok=True)

    low = trimesh.load(job_dir / "mesh" / "low.ply", process=False)
    high = trimesh.load(job_dir / "mesh" / "high.ply", process=False)
    T = load_transform(job_dir)
    cams, _ = load_cameras(job_dir / "geometry" / "cameras.json")
    diag = float(np.linalg.norm(high.bounds[1] - high.bounds[0]))

    report(0.02, "UV unwrap")
    vmap, faces, uv = unwrap(np.asarray(low.vertices), np.asarray(low.faces), size)
    V = np.asarray(low.vertices, np.float64)[vmap]
    N = np.asarray(low.vertex_normals, np.float64)[vmap]
    tangents = vertex_tangents(V, N, faces, uv)
    t_unwrap = time.perf_counter() - t0

    report(0.15, "rasterize atlas")
    tri_map, bary_map = rasterize_uv(uv, faces, size)
    ys, xs = np.nonzero(tri_map >= 0)
    tri, bary = tri_map[ys, xs], bary_map[ys, xs].astype(np.float64)
    del tri_map, bary_map
    f = faces[tri]
    P = np.einsum("mk,mkd->md", bary, V[f])
    Nl = np.einsum("mk,mkd->md", bary, N[f])
    Nl /= np.maximum(np.linalg.norm(Nl, axis=1, keepdims=True), 1e-12)
    Tl = np.einsum("mk,mkd->md", bary, tangents[f, :3])
    Tl -= Nl * (Nl * Tl).sum(1, keepdims=True)
    Tl /= np.maximum(np.linalg.norm(Tl, axis=1, keepdims=True), 1e-12)
    Bl = np.cross(Nl, Tl) * np.sign(np.einsum("mk,mk->m", bary, tangents[f, 3]) + 1e-9)[:, None]

    report(0.25, "bake normals")
    hv, hf = np.asarray(high.vertices, np.float64), np.asarray(high.faces)
    hn = np.asarray(high.vertex_normals, np.float64)
    scene = raycaster(hv, hf)
    max_d = cfg["bake_max_dist"] * diag
    t_best = np.full(len(P), np.inf)
    sign = np.ones(len(P))
    prim = np.zeros(len(P), np.int64)
    hb = np.zeros((len(P), 3))
    for s in (1.0, -1.0):
        t, pr, b = cast(scene, np.concatenate([P, s * Nl], 1))
        better = (t < t_best) & (t <= max_d)
        t_best[better], sign[better], prim[better], hb[better] = t[better], s, pr[better], b[better]
    hit = np.isfinite(t_best)
    P_hi = P.copy()
    P_hi[hit] += (sign[hit] * t_best[hit])[:, None] * Nl[hit]
    n_hi = Nl.copy()
    nh = np.einsum("mk,mkd->md", hb[hit], hn[hf[prim[hit]]])
    n_hi[hit] = nh / np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-12)
    ts = np.stack([(n_hi * Tl).sum(1), (n_hi * Bl).sum(1), (n_hi * Nl).sum(1)], 1)
    del Tl, Bl
    t_bake = time.perf_counter() - t0 - t_unwrap

    report(0.35, "project views")
    k = int(cfg["top_k"])
    best_w = np.zeros((len(P), k), np.float32)
    best_c = np.zeros((len(P), k, 3), np.float32)
    best_v = np.zeros((len(P), k), np.int16)
    tol = np.where(hit, cfg["occlusion_tol"] * diag, max_d)
    Tinv = np.linalg.inv(T)
    views_used = 0
    for vi, cam in enumerate(cams):
        report(0.35 + 0.5 * vi / len(cams), f"project view {vi + 1}/{len(cams)}")
        img = cv2.imread(str(job_dir / cam.image), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        K = cam.K_np
        E = cam.extrinsic_np @ Tinv
        C = np.linalg.inv(E[:3, :3]) @ -E[:3, 3]
        xc = P_hi @ E[:3, :3].T + E[:3, 3]
        z = xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (K[0, 0] * xc[:, 0] + K[0, 1] * xc[:, 1]) / z + K[0, 2]
            y = K[1, 1] * xc[:, 1] / z + K[1, 2]
        vis = (z > 0) & (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        view = C - P_hi
        cos = (n_hi * view).sum(1) / np.maximum(np.linalg.norm(view, axis=1), 1e-12)
        vis &= cos > cfg["min_cos"]
        mask = load_mask(job_dir, cam)
        if mask is not None:
            r = max(1, round(cfg["mask_erode"] * np.hypot(W, H)))
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            idx = np.flatnonzero(vis)
            vis[idx] = mask[np.rint(y[idx]).astype(int), np.rint(x[idx]).astype(int)] > 127
        idx = np.flatnonzero(vis)
        s = min(1.0, cfg["depth_res"] / max(W, H))
        rays = camera_rays(K, E, W, H, s)
        depth, _, _ = cast(scene, rays.reshape(-1, 6))
        depth = np.where(np.isfinite(depth), depth, 1e10).reshape(rays.shape[:2]).astype(np.float32)
        depth = cv2.erode(depth, np.ones((3, 3), np.uint8))  # conservative near occluding edges
        d = sample_bilinear(depth, s * (x[idx] + 0.5) - 0.5, s * (y[idx] + 0.5) - 0.5)
        idx = idx[z[idx] <= d + tol[idx]]
        if len(idx) == 0:
            continue
        views_used += 1
        w = (cos[idx] ** cfg["angle_power"]) * (K[0, 0] / z[idx] / 1000) ** 2 * (frame_sharpness(img) + 1e-6) ** 2
        col = sample_bilinear(img, x[idx], y[idx])[:, ::-1].astype(np.float32)  # BGR -> RGB
        all_w = np.concatenate([best_w[idx], w[:, None].astype(np.float32)], 1)
        all_c = np.concatenate([best_c[idx], col[:, None]], 1)
        all_v = np.concatenate([best_v[idx], np.full((len(idx), 1), vi, np.int16)], 1)
        order = np.argsort(-all_w, axis=1)[:, :k]
        best_w[idx] = np.take_along_axis(all_w, order, 1)
        best_c[idx] = np.take_along_axis(all_c, order[..., None], 1)
        best_v[idx] = np.take_along_axis(all_v, order, 1)

    report(0.88, "blend + fill")
    gains = np.ones((len(cams), 3))
    if cfg["exposure_normalize"] and len(cams) > 1:
        gains = solve_gains(best_c, best_w, best_v, len(cams))
        best_c *= gains[best_v].astype(np.float32)
    wsum = best_w.sum(1)
    seen = wsum > 0
    color = (best_c * best_w[..., None]).sum(1) / np.maximum(wsum, 1e-12)[:, None]

    albedo = np.zeros((size, size, 3), np.float32)
    known = np.zeros((size, size), np.float32)
    albedo[ys[seen], xs[seen]] = color[seen]
    known[ys[seen], xs[seen]] = 1
    albedo = np.clip(push_pull(albedo, known), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "albedo.png"), albedo[..., ::-1])

    nmap = np.zeros((size, size, 3), np.float32)
    nmap[..., 2] = 1
    covered = np.zeros((size, size), np.float32)
    nmap[ys, xs] = ts
    covered[ys, xs] = 1
    nmap = push_pull(nmap, covered)
    nmap /= np.maximum(np.linalg.norm(nmap, axis=2, keepdims=True), 1e-12)
    nmap_u8 = np.clip(np.rint((nmap * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "normal.png"), nmap_u8[..., ::-1])

    np.savez(out / "uv_mesh.npz", vertices=V.astype(np.float32), faces=faces.astype(np.int32),
             uv=uv.astype(np.float32), normals=N.astype(np.float32), tangents=tangents.astype(np.float32))
    report(1.0, "texture done")
    m = len(P)
    return {
        "texture_size": size,
        "triangles": len(faces),
        "charts_texels": m,
        "atlas_coverage_pct": round(100 * m / size**2, 2),
        "seen_pct": round(100 * float(seen.mean()), 2) if m else 0.0,
        "unseen_pct": round(100 * float(1 - seen.mean()), 2) if m else 0.0,
        "bake_hit_pct": round(100 * float(hit.mean()), 2) if m else 0.0,
        "views_used": views_used,
        "gain_range": [round(float(gains.min(initial=1)), 3), round(float(gains.max(initial=1)), 3)],
        "seconds_unwrap": round(t_unwrap, 2),
        "seconds_bake": round(t_bake, 2),
        "seconds": round(time.perf_counter() - t0, 2),
    }
