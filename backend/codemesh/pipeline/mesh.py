"""Mesh stage: fused points -> dense surface (mesh/high.ply) -> game mesh (mesh/low.ply).

mesh_method "hull_depth" (default): visual hull voxel-carved from the full-res masks, free-space
carved by confident engine depth, surfaced by Poisson on its iso-level points (see hull.py);
robust when the object is small in frame and engine depth on it is sparse.
mesh_method "poisson": screened Poisson on fused points (depth by preset), low-density trim, visual-hull carve (drop faces that
project outside the full-res masks in >= k views), keep large components, fill small holes,
manifold cleanup. low: progressive quadric decimation until the mean high->low distance
exceeds a tolerance (budget "auto") or to an explicit triangle count, optional Taubin smoothing
when it lowers the error. Both meshes are then moved into a gravity-aligned object frame
(+Y up from the average camera up, base on y=0, centered in x/z, optional real-world scale from
`known_dimension`), and that transform is recorded in mesh/transform.json
(x_mesh = matrix @ x_world) so texture/export can map cameras into the same frame.

Settings: mesh_method (hull_depth|poisson), hull_resolution (192|256|320 by preset), hull_dilate_px (4),
hull_max_outside (1), hull_depth_tol (0.03, 0 = off), hull_depth_votes (2), hull_taubin (5), hull_floor (true: clip below the
lowest fused points),
preset (draft|standard|high), triangle_budget ("auto"|int), known_dimension
({axis: x|y|z, meters}), poisson_depth, trim_quantile (0.03), carve_views (2),
carve_dilate_px (3), carve_depth_tol (0.02), min_component_ratio (0.1), hole_size (0.03 of the diagonal).
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

from codemesh.pipeline import hull
from codemesh.pipeline.cameras import Camera, load_cameras
from codemesh.pipeline.fuse import find_mask

Progress = Callable[[float, str], None]
Mesh = o3d.geometry.TriangleMesh

PRESET_DEPTH = {"draft": 8, "standard": 9, "high": 10}
PRESET_TOL = {"draft": 0.004, "standard": 0.002, "high": 0.001}  # mean high->low distance / diagonal
BUDGETS = (150_000, 100_000, 70_000, 50_000, 35_000, 25_000, 18_000, 12_000, 8_000)  # 'auto' floor 8k
ERROR_SAMPLES = 60_000


def diagonal(mesh: Mesh) -> float:
    return float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))


def clean(mesh: Mesh) -> Mesh:
    """Remove degenerate/duplicate faces, non-manifold edges and unreferenced vertices (in place)."""
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def poisson(pcd: o3d.geometry.PointCloud, depth: int, trim_quantile: float) -> Mesh:
    """Screened Poisson surface, vertices below the density quantile removed."""
    mesh, dens = Mesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1, n_threads=-1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_quantile))
    return mesh


def _engine_depths(job_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """frame index -> (engine depth min-filtered 3x3, original->engine pixel map) from geometry/views.npz."""
    path = job_dir / "geometry" / "views.npz"
    if not path.exists():
        return {}
    with np.load(path) as npz:
        views = {k: npz[k] for k in ("frame_indices", "depth", "pixel_to_original")}
    kernel = np.ones((3, 3), np.uint8)  # min-filter: silhouette pixels mixing in background depth never vote "free"
    return {int(idx): (cv2.erode(np.nan_to_num(views["depth"][i].astype(np.float32), nan=0, posinf=0, neginf=0), kernel),
                       np.linalg.inv(views["pixel_to_original"][i]))
            for i, idx in enumerate(views["frame_indices"])}


def carve(mesh: Mesh, job_dir: Path, cams: list[Camera], k: int, dilate_px: int, depth_tol: float = 0.0) -> int:
    """Visual-hull + free-space carve in place. A face votes "outside" in a camera when its centroid
    is in front of it and projects inside the image but outside the (dilated) object mask, or (with
    depth_tol > 0) lies more than depth_tol * depth in front of the engine depth seen there: the
    camera saw through that spot, so it is empty space. Faces with >= k votes are dropped.
    The depth test catches blobs glued to the object that stay inside every silhouette.
    Returns faces removed."""
    tris = np.asarray(mesh.triangles)
    cent = np.asarray(mesh.vertices)[tris].mean(axis=1)
    outside = np.zeros(len(cent), np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px > 0 else None
    depths = _engine_depths(job_dir) if depth_tol > 0 else {}
    for cam in cams:
        mask = cv2.imread(str(find_mask(job_dir, cam)), cv2.IMREAD_GRAYSCALE) >= 128
        if kernel is not None:  # tolerate mask boundary error before calling a face "outside"
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        E, K = cam.extrinsic_np, cam.K_np
        pc = cent @ E[:3, :3].T + E[:3, 3]
        front = pc[:, 2] > 1e-6
        uvw = pc[front] @ K.T
        uv = np.rint(uvw[:, :2] / uvw[:, 2:]).astype(np.int64)
        h, w = mask.shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(front)[inside]
        vote = ~mask[uv[inside, 1], uv[inside, 0]]
        if cam.index in depths:
            depth, o2p = depths[cam.index]
            ep = uvw[inside] @ o2p.T
            eu, ev = (np.rint(ep[:, :2] / ep[:, 2:]).astype(np.int64)).T
            eh, ew = depth.shape
            ok = (eu >= 0) & (eu < ew) & (ev >= 0) & (ev < eh)
            d = np.zeros(len(idx), np.float32)
            d[ok] = depth[ev[ok], eu[ok]]
            vote |= (d > 0) & (pc[idx, 2] < d * (1 - depth_tol))
        outside[idx] += vote
    drop = outside >= k
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return int(drop.sum())


def hull_depth(job_dir: Path, cams: list[Camera], pcd: o3d.geometry.PointCloud, settings: dict[str, Any],
               preset: str) -> tuple[Mesh, dict[str, Any]]:
    """Visual hull from the full-res masks, free-space carved by confident engine depth, surfaced
    with Poisson on its iso-level points, Taubin-smoothed; vertex colors from the nearest fused point."""
    floor = None
    if settings.get("hull_floor", True) and len(pcd.points):  # support plane: lowest fused points
        up = -np.mean([c.extrinsic_np[1, :3] for c in cams], axis=0)
        up /= np.linalg.norm(up)
        floor = (up, float(np.quantile(np.asarray(pcd.points) @ up, 0.005)))
    occ, lo, voxel, st = hull.visual_hull(
        job_dir, cams, int(settings.get("hull_resolution", hull.PRESET_RES[preset])),
        dilate_px=int(settings.get("hull_dilate_px", 4)), max_outside=int(settings.get("hull_max_outside", 1)),
        depth_tol=float(settings.get("hull_depth_tol", 0.03)), depth_votes=int(settings.get("hull_depth_votes", 2)),
        floor=floor)
    pts, nrm = hull.surface_points(occ, lo, voxel)
    surf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    surf.normals = o3d.utility.Vector3dVector(nrm)
    mesh = poisson(surf, int(settings.get("poisson_depth", PRESET_DEPTH[preset] - 1)), 0.01)  # ~ hull grid res
    mesh = mesh.filter_smooth_taubin(number_of_iterations=int(settings.get("hull_taubin", 5)))
    if pcd.has_colors() and len(pcd.points):
        _, nn = cKDTree(np.asarray(pcd.points)).query(np.asarray(mesh.vertices))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[nn])
    mesh.compute_vertex_normals()
    return mesh, {"hull": {k: v for k, v in st.items() if k != "iou_per_view"}, "hull_iou_per_view": st["iou_per_view"]}


def keep_large_components(mesh: Mesh, min_ratio: float) -> int:
    """Keep connected components with >= min_ratio * largest component's faces. Returns faces dropped."""
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    drop = counts[labels] < min_ratio * counts.max()
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return int(drop.sum())


def fill_holes(mesh: Mesh, hole_size: float) -> int:
    """Fan-fill boundary loops whose vertices all lie within `hole_size` of the loop centroid
    (in place; new center vertex gets the mean loop color). Returns holes filled.
    Own implementation: Open3D's VTK fill_holes crashes (access violation) on Poisson output."""
    tris = np.asarray(mesh.triangles)
    he = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    n = len(mesh.vertices)
    boundary = he[~np.isin(he[:, 1] * n + he[:, 0], he[:, 0] * n + he[:, 1])]
    # The hole runs opposite to the triangles' half-edges: walk b -> a.
    nxt: dict[int, int] = {}
    ambiguous: set[int] = set()
    for a, b in boundary.tolist():
        if b in nxt:
            ambiguous.add(b)
        nxt[b] = a
    verts = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    new_v, new_c, new_t, seen, filled = [], [], [], set(), 0
    for start in list(nxt):
        if start in seen:
            continue
        loop, v = [start], nxt[start]
        while v != start and v in nxt and v not in seen and len(loop) <= len(boundary):
            loop.append(v)
            v = nxt[v]
        seen.update(loop)
        if v != start or len(loop) < 3 or ambiguous.intersection(loop):
            continue
        center = verts[loop].mean(0)
        if np.linalg.norm(verts[loop] - center, axis=1).max() > hole_size:
            continue
        c = n + len(new_v)
        new_v.append(center)
        if colors is not None:
            new_c.append(colors[loop].mean(0))
        new_t += [(loop[i], loop[(i + 1) % len(loop)], c) for i in range(len(loop))]
        filled += 1
    if filled:
        mesh.vertices = o3d.utility.Vector3dVector(np.vstack([verts, new_v]))
        mesh.triangles = o3d.utility.Vector3iVector(np.vstack([tris, new_t]).astype(np.int32))
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, new_c]))
        mesh.compute_vertex_normals()
    return filled


def distances(src: Mesh, dst: Mesh, n: int = ERROR_SAMPLES, seed: int = 0) -> np.ndarray:
    """Distances from n uniform surface samples of `src` to the surface of `dst`."""
    o3d.utility.random.seed(seed)
    pts = np.asarray(src.sample_points_uniformly(n).points, dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dst))
    return scene.compute_distance(o3d.core.Tensor(pts)).numpy()


def decimate(high: Mesh, budget: str | int, tol: float) -> tuple[Mesh, dict[str, Any]]:
    """Game mesh from `high`. budget "auto": step down BUDGETS while mean high->low distance
    <= tol (absolute units); clamp to [BUDGETS[-1], BUDGETS[0]]. int: decimate to that count."""
    n_high = len(high.triangles)
    simplify = lambda m, n: clean(m.simplify_quadric_decimation(n, boundary_weight=100.0))  # noqa: E731
    err = lambda m: float(distances(high, m).mean())  # noqa: E731
    if budget != "auto":
        low = simplify(high, int(budget)) if int(budget) < n_high else Mesh(high)
    else:
        low, cur = Mesh(high), high
        for n in BUDGETS:
            if n >= n_high:
                continue
            cand = simplify(cur, n)  # progressive: each step decimates the previous result
            if err(cand) > tol and len(low.triangles) <= BUDGETS[0]:
                break
            low = cur = cand
    base = err(low)
    smooth = low.filter_smooth_taubin(number_of_iterations=5)
    smoothed = err(smooth) < base
    if smoothed:
        low = smooth
    return low, {"taubin": smoothed}


def gravity_transform(cams: list[Camera], high: Mesh, low: Mesh, known: dict | None) -> tuple[np.ndarray, dict]:
    """World -> object frame: rotate the mean camera up (-y axis) to +Y, base to y=0, center x/z,
    scale so `known` = {axis, meters} matches the low mesh extent along that axis."""
    up = -np.mean([c.extrinsic_np[1, :3] for c in cams], axis=0)
    up /= np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    axis, c = np.cross(up, y), float(up @ y)
    if np.linalg.norm(axis) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:  # Rodrigues rotation taking `up` onto +Y
        vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    pts = np.asarray(high.vertices) @ R.T
    lo, hi = pts.min(0), pts.max(0)
    t = -np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    scale = 1.0
    if known:
        ax = "xyz".index(known["axis"])
        ext = np.ptp(np.asarray(low.vertices) @ R.T, axis=0)[ax]
        scale = float(known["meters"]) / float(ext)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = scale * R, scale * t
    return M, {"up_world": up.tolist(), "scale": scale, "known_dimension": known}


def run(job_dir: Path, settings: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    """Build mesh/high.ply, mesh/low.ply and mesh/transform.json. Returns metrics."""
    t0 = time.perf_counter()
    report = progress or (lambda f, m: None)
    preset = settings.get("preset", "standard")
    if preset not in PRESET_DEPTH:
        raise ValueError(f"unknown preset {preset!r}")
    budget = settings.get("triangle_budget", "auto")
    if budget != "auto" and int(budget) < 4:
        raise ValueError(f"triangle_budget must be 'auto' or an int >= 4, got {budget!r}")
    cams, _ = load_cameras(job_dir / "geometry" / "cameras.json")
    pcd = o3d.io.read_point_cloud(str(job_dir / "fuse" / "points.ply"))
    if not pcd.has_normals():
        raise RuntimeError("fuse/points.ply has no normals")
    method = settings.get("mesh_method", "hull_depth")
    if method not in ("hull_depth", "poisson"):
        raise ValueError(f"mesh_method must be hull_depth|poisson, got {method!r}")
    m: dict[str, Any] = {"method": method}

    if method == "hull_depth":
        report(0.05, "visual hull + depth carve")
        high, info = hull_depth(job_dir, cams, pcd, settings, preset)
        m.update(info, tris_poisson=len(high.triangles))
    else:
        report(0.05, "poisson surface")
        high = poisson(pcd, int(settings.get("poisson_depth", PRESET_DEPTH[preset])),
                       float(settings.get("trim_quantile", 0.03)))
        m["tris_poisson"] = len(high.triangles)
        report(0.35, "visual-hull carve")
        m["faces_carved"] = carve(high, job_dir, cams, int(settings.get("carve_views", 2)),
                                  int(settings.get("carve_dilate_px", 3)),
                                  float(settings.get("carve_depth_tol", 0.02)))
    m["faces_small_components"] = keep_large_components(high, float(settings.get("min_component_ratio", 0.1)))
    if not len(high.triangles):
        raise RuntimeError("mesh: nothing left after carving; masks and cameras disagree")
    diag = diagonal(high)
    m["holes_filled"] = fill_holes(clean(high), float(settings.get("hole_size", 0.03)) * diag)
    high = clean(high)
    m["tris_high"] = len(high.triangles)

    report(0.55, "decimating")
    tol = PRESET_TOL[preset] * diag
    low, info = decimate(high, budget, tol)
    m.update(info, tris_low=len(low.triangles), budget=budget)
    d_hl, d_lh = distances(high, low), distances(low, high)
    m["mean_high_to_low_pct"] = round(100 * float(d_hl.mean()) / diag, 4)
    m["chamfer_pct"] = round(100 * float(d_hl.mean() + d_lh.mean()) / 2 / diag, 4)
    m["hausdorff_pct"] = round(100 * float(max(d_hl.max(), d_lh.max())) / diag, 4)

    report(0.9, "aligning")
    M, tinfo = gravity_transform(cams, high, low, settings.get("known_dimension"))
    high.transform(M)
    low.transform(M)
    out = job_dir / "mesh"
    out.mkdir(parents=True, exist_ok=True)
    (out / "transform.json").write_text(json.dumps({"matrix": M.tolist(), **tinfo}, indent=1))
    o3d.io.write_triangle_mesh(str(out / "high.ply"), high)
    o3d.io.write_triangle_mesh(str(out / "low.ply"), low)
    m["seconds"] = round(time.perf_counter() - t0, 2)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    report(1.0, f"{m['tris_high']} -> {m['tris_low']} triangles")
    return m
