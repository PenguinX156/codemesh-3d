"""Clean, evidence-fitted base meshes: instead of keeping the reconstructed surface (noisy hull/depth
mesh), rebuild the object from simple modeling primitives fitted to it.

Evidence = mesh/high.ply (visual hull refined by depth, already in the gravity-aligned asset frame,
+Y up). Every generator is watertight by construction; parts are separate closed bodies in one mesh
(no booleans). high.ply stays as the detail source for normal-map baking; low.ply becomes the
primitive mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import gaussian_filter1d, median_filter

RINGS = 48       # height slices of the lathe profile
ANGLE_BINS = 36  # per slice: outer radius per angle bin, median over bins (robust to a handle)


def _sample(mesh: trimesh.Trimesh, n: int = 40000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.asarray(pts)


def _kasa(xz: np.ndarray) -> np.ndarray:
    A = np.c_[xz, np.ones(len(xz))]
    a, b, _ = np.linalg.lstsq(A, -(xz ** 2).sum(1), rcond=None)[0]
    return np.array([-a / 2, -b / 2])


def fit_profile(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (axis centre in x/z, slice heights, outer radius per slice). Two passes: points far
    outside the first profile (a handle, a spout) are dropped before refitting the axis."""
    centre = _kasa(pts[:, [0, 2]])
    y = pts[:, 1]
    edges = np.linspace(y.min(), y.max(), RINGS + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.searchsorted(edges, y) - 1, 0, RINGS - 1)
    for _ in range(2):
        d = pts[:, [0, 2]] - centre
        r = np.linalg.norm(d, axis=1)
        ab = ((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * ANGLE_BINS).astype(int) % ANGLE_BINS
        prof = np.full(RINGS, np.nan)
        for i in range(RINGS):
            s = idx == i
            if s.sum() < 10:
                continue
            outer = np.full(ANGLE_BINS, np.nan)
            np.fmax.at(outer, ab[s], r[s])
            prof[i] = np.nanmedian(outer) if np.isfinite(outer).sum() >= 4 else np.nan
        ok = np.isfinite(prof)
        if ok.sum() < 3:
            raise ValueError("too little evidence for a revolution profile")
        prof = np.interp(mid, mid[ok], prof[ok])
        body = r <= 1.1 * prof[idx]
        centre = _kasa(pts[body][:, [0, 2]])
    prof = gaussian_filter1d(median_filter(prof, 5, mode="nearest"), 3.0, mode="nearest")  # clean, not wavy
    return centre, mid, prof


def revolve(r: np.ndarray, y: np.ndarray, centre: np.ndarray, segments: int) -> trimesh.Trimesh:
    """Revolve the open profile polyline (r[i], y[i]) about the vertical axis at `centre`. The first
    and last profile points must lie on the axis (r = 0): they become poles, so the result is closed."""
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [np.c_[centre[0] + ri * np.cos(t), np.full(segments, yi), centre[1] + ri * np.sin(t)]
             for ri, yi in zip(r[1:-1], y[1:-1])]
    n = len(rings)
    V = np.vstack([*rings, [[centre[0], y[0], centre[1]], [centre[0], y[-1], centre[1]]]])
    p0, p1 = len(V) - 2, len(V) - 1
    j = np.arange(segments)
    k = (j + 1) % segments
    F = [np.c_[np.full(segments, p0), k, j]]
    for i in range(n - 1):
        a, b = i * segments + j, i * segments + k
        F += [np.c_[a, b, b + segments], np.c_[a, b + segments, a + segments]]
    F.append(np.c_[np.full(segments, p1), (n - 1) * segments + j, (n - 1) * segments + k])
    mesh = trimesh.Trimesh(V, np.vstack(F), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def lathe(evidence: trimesh.Trimesh, hollow: bool = False, segments: int = 64,
          wall: float = 0.07, floor: float = 0.1) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Surface of revolution fitted to the evidence. hollow=True: open top with a wall of
    `wall` x max radius and an inner floor at `floor` x height (a visual hull cannot see the
    cavity, so its depth is a category prior, not a measurement)."""
    centre, h, prof = fit_profile(_sample(evidence))
    lo, hi = h[0] - (h[1] - h[0]) / 2, h[-1] + (h[1] - h[0]) / 2
    r = np.r_[0.0, prof[0], prof, prof[-1]]
    y = np.r_[lo, lo, h, hi]
    if hollow:
        w = wall * prof.max()
        keep = h >= lo + floor * (hi - lo)
        ri, yi = np.maximum(prof[keep] - w, 0.2 * prof[keep])[::-1], h[keep][::-1]
        r = np.r_[r, ri[0], ri, ri[-1], 0.0]
        y = np.r_[y, hi, yi, yi[-1], yi[-1]]
    else:
        r, y = np.r_[r, 0.0], np.r_[y, hi]
    mesh = revolve(r, y, centre, segments)
    return mesh, {"part": "lathe", "hollow": hollow, "segments": segments, "rings": RINGS,
                  "axis_xz": centre.round(5).tolist(), "radius_max": round(float(prof.max()), 5),
                  "height": round(float(hi - lo), 5), "_centre": centre, "_h": h, "_prof": prof}


def rounded_box(evidence: trimesh.Trimesh, bevel: float = 0.08) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Box oriented by the x/z principal axes of the evidence (Y stays up), edges rounded with radius
    bevel x smallest extent: the convex hull of eight corner spheres."""
    pts = _sample(evidence)
    xz = pts[:, [0, 2]] - pts[:, [0, 2]].mean(0)
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    R = np.eye(3)
    R[np.ix_([0, 2], [0, 2])] = vt
    local = pts @ R.T
    lo, hi = np.percentile(local, 1, 0), np.percentile(local, 99, 0)
    rad = bevel * float((hi - lo).min())
    sphere = trimesh.creation.icosphere(2, radius=max(rad, 1e-6)).vertices
    corners = np.array(np.meshgrid(*zip(lo + rad, hi - rad), indexing="ij")).reshape(3, -1).T
    mesh = trimesh.convex.convex_hull((corners[:, None] + sphere[None]).reshape(-1, 3))
    mesh.vertices = np.asarray(mesh.vertices) @ R
    return mesh, {"part": "rounded_box", "bevel_radius": round(rad, 5), "extent": (hi - lo).round(5).tolist()}


def tube(curve: np.ndarray, radius: float, sides: int = 16) -> trimesh.Trimesh:
    """Closed circular tube swept along a polyline (parallel-transport frames, capped ends)."""
    T = np.gradient(curve, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for i in range(len(curve)):
        N = N - T[i] * (N @ T[i])
        N /= np.linalg.norm(N)
        B = np.cross(T[i], N)
        rings.append(curve[i] + radius * (np.cos(t)[:, None] * N + np.sin(t)[:, None] * B))
    V = np.vstack([*rings, curve[0], curve[-1]])
    p0, p1, n = len(V) - 2, len(V) - 1, len(curve)
    j = np.arange(sides)
    k = (j + 1) % sides
    F = [np.c_[np.full(sides, p0), k, j]]
    for i in range(n - 1):
        a, b = i * sides + j, i * sides + k
        F += [np.c_[a, b, b + sides], np.c_[a, b + sides, a + sides]]
    F.append(np.c_[np.full(sides, p1), (n - 1) * sides + j, (n - 1) * sides + k])
    mesh = trimesh.Trimesh(V, np.vstack(F), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def handle(evidence: trimesh.Trimesh, fit: dict[str, Any], min_frac: float = 0.01
           ) -> tuple[trimesh.Trimesh | None, dict[str, Any]]:
    """Evidence outside the lathe body (residual) -> planar curve ordered by angle around its
    attachment midpoint -> swept tube. None if the residual is too small to be a part."""
    pts = _sample(evidence)
    c, h, prof = fit["_centre"], fit["_h"], fit["_prof"]
    d = pts[:, [0, 2]] - c
    r = np.linalg.norm(d, axis=1)
    rb = np.interp(pts[:, 1], h, prof)
    res = pts[r > 1.12 * rb + 0.02 * prof.max()]
    rep: dict[str, Any] = {"part": "handle", "residual_frac": round(len(res) / len(pts), 4)}
    if len(res) < min_frac * len(pts):
        return None, {**rep, "skipped": "residual too small"}
    # Plane: the axis and the mean direction of the residual (a handle lies in a plane through the axis).
    dirs = res[:, [0, 2]] - c
    u = np.median(dirs / np.linalg.norm(dirs, axis=1, keepdims=True), 0)
    u /= np.linalg.norm(u)
    s = dirs @ u  # radial coordinate in the handle plane
    ymid = float(np.median(res[:, 1]))
    pivot_s = float(np.interp(ymid, h, prof))
    ang = np.arctan2(res[:, 1] - ymid, s - pivot_s)
    bins = np.linspace(ang.min(), ang.max(), 17)
    idx = np.clip(np.searchsorted(bins, ang) - 1, 0, 15)
    pts2 = [(np.median(s[idx == i]), np.median(res[idx == i, 1])) for i in range(16) if np.count_nonzero(idx == i) >= 5]
    if len(pts2) < 4:
        return None, {**rep, "skipped": "residual is not a curve"}
    p = np.array(pts2)
    w = dirs @ np.array([-u[1], u[0]])  # out-of-plane coordinate: its spread is the tube diameter
    radius = float(np.clip((np.percentile(w, 95) - np.percentile(w, 5)) / 2, 0.04 * prof.max(), 0.15 * prof.max()))
    # Push both ends into the body so the parts overlap (no gap at the attachment).
    ends = [np.array([np.interp(p[e, 1], h, prof) - radius, p[e, 1]]) for e in (0, -1)]
    p = np.vstack([ends[0], p, ends[1]])
    p2 = np.vstack([np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(p)), p[:, k]) for k in (0, 1)]).T
    p2 = np.c_[gaussian_filter1d(p2[:, 0], 1.5, mode="nearest"), gaussian_filter1d(p2[:, 1], 1.5, mode="nearest")]
    curve = np.c_[c[0] + p2[:, 0] * u[0], p2[:, 1], c[1] + p2[:, 0] * u[1]]
    return tube(curve, radius), {**rep, "tube_radius": round(radius, 5), "curve_points": len(curve)}


def voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, res: int = 64) -> float:
    """Occupancy IoU of two closed meshes on a shared grid (quick fit check, not the mask IoU)."""
    lo = np.minimum(a.bounds[0], b.bounds[0])
    hi = np.maximum(a.bounds[1], b.bounds[1])
    g = np.stack(np.meshgrid(*[np.linspace(lo[k], hi[k], res) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
    occ = []
    for m in (a, b):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
        occ.append(scene.compute_occupancy(o3d.core.Tensor(g.astype(np.float32))).numpy() > 0)
    return float((occ[0] & occ[1]).sum() / max((occ[0] | occ[1]).sum(), 1))


def build(job_dir: Path, kind: str, hollow: bool = False, segments: int = 64, bevel: float = 0.08,
          with_handle: bool = False) -> dict[str, Any]:
    """Fit `kind` ('lathe' | 'rounded_box') to mesh/high.ply and write it as mesh/low.ply."""
    evidence = trimesh.load(job_dir / "mesh" / "high.ply", force="mesh", process=False)
    parts, reports = [], []
    if kind == "lathe":
        body, fit = lathe(evidence, hollow, segments)
        parts.append(body)
        reports.append({k: v for k, v in fit.items() if not k.startswith("_")})
        if with_handle:
            hnd, hrep = handle(evidence, fit)
            reports.append(hrep)
            if hnd is not None:
                parts.append(hnd)
                hrep["watertight"] = bool(hnd.is_watertight)
    elif kind == "rounded_box":
        body, rep = rounded_box(evidence, bevel)
        parts.append(body)
        reports.append(rep)
    else:
        raise ValueError(f"unknown primitive {kind!r}")
    reports[0]["watertight"] = bool(parts[0].is_watertight)
    mesh = trimesh.util.concatenate(parts)
    mesh.export(job_dir / "mesh" / "low.ply")
    return {"kind": kind, "parts": reports, "triangles": int(len(mesh.faces)),
            "watertight_all": all(p.is_watertight for p in parts), "hull_voxel_iou": round(voxel_iou(mesh, evidence), 4)}
