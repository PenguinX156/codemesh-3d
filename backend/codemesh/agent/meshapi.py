"""Procedural modeling API that Muse-written `build(api, evidence)` code programs against.

Asset frame: +Y up, metres when the job is metric. Every primitive returns one closed (watertight)
trimesh body; `combine` concatenates bodies without boolean ops. UVs are left to the texture stage,
which unwraps the final mesh itself. `Evidence` exposes read-only measurements of the visual hull.
shortcut: no bevels (the arg is accepted and ignored), upgrade if Muse output looks too boxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh


def _ok(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m.merge_vertices()
    m.fix_normals()
    return m


def box(w: float, h: float, d: float, bevel: float = 0.0) -> trimesh.Trimesh:
    """Axis-aligned box centred at the origin (w along X, h along Y, d along Z)."""
    return trimesh.creation.box(extents=(float(w), float(h), float(d)))


def _y_up(m: trimesh.Trimesh) -> trimesh.Trimesh:
    # trimesh builds round primitives along Z; the asset frame is +Y up.
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return m


def cylinder(r: float, h: float, segments: int = 48) -> trimesh.Trimesh:
    """Cylinder along Y, base at y=0."""
    return translate(_y_up(trimesh.creation.cylinder(radius=float(r), height=float(h), sections=int(segments))),
                     0, h / 2, 0)


def capsule(r: float, h: float) -> trimesh.Trimesh:
    """Capsule along Y centred at the origin; h is the straight section length."""
    return _y_up(trimesh.creation.capsule(height=float(h), radius=float(r)))


def torus(major: float, minor: float, segments: int = 48) -> trimesh.Trimesh:
    """Torus in the XZ plane (hole along Y) centred at the origin."""
    return _y_up(trimesh.creation.torus(major_radius=float(major), minor_radius=float(minor),
                                        major_sections=int(segments), minor_sections=max(8, int(segments) // 3)))


def revolve(profile, segments: int = 64, wall: float | None = None, floor: float | None = None) -> trimesh.Trimesh:
    """Lathe a profile [(r, y), ...] (bottom to top, r >= 0) around +Y.

    Solid: the profile is closed through the axis. With `wall`, the body is a vessel: the outer
    profile, a rim, then an inner surface offset inward by `wall` down to `floor` above the base.
    """
    p = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        raise ValueError("revolve needs at least 2 profile points")
    p[:, 0] = np.maximum(p[:, 0], 0.0)
    if wall:
        y0 = p[0, 1] + float(floor if floor is not None else wall)
        inner = [(max(r - wall, 1e-4), y) for r, y in p[::-1] if y >= y0]
        ring = [(0.0, p[0, 1]), *map(tuple, p), *inner, (0.0, y0)]
    else:
        ring = [(0.0, p[0, 1]), *map(tuple, p), (0.0, p[-1, 1])]
    ring = np.asarray([pt for i, pt in enumerate(ring) if i == 0 or not np.allclose(pt, ring[i - 1])])
    return _lathe(ring, int(segments))


def _lathe(ring: np.ndarray, n: int) -> trimesh.Trimesh:
    """Revolve a polyline whose first and last points lie on the axis; poles become single vertices."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts, faces = [np.array([0.0, ring[0, 1], 0.0])], []
    mids = ring[1:-1]
    for r, y in mids:
        verts.extend(np.c_[r * np.cos(a), np.full(n, y), r * np.sin(a)])
    top = len(verts)
    verts.append(np.array([0.0, ring[-1, 1], 0.0]))
    ring_idx = lambda k: 1 + k * n + np.arange(n)
    nxt = lambda idx: np.roll(idx, -1)
    first, last = ring_idx(0), ring_idx(len(mids) - 1)
    faces += [[0, j, i] for i, j in zip(first, nxt(first))]
    for k in range(len(mids) - 1):
        i0, i1 = ring_idx(k), ring_idx(k + 1)
        faces += [[i, j, nj] for i, j, nj in zip(i0, nxt(i0), nxt(i1))]
        faces += [[i, nj, ni] for i, nj, ni in zip(i0, nxt(i1), i1)]
    faces += [[top, i, j] for i, j in zip(last, nxt(last))]
    return _ok(trimesh.Trimesh(np.vstack(verts), np.asarray(faces), process=True))


def sweep(path, radius=0.005, closed: bool = False, segments: int = 16) -> trimesh.Trimesh:
    """Tube along a 3D polyline; radius is a float or (rx, ry) for an elliptical section."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("sweep needs at least 2 path points")
    rx, ry = (radius, radius) if np.isscalar(radius) else radius
    m = len(pts)
    tang = np.gradient(pts, axis=0) if not closed else np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 1.0, 0.0]) if abs(tang[0, 1]) < .9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(tang[0], ref)
    normal /= np.linalg.norm(normal)
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = []
    for i in range(m):  # parallel transport keeps the section from twisting
        normal = normal - tang[i] * (normal @ tang[i])
        normal /= np.linalg.norm(normal) + 1e-12
        binormal = np.cross(tang[i], normal)
        verts.extend(pts[i] + np.outer(rx * np.cos(a), normal) + np.outer(ry * np.sin(a), binormal))
    faces = []
    rings = m if closed else m - 1
    for i in range(rings):
        i0, i1 = i * segments + np.arange(segments), ((i + 1) % m) * segments + np.arange(segments)
        faces += [[p, q, nq] for p, q, nq in zip(i0, np.roll(i0, -1), np.roll(i1, -1))]
        faces += [[p, nq, q] for p, nq, q in zip(i0, np.roll(i1, -1), i1)]
    verts = np.asarray(verts)
    if not closed:
        start, end = len(verts), len(verts) + 1
        verts = np.vstack([verts, pts[0], pts[-1]])
        r0, r1 = np.arange(segments), (m - 1) * segments + np.arange(segments)
        faces += [[start, j, i] for i, j in zip(r0, np.roll(r0, -1))]
        faces += [[end, i, j] for i, j in zip(r1, np.roll(r1, -1))]
    return _ok(trimesh.Trimesh(verts, np.asarray(faces), process=True))


def extrude(polygon2d, depth: float, bevel: float = 0.0) -> trimesh.Trimesh:
    """Extrude a star-shaped polygon [(x, z), ...] upward along +Y from y=0 by `depth`.

    shortcut: fan triangulation from the centroid, so only star-shaped outlines; add ear clipping if needed.
    """
    p = np.asarray(polygon2d, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        raise ValueError("extrude needs at least 3 points")
    n, c = len(p), p.mean(0)
    verts = np.vstack([np.c_[p[:, 0], np.zeros(n), p[:, 1]], np.c_[p[:, 0], np.full(n, depth), p[:, 1]],
                       [[c[0], 0, c[1]], [c[0], depth, c[1]]]])
    i, j = np.arange(n), np.roll(np.arange(n), -1)
    faces = np.vstack([np.c_[i, j, j + n], np.c_[i, j + n, i + n], np.c_[np.full(n, 2 * n), j, i],
                       np.c_[np.full(n, 2 * n + 1), i + n, j + n]])
    return _ok(trimesh.Trimesh(verts, faces, process=True))


def translate(m: trimesh.Trimesh, x: float, y: float, z: float) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_translation([float(x), float(y), float(z)])
    return m


def rotate(m: trimesh.Trimesh, degrees: float, axis=(0, 1, 0), about=(0, 0, 0)) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(float(degrees)), axis, about))
    return m


def scale(m: trimesh.Trimesh, sx: float, sy: float | None = None, sz: float | None = None) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_scale([sx, sx if sy is None else sy, sx if sz is None else sz])
    return m


def mirror(m: trimesh.Trimesh, axis: int = 0) -> trimesh.Trimesh:
    """Mirror across the plane through the origin normal to axis 0/1/2 (winding fixed)."""
    s = np.ones(3)
    s[int(axis)] = -1
    m = m.copy()
    m.apply_transform(np.diag([*s, 1.0]))  # trimesh re-winds faces for reflections
    return m


def array(m: trimesh.Trimesh, count: int, offset=(0, 0, 0), degrees: float = 0.0, axis=(0, 1, 0)) -> list:
    """`count` copies, each shifted by offset and rotated by `degrees` about axis more than the last."""
    return [translate(rotate(m, degrees * k, axis), *(np.asarray(offset) * k)) for k in range(int(count))]


def combine(parts) -> trimesh.Trimesh:
    """Concatenate closed bodies into one mesh (bodies may overlap; no booleans)."""
    flat = []
    for p in parts:
        flat.extend(p if isinstance(p, (list, tuple)) else [p])
    if not flat:
        raise ValueError("combine needs at least one part")
    return trimesh.util.concatenate(flat)


API = SimpleNamespace(box=box, cylinder=cylinder, capsule=capsule, torus=torus, revolve=revolve, sweep=sweep,
                      extrude=extrude, translate=translate, rotate=rotate, scale=scale, mirror=mirror,
                      array=array, combine=combine, pi=float(np.pi), sin=np.sin, cos=np.cos,
                      linspace=lambda a, b, n: [float(v) for v in np.linspace(a, b, int(n))])


class Evidence:
    """Read-only measurements of the visual hull (mesh/high.ply) in the asset frame."""

    def __init__(self, job_dir: Path):
        job_dir = Path(job_dir)
        hull = trimesh.load(job_dir / "mesh" / "high.ply", force="mesh", process=False)
        self.points = np.asarray(hull.sample(20000) if len(hull.faces) else hull.vertices)
        lo, hi = self.points.min(0), self.points.max(0)
        self.bbox = {"min": lo.round(4).tolist(), "max": hi.round(4).tolist(), "size": (hi - lo).round(4).tolist()}
        self.height = float(hi[1] - lo[1])
        self.base_y, self.top_y = float(lo[1]), float(hi[1])
        cams = job_dir / "geometry" / "cameras.json"
        self.metric = bool(json.loads(cams.read_text()).get("metric_scale")) if cams.exists() else False
        self.axis = [float(v) for v in np.median(self.points[:, [0, 2]], 0)]

    def radius_profile(self, n: int = 12) -> list:
        """[(y, r)] radial extent (90th percentile) about the vertical axis per height slice."""
        y = self.points[:, 1]
        r = np.linalg.norm(self.points[:, [0, 2]] - self.axis, axis=1)
        edges = np.linspace(y.min(), y.max(), int(n) + 1)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (y >= a) & (y <= b)
            if sel.any():
                out.append((round(float((a + b) / 2), 4), round(float(np.percentile(r[sel], 90)), 4)))
        return out

    def residual(self, body: trimesh.Trimesh, tol: float = 0.003) -> list:
        """Hull surface points farther than `tol` outside `body` (e.g. where a handle must go), <= 400."""
        import open3d as o3d  # trimesh proximity needs rtree, which is not installed

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(body.vertices.astype(np.float32)),
                            o3d.core.Tensor(body.faces.astype(np.uint32)))
        pts = self.points[::10]
        d = scene.compute_signed_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()  # negative inside
        pts = pts[d > tol]
        return pts[:: max(1, len(pts) // 400)].round(4).tolist()

    def opening(self) -> dict:
        """Rim height and an inner-depth estimate (hulls fill cavities, so depth is a guess: 90% of height)."""
        prof = self.radius_profile(12)
        return {"rim_y": self.top_y, "rim_radius": prof[-1][1] if prof else 0.0, "inner_depth_estimate": .9 * self.height}

    def summary(self) -> dict:
        return {"bbox": self.bbox, "height": self.height, "metric": self.metric, "axis_xz": self.axis,
                "radius_profile": self.radius_profile(12), "opening": self.opening()}


def evidence_namespace(ev: Evidence) -> SimpleNamespace:
    """What sandboxed code sees: plain values and bound methods only."""
    return SimpleNamespace(bbox=ev.bbox, height=ev.height, base_y=ev.base_y, top_y=ev.top_y, metric=ev.metric,
                           axis=ev.axis, radius_profile=ev.radius_profile, residual=ev.residual,
                           opening=ev.opening, silhouette_widths=getattr(ev, "silhouette_widths", {}))
