"""Post-mesh geometry operations the agent can ask for (applied to mesh/high.ply + mesh/low.ply).

Both meshes are already in the gravity-aligned asset frame (+Y up, bbox centred on x/z), so the
symmetry planes and the revolution axis are simple there. The loop keeps an operation only when
the re-run evaluate improves silhouette IoU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

SYMMETRIES = ("none", "mirror_x", "mirror_z", "rotational_y")
LATHE_RINGS = 48  # height bins of the revolution profile


def _io(job_dir: Path) -> tuple[Path, Path]:
    return job_dir / "mesh" / "high.ply", job_dir / "mesh" / "low.ply"


def mirror(v: np.ndarray, axis: int) -> np.ndarray:
    """Symmetrize vertices across the plane coord[axis] = 0: average each vertex with the mirror
    image of its nearest neighbour on the mirrored side."""
    m = v.copy()
    m[:, axis] *= -1
    _, nn = cKDTree(m).query(v)
    return (v + m[nn]) / 2


def fit_axis(xz: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle fit (Kasa) of x/z points -> (centre, radius); works on a partial arc."""
    A = np.c_[xz, np.ones(len(xz))]
    a, b, c = np.linalg.lstsq(A, -(xz ** 2).sum(1), rcond=None)[0]
    centre = np.array([-a / 2, -b / 2])
    return centre, float(np.sqrt(max(centre @ centre - c, 0.0)))


def lathe(v: np.ndarray, segments: int) -> o3d.geometry.TriangleMesh:
    """Surface of revolution about the fitted vertical axis: median radius per height ring, both
    ends capped. Turns a partial fragment of a round object into the whole object."""
    centre, _ = fit_axis(v[:, [0, 2]])
    y = v[:, 1]
    r = np.linalg.norm(v[:, [0, 2]] - centre, axis=1)
    edges = np.linspace(y.min(), y.max(), LATHE_RINGS + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.searchsorted(edges, y) - 1, 0, LATHE_RINGS - 1)
    prof = np.array([np.median(r[idx == i]) if np.count_nonzero(idx == i) >= 3 else np.nan
                     for i in range(LATHE_RINGS)])
    ok = ~np.isnan(prof)
    if ok.sum() < 2:
        raise ValueError("too few points for a revolution profile")
    prof = np.interp(mid, mid[ok], prof[ok])
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.c_[np.cos(t), np.sin(t)]
    verts = [np.c_[centre[0] + p * ring[:, 0], np.full(segments, h), centre[1] + p * ring[:, 1]]
             for h, p in zip(mid, prof)]
    verts.append([[centre[0], mid[0], centre[1]], [centre[0], mid[-1], centre[1]]])
    V = np.vstack(verts)
    bottom, top = len(V) - 2, len(V) - 1
    F = []
    for i in range(LATHE_RINGS - 1):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            F += [[a, b + segments, b], [a, a + segments, b + segments]]
    last = (LATHE_RINGS - 1) * segments
    F += [[bottom, j, (j + 1) % segments] for j in range(segments)]
    F += [[top, last + (j + 1) % segments, last + j] for j in range(segments)]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.array(F)))
    mesh.compute_vertex_normals()
    return mesh


def apply_symmetry(job_dir: Path, kind: str, triangle_budget: int | None = None) -> str:
    """Apply `kind` to mesh/high.ply and mesh/low.ply in place; returns a one-line note."""
    if kind not in SYMMETRIES:
        raise ValueError(f"unknown symmetry {kind!r}")
    if kind == "none":
        return "no symmetry"
    hp, lp = _io(job_dir)
    high = o3d.io.read_triangle_mesh(str(hp))
    v = np.asarray(high.vertices)
    if kind == "rotational_y":
        low_segments = int(np.clip((triangle_budget or 4000) / (2 * LATHE_RINGS), 16, 256))
        o3d.io.write_triangle_mesh(str(hp), lathe(v, 128))
        o3d.io.write_triangle_mesh(str(lp), lathe(v, low_segments))
        return f"rebuilt as a surface of revolution about +Y ({LATHE_RINGS} rings, low {low_segments} segments)"
    axis = 0 if kind == "mirror_x" else 2
    for p in (hp, lp):
        mesh = o3d.io.read_triangle_mesh(str(p))
        mesh.vertices = o3d.utility.Vector3dVector(mirror(np.asarray(mesh.vertices), axis))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(p), mesh)
    return f"vertices symmetrized across the {'yz' if axis == 0 else 'xy'} plane"
