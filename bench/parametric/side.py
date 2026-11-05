"""Shaded front + side orthographic render of mesh/low.ply (shows the primitive parts without texture).

uv run --project backend python bench/parametric/side.py <job_dir> <out.png>
"""
import sys

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

m = trimesh.load(f"{sys.argv[1]}/mesh/low.ply", force="mesh", process=False)
s = o3d.t.geometry.RaycastingScene()
s.add_triangles(o3d.core.Tensor(m.vertices.astype(np.float32)), o3d.core.Tensor(m.faces.astype(np.uint32)))
lo, hi = m.bounds
pad = 0.08 * float(np.ptp(m.bounds, 0).max())
imgs = []
for ax, side in ((2, 0), (0, 2)):
    U, V = np.meshgrid(np.linspace(lo[side] - pad, hi[side] + pad, 400), np.linspace(hi[1] + pad, lo[1] - pad, 320))
    o = np.zeros((*U.shape, 3))
    o[..., side], o[..., 1], o[..., ax] = U, V, lo[ax] - 1
    d = np.zeros_like(o)
    d[..., ax] = 1
    r = s.cast_rays(o3d.core.Tensor(np.concatenate([o, d], -1).astype(np.float32)))
    n, hit = r["primitive_normals"].numpy(), np.isfinite(r["t_hit"].numpy())
    imgs.append(np.where(hit, (np.clip(np.abs(n @ np.array([.5, .7, .5])), 0, 1) * 230).astype(np.uint8), 40))
Image.fromarray(np.concatenate(imgs, 1)).save(sys.argv[2])
