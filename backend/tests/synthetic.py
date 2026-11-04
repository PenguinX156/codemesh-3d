"""Synthetic multi-view scene for pipeline tests (fuse, mesh, texture, evaluate).

A textured box + sphere union standing on the world z=0 plane (world up = +Z, so stages that
estimate "up" from cameras have real work to do), seen by a low and an elevated camera ring.
Everything is ray cast with Open3D's CPU RaycastingScene, so no GPU or renderer is needed.

`make_scene(job_dir)` writes the stage inputs exactly as the real pipeline does:
  frames/frames.json + frames/<NNNN>.png (full-res RGB), masks/<NNNN>.png (255 = object),
  geometry/cameras.json (full-res cameras), geometry/views.npz (engine-res outputs with
  mild depth noise, boundary bleed, confidence and pixel_to_original).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from codemesh.pipeline.cameras import Camera, save_cameras

FULL_SIZE = (960, 720)  # (width, height) of the "original" frames
ENGINE_SIZE = (518, 388)  # (width, height) of the engine outputs
BACKGROUND_DEPTH = 3.5  # engines predict depth everywhere; the background is a far wall


@dataclass
class Scene:
    """Ground truth returned by `make_scene` (world frame)."""

    mesh: o3d.geometry.TriangleMesh  # union of the primitives (interiors overlap; fine for distances)
    visible_points: np.ndarray  # (M, 3) surface points seen by at least one full-res camera
    diagonal: float  # bbox diagonal of the object
    up: np.ndarray  # world up vector (+Z)
    cameras: list[Camera]


def object_mesh() -> o3d.geometry.TriangleMesh:
    """Box 0.6 x 0.4 x 0.4 on z=0 plus a sphere (r 0.22) sitting on top, off-center."""
    box = o3d.geometry.TriangleMesh.create_box(0.6, 0.4, 0.4).translate((-0.3, -0.2, 0.0))
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.22, resolution=60).translate((0.1, 0.0, 0.5))
    return box + sphere


def texture(points: np.ndarray) -> np.ndarray:
    """Procedural 3D texture (checker + smooth color bands), uint8 RGB per point."""
    cells = np.floor(points / 0.04).astype(np.int64).sum(axis=1) % 2
    base = 0.5 + 0.25 * np.sin(points @ np.array([[9.0, 4.0, 2.0], [-3.0, 8.0, 5.0], [6.0, -5.0, 7.0]]).T)
    return np.clip(255 * (base * (0.6 + 0.4 * cells[:, None])), 0, 255).astype(np.uint8)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 world -> camera extrinsic, OpenCV axes (x right, y down, z forward)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    E = np.eye(4)
    E[:3, :3] = np.stack([x, y, z])
    E[:3, 3] = -E[:3, :3] @ eye
    return E


def ring_cameras(n_ring: int, n_top: int, radius: float = 1.6) -> list[np.ndarray]:
    """Extrinsics on a low ring (10 deg elevation) and an elevated ring (40 deg)."""
    target, up = np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 1.0])
    out = []
    for n, elev, phase in ((n_ring, 10.0, 0.0), (n_top, 40.0, 0.5)):
        for i in range(n):
            a, e = 2 * np.pi * (i + phase) / n, np.radians(elev)
            eye = target + radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
            out.append(look_at(eye, target, up))
    return out


def cast(scene: o3d.t.geometry.RaycastingScene, K: np.ndarray, E: np.ndarray, size: tuple[int, int]) -> dict:
    """Ray cast one pinhole view (pixel centers at integer coords). Returns z-depth (inf = miss),
    world hit points and unit normals, each (H, W, ...)."""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dirs_cam = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T  # z component = 1
    R, t = E[:3, :3], E[:3, 3]
    dirs = dirs_cam @ R  # camera -> world (R^T d)
    origin = np.broadcast_to(-R.T @ t, dirs.shape)
    rays = np.concatenate([origin, dirs], -1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    depth = ans["t_hit"].numpy()  # equals z-depth because dirs_cam.z == 1
    hits = origin + dirs * np.where(np.isfinite(depth), depth, 0)[..., None]
    return {"depth": depth, "points": hits, "normals": ans["primitive_normals"].numpy(), "dirs": dirs}


def make_scene(job_dir: Path, n_ring: int = 16, n_top: int = 8, seed: int = 0,
               full_size: tuple[int, int] = FULL_SIZE, engine_size: tuple[int, int] = ENGINE_SIZE,
               depth_noise: float = 0.003) -> Scene:
    """Render the scene into `job_dir` (see module doc). depth_noise = relative depth std."""
    rng = np.random.default_rng(seed)
    mesh = object_mesh()
    mesh.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for d in ("frames", "masks", "geometry"):
        (job_dir / d).mkdir(parents=True, exist_ok=True)

    (W, H), (w, h) = full_size, engine_size
    K_full = np.array([[0.95 * W, 0, (W - 1) / 2], [0, 0.95 * W, (H - 1) / 2], [0, 0, 1]])
    # Engine pixel (u, v) -> original pixel, pixel-center aligned plain resize.
    sx, sy = W / w, H / h
    p2o = np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1]])
    K_eng = np.linalg.inv(p2o) @ K_full

    cams, frames, visible = [], [], []
    depths, confs, points = [], [], []
    for i, E in enumerate(ring_cameras(n_ring, n_top)):
        name = f"{i:04d}.png"
        full = cast(scene, K_full, E, full_size)
        hit = np.isfinite(full["depth"])
        rgb = rng.integers(90, 140, (H, W, 3), dtype=np.uint8)  # noisy gray background
        rgb[hit] = texture(full["points"][hit])
        cv2.imwrite(str(job_dir / "frames" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(job_dir / "masks" / name), (hit * 255).astype(np.uint8))
        visible.append(full["points"][hit][::7])
        frames.append({"index": i, "file": name, "width": W, "height": H})
        cams.append(Camera(i, f"frames/{name}", W, H, K_full.tolist(), E.tolist()))

        eng = cast(scene, K_eng, E, engine_size)
        hit = np.isfinite(eng["depth"])
        depth = np.where(hit, eng["depth"], BACKGROUND_DEPTH).astype(np.float32)
        depth *= 1 + depth_noise * rng.standard_normal(depth.shape).astype(np.float32)
        depth = cv2.blur(depth, (3, 3))  # boundary bleed ("flying pixels") like real networks
        # Confidence: lower at grazing angles, plus a few low-confidence gross outliers.
        cos = np.abs((eng["normals"] * eng["dirs"]).sum(-1)) / np.linalg.norm(eng["dirs"], axis=-1)
        conf = np.where(hit, 1 + 4 * cos, 1.0).astype(np.float32)
        bad = rng.random(depth.shape) < 0.02
        depth[bad] *= 1 + 0.2 * rng.standard_normal(bad.sum()).astype(np.float32)
        conf[bad] = 0.5
        depths.append(depth)
        confs.append(conf)
        dirs_cam = np.stack([*np.meshgrid(np.arange(w), np.arange(h)), np.ones((h, w))], -1) @ np.linalg.inv(K_eng).T
        cam_pts = dirs_cam * depth[..., None]
        points.append(((cam_pts - E[:3, 3]) @ E[:3, :3]).astype(np.float32))

    n = len(cams)
    (job_dir / "frames" / "frames.json").write_text(json.dumps({"frames": frames}, indent=1))
    save_cameras(job_dir / "geometry" / "cameras.json", cams, metric_scale=True)
    np.savez(
        job_dir / "geometry" / "views.npz",
        depth=np.stack(depths), depth_conf=np.stack(confs),
        points=np.stack(points), points_conf=np.stack(confs),
        pixel_to_original=np.repeat(p2o[None], n, 0), intrinsics=np.repeat(K_eng[None], n, 0),
        extrinsics=np.stack([np.asarray(c.extrinsic) for c in cams]), frame_indices=np.arange(n),
    )
    bb = mesh.get_axis_aligned_bounding_box()
    return Scene(mesh, np.concatenate(visible), float(np.linalg.norm(bb.get_extent())),
                 np.array([0.0, 0.0, 1.0]), cams)


def mesh_distance(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Unsigned distance from each point to the mesh surface."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
