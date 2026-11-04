"""Evaluator accuracy, frame conventions and bounded CPU runtime.

Run the optional 40-view / 32768-triangle benchmark with:
    uv run --project backend python backend/tests/test_evaluate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import look_at  # noqa: E402

from codemesh.pipeline import evaluate  # noqa: E402
from codemesh.pipeline.cameras import Camera, save_cameras  # noqa: E402

TRANSFORM = np.array([[1.7, 0, 0, .3], [0, 0, 1.7, -.2], [0, -1.7, 0, .4], [0, 0, 0, 1]])


def _job(job: Path, n_views: int = 8, subdivisions: int = 2, size: tuple[int, int] = (192, 144)) -> dict:
    for folder in ("geometry", "mesh", "texture", "frames", "masks", "fuse"):
        (job / folder).mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    world_vertices = np.asarray(mesh.vertices)
    # Split corners so seam UVs cannot interpolate across the sphere's longitude wrap.
    vertices = world_vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    uv = np.column_stack([np.arctan2(vertices[:, 1], vertices[:, 0]) / (2 * np.pi) + .5,
                          np.arcsin(vertices[:, 2]) / np.pi + .5]).reshape(-1, 3, 2)
    seam = np.ptp(uv[:, :, 0], axis=1) > .5
    uv[seam, :, 0] = np.where(uv[seam, :, 0] < .5, uv[seam, :, 0] + 1, uv[seam, :, 0])
    # Keep all test UVs inside the image, including the longitude seam.
    uv = uv.reshape(-1, 2) * np.array([.7, .9]) + np.array([.02, .05])
    y, x = np.indices((128, 160))
    checker = ((x // 16 + y // 16) % 2)[..., None]
    albedo = np.where(checker, np.array([35, 200, 240]), np.array([220, 65, 40])).astype(np.uint8)
    asset_vertices = vertices @ TRANSFORM[:3, :3].T + TRANSFORM[:3, 3]
    asset_mesh = trimesh.Trimesh(asset_vertices, faces, process=False)
    asset_mesh.export(job / "mesh" / "low.ply")
    (job / "mesh" / "transform.json").write_text(json.dumps({"matrix": TRANSFORM.tolist()}))
    np.savez(job / "texture" / "uv_mesh.npz", vertices=asset_vertices.astype(np.float32), faces=faces, uv=uv)
    cv2.imwrite(str(job / "texture" / "albedo.png"), albedo)
    cloud, _ = trimesh.sample.sample_surface(mesh, 20000, seed=7)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    o3d.io.write_point_cloud(str(job / "fuse" / "points.ply"), pcd)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    w, h = size
    K = np.array([[.95 * w, 0, (w - 1) / 2], [0, .95 * w, (h - 1) / 2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    directions = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    cams = []
    for i in range(n_views):
        angle = i * 2 * np.pi / n_views
        eye = np.array([3.7 * np.cos(angle), 3.7 * np.sin(angle), .7])
        E = look_at(eye, np.zeros(3), np.array([0., 0., 1.]))
        dirs = directions @ E[:3, :3]
        rays = np.concatenate([np.broadcast_to(eye, dirs.shape), dirs], -1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        hit = np.isfinite(result["t_hit"].numpy())
        ids = result["primitive_ids"].numpy()[hit]
        b = result["primitive_uvs"].numpy()[hit]
        weights = np.column_stack([1 - b.sum(1), b])
        coords = np.einsum("ni,nij->nj", weights, uv[faces[ids]])
        # Independent reference lookup (no evaluator rendering helpers).
        mx = np.zeros((h, w), np.float32)
        my = np.zeros((h, w), np.float32)
        mx[hit] = coords[:, 0] * albedo.shape[1] - .5
        my[hit] = (1 - coords[:, 1]) * albedo.shape[0] - .5
        photo = cv2.remap(albedo, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        photo[~hit] = 100
        cv2.imwrite(str(job / "frames" / f"{i:04d}.png"), photo)
        cv2.imwrite(str(job / "masks" / f"{i:04d}.png"), hit.astype(np.uint8) * 255)
        cams.append(Camera(i, f"frames/{i:04d}.png", w, h, K.tolist(), E.tolist()))
    save_cameras(job / "geometry" / "cameras.json", cams, metric_scale=True)
    return {"mesh": asset_mesh, "vertices": asset_vertices, "faces": faces, "uv": uv, "albedo": albedo}


def test_same_mesh_and_scale_regression(tmp_path: Path) -> None:
    _job(tmp_path)
    progress = []
    result = evaluate.run(tmp_path, {"holdout_indices": [1, 5]}, lambda f, msg: progress.append(f))
    assert result["views_evaluated"] == 8
    assert result["iou_mean"] > .98
    assert result["iou_holdout_mean"] > .98
    assert result["boundary_f_mean"] > .99
    assert result["psnr_mean"] > 48
    assert result["ssim_mean"] > .995
    assert result["chamfer_pct_mean"] < 1
    assert progress[0] == 0 and progress[-1] == 1
    metrics = json.loads((tmp_path / "evaluate" / "metrics.json").read_text())
    assert metrics["holdout"]["views_evaluated"] == 2
    assert metrics["holdout"]["psnr_mean"] > 48
    assert metrics["geometry"]["chamfer_pct_p90"] < 1.5
    images = list((tmp_path / "evaluate").glob("compare_*.png"))
    assert len(images) == 6
    assert all(cv2.imread(str(path)).shape[1] <= 1600 for path in images)

    mesh = trimesh.load(tmp_path / "mesh" / "low.ply", process=False)
    mesh.vertices = (mesh.vertices - TRANSFORM[:3, 3]) * 1.1 + TRANSFORM[:3, 3]
    mesh.export(tmp_path / "mesh" / "low.ply")
    worse = evaluate.run(tmp_path, {})
    assert worse["iou_mean"] < result["iou_mean"] - .1
    assert worse["chamfer_pct_mean"] > result["chamfer_pct_mean"]
    assert worse["iou_holdout_mean"] is None


def test_glb_fallback_and_resizing(tmp_path: Path) -> None:
    data = _job(tmp_path, n_views=2, size=(384, 288))
    visual = trimesh.visual.texture.TextureVisuals(uv=data["uv"], image=Image.fromarray(data["albedo"][..., ::-1]))
    mesh = trimesh.Trimesh(data["vertices"], data["faces"], visual=visual, process=False)
    (tmp_path / "export").mkdir()
    # Also exercise a GLB scene node transform independently of mesh/transform.json.
    local = mesh.copy()
    local.apply_translation([-.4, .2, -.1])
    node_transform = np.eye(4)
    node_transform[:3, 3] = [.4, -.2, .1]
    scene = trimesh.Scene()
    scene.add_geometry(local, transform=node_transform)
    scene.export(tmp_path / "export" / "model.glb")
    (tmp_path / "texture" / "uv_mesh.npz").unlink()
    (tmp_path / "texture" / "albedo.png").unlink()
    result = evaluate.run(tmp_path, {"eval_long_side": 192})
    assert result["iou_mean"] > .98
    assert result["psnr_mean"] > 25
    metrics = json.loads((tmp_path / "evaluate" / "metrics.json").read_text())
    assert metrics["views"][0]["width"] == 192
    assert metrics["views"][0]["height"] == 144


def test_empty_intersection_and_masked_ssim() -> None:
    mask = np.zeros((32, 32), bool)
    a = np.zeros((32, 32, 3), np.uint8)
    b = np.full_like(a, 255)
    assert evaluate._appearance(a, b, mask) == (None, None)
    assert evaluate._silhouette(mask, mask) == (1, 1)
    mask[10:22, 10:22] = True
    b[mask] = 0
    psnr, ssim = evaluate._appearance(a, b, mask)
    assert psnr == pytest.approx(100)
    assert ssim == pytest.approx(1)
    b[mask] = 255
    psnr, ssim = evaluate._appearance(a, b, mask)
    assert psnr == pytest.approx(0)
    assert ssim < .001
    assert evaluate._silhouette(mask, np.zeros_like(mask)) == (0, 0)


def test_missing_geometry_and_evidence(tmp_path: Path) -> None:
    _job(tmp_path, n_views=2)
    (tmp_path / "fuse" / "points.ply").unlink()
    (tmp_path / "masks" / "0001.png").unlink()
    result = evaluate.run(tmp_path, {})
    assert result["views_evaluated"] == 1
    assert result["chamfer_pct_mean"] is None
    metrics = json.loads((tmp_path / "evaluate" / "metrics.json").read_text())
    assert metrics["skipped"][0]["index"] == 1
    (tmp_path / "masks" / "0000.png").unlink()
    with pytest.raises(ValueError, match="No cameras"):
        evaluate.run(tmp_path, {})


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="codemesh-evaluate-") as folder:
        job = Path(folder)
        _job(job, n_views=40, subdivisions=5, size=(1024, 768))
        started = time.perf_counter()
        result = evaluate.run(job, {"holdout_indices": list(range(0, 40, 5))})
        print(json.dumps({"benchmark_wall_seconds": time.perf_counter() - started,
                          "triangles": 20480, **result}, indent=2))
