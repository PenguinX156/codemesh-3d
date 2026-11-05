"""Fuse + mesh stages on the synthetic box+sphere scene (tests/synthetic.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

from codemesh.pipeline import fuse, hull, mesh
from synthetic import Scene, make_scene, mesh_distance


@pytest.fixture(scope="module")
def job(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Scene]:
    job_dir = tmp_path_factory.mktemp("job")
    scene = make_scene(job_dir)
    fuse.run(job_dir, {})
    return job_dir, scene


def copy_job(src: Path, dst: Path) -> Path:
    for d in ("frames", "masks", "geometry", "fuse"):
        shutil.copytree(src / d, dst / d)
    return dst


def load_world_mesh(job_dir: Path, name: str) -> o3d.geometry.TriangleMesh:
    """mesh/<name>.ply mapped back to the world frame with mesh/transform.json."""
    M = np.array(json.loads((job_dir / "mesh" / "transform.json").read_text())["matrix"])
    m = o3d.io.read_triangle_mesh(str(job_dir / "mesh" / f"{name}.ply"))
    return m.transform(np.linalg.inv(M))


def chamfer_pct(m: o3d.geometry.TriangleMesh, scene: Scene) -> float:
    """Mean of accuracy (mesh -> GT) and completeness (visible GT -> mesh), % of the diagonal."""
    acc = mesh_distance(np.asarray(m.sample_points_uniformly(60_000).points), scene.mesh).mean()
    comp = mesh_distance(scene.visible_points[::4], m).mean()
    return 100 * (acc + comp) / 2 / scene.diagonal


def test_fuse_points_accurate_colored_oriented(job: tuple[Path, Scene]) -> None:
    job_dir, scene = job
    pcd = o3d.io.read_point_cloud(str(job_dir / "fuse" / "points.ply"))
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    assert pcd.has_colors() and len(pts) > 50_000
    d = mesh_distance(pts, scene.mesh)
    assert d.mean() < 0.003 * scene.diagonal  # no flying pixels / background points survive
    assert np.quantile(d, 0.999) < 0.01 * scene.diagonal
    outward = ((pts - np.array([0.0, 0.0, 0.3])) * nrm).sum(1) > 0  # object is star-shaped about this point
    assert outward.mean() > 0.95


def test_mesh_fidelity_budget_and_frame(job: tuple[Path, Scene], tmp_path: Path) -> None:
    job_dir = copy_job(job[0], tmp_path)
    scene = job[1]
    m = mesh.run(job_dir, {"preset": "standard"})
    high, low = load_world_mesh(job_dir, "high"), load_world_mesh(job_dir, "low")
    assert chamfer_pct(high, scene) < 1.0
    assert chamfer_pct(low, scene) < 1.0
    assert m["method"] == "hull_depth" and m["hull"]["iou_mean"] > 0.85
    assert 7_500 <= m["tris_low"] <= 150_000 and m["tris_low"] < m["tris_high"]
    assert low.has_vertex_colors() and low.is_edge_manifold()
    # Gravity frame: world +Z (true up) -> +Y, object standing on y = 0.
    M = np.array(json.loads((job_dir / "mesh" / "transform.json").read_text())["matrix"])
    assert np.allclose(M[:3, :3] @ scene.up, [0, 1, 0], atol=1e-6)
    saved = o3d.io.read_triangle_mesh(str(job_dir / "mesh" / "high.ply"))
    assert abs(saved.get_min_bound()[1]) < 1e-6


def test_explicit_budget_and_known_dimension(job: tuple[Path, Scene], tmp_path: Path) -> None:
    job_dir = copy_job(job[0], tmp_path)
    m = mesh.run(job_dir, {"preset": "draft", "triangle_budget": 5000,
                           "known_dimension": {"axis": "y", "meters": 1.44}})
    assert m["tris_low"] <= 5000
    low = o3d.io.read_triangle_mesh(str(job_dir / "mesh" / "low.ply"))
    assert np.ptp(np.asarray(low.vertices)[:, 1]) == pytest.approx(1.44, rel=1e-6)
    scale = json.loads((job_dir / "mesh" / "transform.json").read_text())["scale"]
    assert scale == pytest.approx(2.0, rel=0.03)  # true height is 0.72 m


def test_carve_removes_outside_blob(job: tuple[Path, Scene], tmp_path: Path) -> None:
    """A blob of bogus points glued to the box side (outside every mask) must be carved away."""
    job_dir = copy_job(job[0], tmp_path)
    pcd = o3d.io.read_point_cloud(str(job_dir / "fuse" / "points.ply"))
    rng = np.random.default_rng(1)
    d = rng.standard_normal((20_000, 3))
    d[:, 0] = np.abs(d[:, 0])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    blob = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([0.3, 0.0, 0.2]) + 0.1 * d))
    blob.normals = o3d.utility.Vector3dVector(d)
    blob.colors = o3d.utility.Vector3dVector(np.full((len(d), 3), 0.5))
    o3d.io.write_point_cloud(str(job_dir / "fuse" / "points.ply"), pcd + blob)

    def blob_vertices(settings: dict) -> int:
        mesh.run(job_dir, {"preset": "draft", "mesh_method": "poisson", **settings})
        v = np.asarray(load_world_mesh(job_dir, "high").vertices)
        return int((v[:, 0] > 0.33).sum())

    assert blob_vertices({"carve_views": 10_000}) > 100  # carve disabled: the blob is meshed
    assert blob_vertices({}) < 5


def test_mesh_poisson_method_still_works(job: tuple[Path, Scene], tmp_path: Path) -> None:
    job_dir = copy_job(job[0], tmp_path)
    m = mesh.run(job_dir, {"preset": "draft", "mesh_method": "poisson"})
    assert m["method"] == "poisson" and chamfer_pct(load_world_mesh(job_dir, "high"), job[1]) < 1.0
    with pytest.raises(ValueError):
        mesh.run(job_dir, {"mesh_method": "nope"})


def test_visual_hull_matches_masks_and_depth_carves_inside(job: tuple[Path, Scene]) -> None:
    """The hull reprojects onto every mask; depth carving only removes voxels, never the surface."""
    job_dir, scene = job
    cams = scene.cameras
    occ, lo, voxel, st = hull.visual_hull(job_dir, cams, 128)
    assert st["iou_min"] > 0.8 and st["center_ray_miss_over_radius_max"] < 0.2
    occ_d, lo_d, _, st_d = hull.visual_hull(job_dir, cams, 128, depth_tol=0.03)
    assert np.allclose(lo, lo_d) and (occ_d <= occ).all() and st_d["depth_views"] == len(cams)
    # Depth carving removes empty space between hull and object (box + sphere, see synthetic.py), not the object.
    gone = hull.grid_centers(lo, voxel, occ.shape)[(occ & ~occ_d).ravel()]
    in_box = (np.abs(gone[:, 0]) < 0.3 - voxel) & (np.abs(gone[:, 1]) < 0.2 - voxel) & (gone[:, 2] > voxel) & (gone[:, 2] < 0.4 - voxel)
    in_sphere = np.linalg.norm(gone - [0.1, 0.0, 0.5], axis=1) < 0.22 - voxel
    assert (in_box | in_sphere).mean() < 0.05
    pts, nrm = hull.surface_points(occ_d, lo, voxel)
    d = mesh_distance(pts, scene.mesh)
    assert np.median(d) < 2 * voxel  # iso-level points sit on the true surface
    outward = ((pts - np.array([0.0, 0.0, 0.3])) * nrm).sum(1) > 0
    assert outward.mean() > 0.9


def test_visual_hull_drops_a_view_inconsistent_mask(job: tuple[Path, Scene], tmp_path: Path) -> None:
    """One bad (empty-region) mask must not cut the hull: all-but-one views must agree."""
    job_dir = copy_job(job[0], tmp_path)
    cams = job[1].cameras

    mp = fuse.find_mask(job_dir, cams[0])
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    m[: m.shape[0] // 2] = 0  # half the object vanishes in one view
    cv2.imwrite(str(mp), m)
    _, _, _, st = hull.visual_hull(job_dir, cams, 96)
    assert st["iou_min"] > 0.4 and np.median(list(st["iou_per_view"].values())) > 0.8
