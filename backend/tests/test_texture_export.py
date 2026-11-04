"""Texture + export stages on synthetic scenes.

Run the 2048/4096 runtime benchmark (32 views, ~30k-tri low mesh) with:
    uv run --project backend python backend/tests/test_texture_export.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from codemesh.pipeline import export, texture
from codemesh.pipeline.cameras import Camera, load_cameras, save_cameras
from synthetic import make_scene

# World (Z up) -> asset (Y up, x2 scale, shifted): exercises the camera transform path.
WORLD_TO_ASSET = np.array([[2.0, 0, 0, 0.1], [0, 0, 2.0, -0.2], [0, -2.0, 0, 0.3], [0, 0, 0, 1]])
HELD_OUT = (3, 19)


def write_meshes(job: Path, high: o3d.geometry.TriangleMesh, low_tris: int, T: np.ndarray) -> None:
    (job / "mesh").mkdir(parents=True, exist_ok=True)
    low = high.simplify_quadric_decimation(low_tris)
    for name, m in (("high", high), ("low", low)):
        m = o3d.geometry.TriangleMesh(m).transform(T)
        m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(job / "mesh" / f"{name}.ply"), m)
    (job / "mesh" / "transform.json").write_text(json.dumps({"world_to_asset": T.tolist()}))


def prepare_job(job: Path, n_ring: int = 16, n_top: int = 8, subdivide: int = 0, low_tris: int = 2000) -> list[Camera]:
    """Synthetic job with mesh/ inputs; returns the held-out cameras (removed from cameras.json)."""
    scene = make_scene(job, n_ring=n_ring, n_top=n_top)
    high = scene.mesh.subdivide_midpoint(subdivide) if subdivide else scene.mesh
    write_meshes(job, high, low_tris, WORLD_TO_ASSET)
    cams, meta = load_cameras(job / "geometry" / "cameras.json")
    held = [c for c in cams if c.index in HELD_OUT]
    save_cameras(job / "geometry" / "cameras.json", [c for c in cams if c.index not in HELD_OUT], meta["metric_scale"])
    return held


def held_out_psnr(job: Path, held: list[Camera]) -> float:
    """Masked PSNR of the unshaded textured low mesh vs. the ground-truth frames of held-out views."""
    mesh = dict(np.load(job / "texture" / "uv_mesh.npz"))
    albedo = cv2.imread(str(job / "texture" / "albedo.png"))[..., ::-1].copy()
    err, n = 0.0, 0
    for cam in held:
        gt = cv2.imread(str(job / cam.image))[..., ::-1].astype(np.float64)
        gt_mask = cv2.imread(str(job / "masks" / Path(cam.image).name), cv2.IMREAD_GRAYSCALE) > 127
        gt_mask = cv2.erode(gt_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # skip AA-free edges
        rays = texture.camera_rays(cam.K_np, cam.extrinsic_np @ np.linalg.inv(WORLD_TO_ASSET), cam.width, cam.height)
        rgb, hit = export.render(mesh, albedo, None, rays)
        m = hit & gt_mask
        err += float(((rgb[m] - gt[m]) ** 2).sum())
        n += int(m.sum()) * 3
    return 10 * np.log10(255.0**2 / (err / n))


@pytest.fixture(scope="module")
def textured_job(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Camera], dict, dict]:
    job = tmp_path_factory.mktemp("texjob")
    held = prepare_job(job)
    tex = texture.run(job, {"texture_size": 1024})
    exp = export.run(job, {})
    return job, held, tex, exp


def test_texture_outputs_and_psnr(textured_job: tuple[Path, list[Camera], dict, dict]) -> None:
    job, held, tex, _ = textured_job
    for name in ("albedo.png", "normal.png"):
        img = cv2.imread(str(job / "texture" / name))
        assert img.shape == (1024, 1024, 3)
    assert tex["views_used"] == 22
    assert tex["seen_pct"] > 60, tex  # the union mesh has interior faces no camera can see
    assert tex["bake_hit_pct"] > 95, tex
    psnr = held_out_psnr(job, held)
    print(f"\nheld-out PSNR {psnr:.2f} dB, metrics {tex}")
    assert psnr > 20.0  # errors sit on the aliased (1 sample/pixel) hard checker edges of the GT


def test_glb_and_preview(textured_job: tuple[Path, list[Camera], dict, dict]) -> None:
    job, _, tex, exp = textured_job
    assert (job / "export" / "model.glb").stat().st_size == exp["glb_bytes"]
    assert exp["triangles"] == tex["triangles"] and exp["textures"] == 2
    doc, _ = export.glb_json((job / "export" / "model.glb").read_bytes())
    mat = doc["materials"][0]
    assert mat["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert mat["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.8)
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["TANGENT"]]
    assert acc["type"] == "VEC4" and acc["componentType"] == 5126
    preview = cv2.imread(str(job / "export" / "preview.png"))
    assert preview.shape == (768, 1024, 3) and (preview != 235).any(axis=2).mean() > 0.05


def test_normal_map_bakes_bumps(tmp_path: Path) -> None:
    """Bumpy sphere (high) onto an icosphere (low): the decoded normal map (glTF rule
    B = cross(N, T) * w) must reproduce the high normals far better than the low normals do."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=200)
    v = np.asarray(sphere.vertices)
    v *= (1 + 0.03 * np.sin(8 * v[:, 0]) * np.sin(8 * v[:, 1]) * np.sin(8 * v[:, 2]))[:, None]
    sphere.vertices = o3d.utility.Vector3dVector(v)
    write_meshes(tmp_path, sphere, 1000, np.eye(4))
    save_cameras(tmp_path / "geometry" / "cameras.json", [], metric_scale=False)
    metrics = texture.run(tmp_path, {"texture_size": 1024})
    assert metrics["bake_hit_pct"] > 98

    mesh = dict(np.load(tmp_path / "texture" / "uv_mesh.npz"))
    nmap = cv2.imread(str(tmp_path / "texture" / "normal.png"))[..., ::-1].copy()
    rng = np.random.default_rng(1)
    F = mesh["faces"]
    tri = rng.integers(0, len(F), 20000)
    b = rng.dirichlet(np.ones(3), len(tri)) * 0.9 + 0.1 / 3  # stay off the chart borders
    lerp = lambda a: np.einsum("mk,mkd->md", b, a[F[tri]])
    P, N, tan = lerp(mesh["vertices"]), lerp(mesh["normals"]), lerp(mesh["tangents"])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = tan[:, :3] - N * (N * tan[:, :3]).sum(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.cross(N, T) * np.sign(tan[:, 3:4])
    px = texture.uv_to_pixel(lerp(mesh["uv"]), 1024)
    ts = texture.sample_bilinear(nmap, px[:, 0], px[:, 1]).astype(np.float64) / 127.5 - 1
    assert np.linalg.norm(ts[:, :2], axis=1).mean() > 0.02  # not flat (measured ~0.045)
    n_baked = ts[:, :1] * T + ts[:, 1:2] * B + ts[:, 2:3] * N
    n_baked /= np.linalg.norm(n_baked, axis=1, keepdims=True)

    sphere.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sphere))
    ans = scene.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)))
    n_true = ans["primitive_normals"].numpy()
    err_baked = np.degrees(np.arccos(np.clip((n_baked * n_true).sum(1), -1, 1)))
    err_low = np.degrees(np.arccos(np.clip((N * n_true).sum(1), -1, 1)))
    print(f"\nnormal error median: baked {np.median(err_baked):.2f} deg, low mesh {np.median(err_low):.2f} deg")
    assert np.median(err_baked) < 0.5 * np.median(err_low)


def bench(root: Path) -> None:
    """Runtime at 2048 and 4096 on a ~30k-tri low mesh with 32 views (prints real numbers)."""
    root.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    held = prepare_job(root, n_ring=22, n_top=12, subdivide=2, low_tris=30000)  # 34 rendered, 32 used
    print(f"scene {time.perf_counter() - t:.1f} s")
    for size in (2048, 4096):
        tex = texture.run(root, {"texture_size": size})
        exp = export.run(root, {})
        print(f"{size}: texture {tex}\n      export {exp}\n      held-out PSNR {held_out_psnr(root, held):.2f} dB")


if __name__ == "__main__":
    bench(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("jobs/_bench_texture"))
