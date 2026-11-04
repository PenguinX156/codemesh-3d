"""Compare asset-frame meshes with original photos, masks and world-frame points.

UVs are V-up; albedo is evaluated without lighting. PSNR is capped at 100 dB
for an exact match. Missing/empty appearance intersections are JSON null.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from codemesh.pipeline.cameras import Camera, load_cameras

Progress = Callable[[float, str], None] | None


def _transform(job_dir: Path) -> np.ndarray:
    path = job_dir / "mesh" / "transform.json"
    if not path.exists():
        return np.eye(4)
    data = json.loads(path.read_text())
    for key in ("world_to_asset", "matrix", "transform"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float64).reshape(4, 4)
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite transform: {path}")
            return matrix
    raise ValueError(f"{path}: missing world_to_asset/matrix/transform")


def _scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
    return scene


def _textures(job_dir: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load the atlas mesh, or exported GLB nodes with their node transforms applied."""
    atlas = job_dir / "texture" / "albedo.png"
    albedo = cv2.imread(str(atlas)) if atlas.exists() else None
    path = job_dir / "texture" / "uv_mesh.npz"
    if path.exists() and albedo is not None:
        with np.load(path, allow_pickle=False) as data:
            vertices, faces, uv = (data[key].copy() for key in ("vertices", "faces", "uv"))
        return [(vertices, faces, uv, albedo)]
    scene = trimesh.load(job_dir / "export" / "model.glb", force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(matrix)
        uv = getattr(mesh.visual, "uv", None)
        material = getattr(mesh.visual, "material", None)
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        texture = np.asarray(image.convert("RGB"))[..., ::-1].copy() if image is not None else albedo
        if uv is None or texture is None:
            raise ValueError(f"GLB node {node!r} has no UVs/albedo")
        parts.append((np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(uv), texture))
    if not parts:
        raise ValueError("No textured geometry in export/model.glb")
    return parts


def _rays(cam: Camera, inverse_transform: np.ndarray, size: tuple[int, int]) -> o3d.core.Tensor:
    w, h = size
    sx, sy = w / cam.width, h / cam.height
    # OpenCV resize maps destination pixel centers to (u + .5) / scale - .5.
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    K = resize @ cam.K_np
    E = cam.extrinsic_np @ inverse_transform
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    directions = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    inverse_rotation = np.linalg.inv(E[:3, :3])
    directions = directions @ inverse_rotation.T
    origin = np.broadcast_to(-inverse_rotation @ E[:3, 3], directions.shape)
    return o3d.core.Tensor(np.concatenate([origin, directions], -1).astype(np.float32))


def _render(scene: o3d.t.geometry.RaycastingScene, parts: list, rays: o3d.core.Tensor) -> tuple[np.ndarray, np.ndarray]:
    result = scene.cast_rays(rays, nthreads=4)
    hit = np.isfinite(result["t_hit"].numpy())
    rgb = np.full((*hit.shape, 3), 235, dtype=np.uint8)
    ids = result["primitive_ids"].numpy()
    geometry = result["geometry_ids"].numpy()
    bary = result["primitive_uvs"].numpy()
    for geometry_id, (_, faces, uv, albedo) in enumerate(parts):
        selected = hit & (geometry == geometry_id)
        if not selected.any():
            continue
        weights = bary[selected]
        weights = np.column_stack([1 - weights.sum(1), weights])
        corners = uv[faces[ids[selected]]]
        coords = np.einsum("ni,nij->nj", weights, corners)
        # cv2.remap dimensions must stay below 32767 even for flat hit arrays.
        n, stride = len(coords), 2048
        rows = (n + stride - 1) // stride
        maps = np.zeros((rows * stride, 2), dtype=np.float32)
        maps[:n, 0] = coords[:, 0] * albedo.shape[1] - .5
        maps[:n, 1] = (1 - coords[:, 1]) * albedo.shape[0] - .5
        samples = cv2.remap(albedo, maps[:, 0].reshape(rows, stride), maps[:, 1].reshape(rows, stride),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb[selected] = samples.reshape(-1, 3)[:n]
    return rgb, hit


def _silhouette(render: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    union = np.count_nonzero(render | mask)
    iou = float(np.count_nonzero(render & mask) / union) if union else 1.0
    kernel = np.ones((3, 3), np.uint8)
    boundaries = [m & ~cv2.erode(m.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                borderValue=0).astype(bool) for m in (render, mask)]
    a, b = boundaries
    if not a.any() or not b.any():
        return iou, float(not a.any() and not b.any())
    distance_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = float(np.mean(distance_b[a] <= 2))
    recall = float(np.mean(distance_a[b] <= 2))
    return iou, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _appearance(photo: np.ndarray, render: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    if not mask.any():
        return None, None
    # Normalize Gaussian statistics by valid weight: background cannot inflate SSIM.
    y, x = np.nonzero(mask)
    region = np.s_[max(0, y.min() - 5):y.max() + 6, max(0, x.min() - 5):x.max() + 6]
    mask = mask[region]
    a, b = photo[region].astype(np.float32) / 255, render[region].astype(np.float32) / 255
    mse = float(np.mean((a[mask] - b[mask]) ** 2))
    psnr = float(-10 * np.log10(max(mse, 1e-10)))
    weight = mask.astype(np.float32)[..., None]
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), 1.5)
    norm = np.maximum(blur(weight[..., 0])[..., None], 1e-8)
    mu_a, mu_b = blur(a * weight) / norm, blur(b * weight) / norm
    var_a = np.maximum(blur(a * a * weight) / norm - mu_a * mu_a, 0)
    var_b = np.maximum(blur(b * b * weight) / norm - mu_b * mu_b, 0)
    cov = blur(a * b * weight) / norm - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + .01**2) * (2 * cov + .03**2)
            / ((mu_a**2 + mu_b**2 + .01**2) * (var_a + var_b + .03**2)))
    return psnr, float(np.clip(ssim[mask].mean(), -1, 1))


def _chamfer(mesh: trimesh.Trimesh, job_dir: Path, transform: np.ndarray) -> dict:
    path = job_dir / "fuse" / "points.ply"
    if not path.exists():
        return {"chamfer_pct_mean": None, "chamfer_pct_p90": None}
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        raise ValueError(f"Empty point cloud: {path}")
    points = points @ transform[:3, :3].T + transform[:3, 3]
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diagonal <= 0:
        raise ValueError("Fused point cloud has a zero bbox diagonal")
    rng = np.random.default_rng(0)
    if len(points) > 20000:
        points = points[rng.choice(len(points), 20000, replace=False)]
    triangles = mesh.triangles
    areas = mesh.area_faces
    chosen = rng.choice(len(areas), 20000, p=areas / areas.sum())
    bary = rng.random((len(chosen), 2))
    bary[bary.sum(1) > 1] = 1 - bary[bary.sum(1) > 1]
    corners = triangles[chosen]
    samples = corners[:, 0] + bary[:, :1] * (corners[:, 1] - corners[:, 0]) + bary[:, 1:] * (corners[:, 2] - corners[:, 0])
    a = cKDTree(points).query(samples)[0]
    b = cKDTree(samples).query(points)[0]
    return {"chamfer_pct_mean": float((a.mean() + b.mean()) * .5 / diagonal * 100),
            "chamfer_pct_p90": float(np.percentile(np.concatenate([a, b]), 90) / diagonal * 100),
            "bbox_diagonal": diagonal, "bbox_source": "fuse/points.ply transformed to asset frame",
            "surface_samples": len(samples), "point_samples": len(points)}


def _evidence(job_dir: Path, cam: Camera) -> tuple[Path, Path]:
    stems = list(dict.fromkeys([Path(cam.image).stem, str(cam.index), f"{cam.index:04d}", f"{cam.index:06d}"]))
    photo = job_dir / cam.image
    if not photo.is_file():
        photo = next((job_dir / "frames" / f"{stem}{ext}" for stem in stems for ext in (".png", ".jpg", ".jpeg")
                      if (job_dir / "frames" / f"{stem}{ext}").is_file()), photo)
    mask = next((job_dir / "masks" / f"{stem}.png" for stem in stems
                 if (job_dir / "masks" / f"{stem}.png").is_file()), job_dir / "masks" / f"{cam.index}.png")
    return photo, mask


def _comparison(path: Path, photo: np.ndarray, render: np.ndarray, hit: np.ndarray, mask: np.ndarray, metrics: dict) -> None:
    diff = np.full_like(photo, 28)
    diff[hit & mask], diff[hit & ~mask], diff[~hit & mask] = (40, 205, 40), (40, 40, 235), (235, 90, 40)
    height, width = photo.shape[:2]
    size = (min(width, 533), max(1, round(height * min(width, 533) / width)))
    panels = [cv2.resize(img, size, interpolation=cv2.INTER_AREA) for img in (photo, render, diff)]
    canvas = np.full((size[1] + 68, size[0] * 3, 3), 245, np.uint8)
    canvas[34:-34] = np.concatenate(panels, axis=1)
    for i, title in enumerate(("Original photo", "Texture render", "Silhouette: green=match")):
        cv2.putText(canvas, title, (i * size[0] + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
    text = f"View {metrics['index']}   IoU {metrics['iou']:.3f}   red=render only   blue=mask only"
    cv2.putText(canvas, text, (8, canvas.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 25, 25), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot write {path}")


def _aggregate(views: list[dict]) -> dict:
    result = {"views_evaluated": len(views)}
    for name in ("iou", "boundary_f", "psnr", "ssim"):
        values = [v[name] for v in views if v[name] is not None]
        result[f"{name}_mean"] = float(np.mean(values)) if values else None
    return result


def run(job_dir: Path, settings: dict, progress: Progress = None) -> dict:
    """Write evaluate/metrics.json and up to six labeled comparisons; return summary.

    settings: eval_long_side (default 1024, no upscaling), holdout_indices.
    Chamfer is symmetric, unsquared and normalized by the fused cloud bbox.
    """
    start = time.perf_counter()
    job_dir = Path(job_dir)
    report = progress or (lambda fraction, message: None)
    long_side = int(settings.get("eval_long_side", 1024))
    if long_side <= 0:
        raise ValueError("eval_long_side must be positive")
    holdout = {int(index) for index in settings.get("holdout_indices", [])}
    cams, _ = load_cameras(job_dir / "geometry" / "cameras.json")
    transform = _transform(job_dir)
    inverse_transform = np.linalg.inv(transform)
    mesh = trimesh.load(job_dir / "mesh" / "low.ply", force="mesh", process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError("Low mesh must have finite vertices and nonzero surface area")
    low_scene = _scene(mesh.vertices, mesh.faces)
    parts = _textures(job_dir)
    texture_scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    for vertices, faces, uv, _ in parts:
        if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
            raise ValueError("Expected finite per-vertex UVs of shape (N, 2)")
        texture_scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                                    o3d.core.Tensor(faces.astype(np.uint32)))
    out = job_dir / "evaluate"
    out.mkdir(parents=True, exist_ok=True)
    # Only stale evaluator artifacts are removed on reruns.
    for path in out.glob("compare_*.png"):
        path.unlink()
    views, skipped = [], []
    compare_indices = set(np.linspace(0, len(cams) - 1, min(6, len(cams)), dtype=int))
    report(0, "Evaluating original photos and masks")
    for i, cam in enumerate(cams):
        photo_path, mask_path = _evidence(job_dir, cam)
        if not photo_path.is_file() or not mask_path.is_file():
            skipped.append({"index": cam.index, "reason": "Missing original photo or mask"})
            continue
        photo, mask = cv2.imread(str(photo_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if photo is None or mask is None:
            raise ValueError(f"Unreadable evidence for camera {cam.index}")
        if photo.shape[:2] != (cam.height, cam.width) or mask.shape != photo.shape[:2]:
            raise ValueError(f"Camera {cam.index}: original photo/mask dimensions disagree with cameras.json")
        scale = min(1.0, long_side / max(cam.width, cam.height))
        size = (max(1, round(cam.width * scale)), max(1, round(cam.height * scale)))
        photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0
        rays = _rays(cam, inverse_transform, size)
        hit = np.isfinite(low_scene.cast_rays(rays, nthreads=4)["t_hit"].numpy())
        render, texture_hit = _render(texture_scene, parts, rays)
        iou, boundary_f = _silhouette(hit, mask)
        intersection = hit & texture_hit & mask
        psnr, ssim = _appearance(photo, render, intersection)
        view = {"index": cam.index, "holdout": cam.index in holdout, "width": size[0], "height": size[1],
                "iou": iou, "boundary_f": boundary_f, "psnr": psnr, "ssim": ssim,
                "appearance_pixels": int(intersection.sum())}
        views.append(view)
        if i in compare_indices:
            _comparison(out / f"compare_{cam.index}.png", photo, render, hit, mask, view)
        report((i + 1) / max(1, len(cams)) * .95, f"Evaluated view {cam.index}")
    if not views:
        raise ValueError("No cameras with original photos and masks to evaluate")
    geometry = _chamfer(mesh, job_dir, transform)
    all_views = _aggregate(views)
    held_out = _aggregate([v for v in views if v["holdout"]])
    summary = {**all_views, "iou_holdout_mean": held_out["iou_mean"],
               "chamfer_pct_mean": geometry["chamfer_pct_mean"], "seconds": time.perf_counter() - start}
    metrics = {**summary, "all": all_views, "holdout": held_out, "geometry": geometry, "views": views,
               "skipped": skipped, "holdout_indices": sorted(holdout), "psnr_cap_db": 100,
               "boundary_tolerance_px": 2, "ssim": "RGB, 11x11 Gaussian sigma=1.5, mask-normalized moments"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    report(1, "Evaluation done")
    return summary
