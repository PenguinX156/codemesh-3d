"""Final asset checks and capture advice; writes quality/report.json."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

Progress = Callable[[float, str], None] | None
HARD_CHECKS = frozenset({
    "single_component", "watertight", "manifold", "no_degenerate_faces",
    "triangle_count", "has_uv_and_textures", "silhouette_iou",
})
TOPOLOGY_CHECKS = frozenset({
    "single_component", "watertight", "manifold", "winding_consistent",
    "no_degenerate_faces", "no_duplicate_faces", "scale_sane",
})


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _textures(parts: list[trimesh.Trimesh]) -> dict[str, bool]:
    present = {"uv": bool(parts), "base_color_texture": bool(parts), "normal_texture": bool(parts)}
    for part in parts:
        uv = getattr(part.visual, "uv", None)
        present["uv"] &= (uv is not None and np.shape(uv) == (len(part.vertices), 2)
                          and bool(np.isfinite(uv).all()))
        material = getattr(part.visual, "material", None)
        materials = getattr(material, "materials", [material])
        face_materials = getattr(part.visual, "face_materials", None)
        if face_materials is not None:
            materials = [materials[i] for i in np.unique(face_materials)]
        present["base_color_texture"] &= bool(materials) and all(
            getattr(m, "baseColorTexture", None) is not None or getattr(m, "image", None) is not None
            for m in materials)
        present["normal_texture"] &= bool(materials) and all(
            getattr(m, "normalTexture", None) is not None for m in materials)
    return present


def _capture_hints(job_dir: Path) -> tuple[list[str], float | None, int]:
    manifest = job_dir / "frames" / "frames.json"
    if not manifest.is_file():
        manifest = job_dir / "frames.json"
    frames = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
    entries = frames.get("frames", []) if isinstance(frames, dict) else frames
    masks_dir = job_dir / "masks"
    if entries:
        masks = []
        for entry in entries:
            index = int(entry["index"])
            stems = list(dict.fromkeys([Path(entry.get("file", str(index))).stem,
                                       str(index), f"{index:04d}", f"{index:06d}"]))
            path = next((masks_dir / f"{stem}.png" for stem in stems
                         if (masks_dir / f"{stem}.png").is_file()), None)
            if path is not None:
                masks.append(path)
    else:
        masks = sorted(p for p in masks_dir.glob("*.png") if p.stem != "preview")
    coverage = []
    for path in masks:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
            coverage.append(float(np.count_nonzero(pixels) / pixels.size))
    mean = float(np.mean(coverage)) if coverage else None
    hints = []
    if mean is not None and mean < .05:
        hints.append(f"Move closer: the object fills only {mean * 100:.1f}% of the frame.")
    dropped = 0
    valid_path = masks_dir / "valid.json"
    if valid_path.is_file():
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        if isinstance(valid, list):
            # segment writes a list of retained frame indices, not a drop count.
            if entries:
                dropped = len({int(e["index"]) for e in entries} - set(valid))
            else:
                dropped = max(0, len(masks) - len(set(valid)))
        elif isinstance(valid, dict):
            excluded = valid.get("dropped", [])
            dropped = len(excluded) if isinstance(excluded, list) else int(excluded)
    if dropped:
        hints.append(f"{dropped} frame{'s were' if dropped != 1 else ' was'} dropped because the object mask was unusable. "
                     "Keep the whole object visible while capturing.")
    return hints, mean, dropped


def run(job_dir: Path, settings: dict, progress: Progress = None) -> dict:
    """Check the exported GLB without repairing it; return the quality summary.

    Only HARD_CHECKS determine ``passed``. Winding, duplicate faces, scale,
    boundary F-score and Chamfer remain visible diagnostics. Coincident GLB
    vertices are welded for topology only, preserving UV seams for inspection.
    Known metric scale is already applied by export; sizes here are not rescaled.
    """
    start = time.perf_counter()
    job_dir = Path(job_dir)
    notify = progress or (lambda fraction, message: None)
    notify(0, "Checking exported asset")
    scene = trimesh.load(job_dir / "export" / "model.glb", force="scene", process=False)
    parts = [part for part in scene.dump() if isinstance(part, trimesh.Trimesh) and len(part.faces)]
    texture_status = _textures(parts)
    combined = trimesh.util.concatenate(parts)
    # Exact position welding joins UV/normal seams without hiding bad faces.
    vertices, inverse = np.unique(combined.vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[combined.faces], process=False)
    triangles = len(mesh.faces)
    finite = bool(len(vertices) and np.isfinite(vertices).all())
    extents = np.ptp(vertices, axis=0) if finite else np.zeros(3)
    diagonal = float(np.linalg.norm(extents))
    scale_ok = finite and np.isfinite(diagonal) and diagonal > 0
    labels = trimesh.graph.connected_component_labels(mesh.edges_unique, node_count=len(vertices))
    counts = np.bincount(labels[mesh.faces[:, 0]]) if triangles else np.array([], dtype=int)
    components = int(np.count_nonzero(counts))
    outside = float(1 - counts.max() / triangles) if triangles else 0.0
    _, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    nonmanifold = int(np.count_nonzero(edge_counts > 2))
    area_threshold = 1e-12 * diagonal**2 if scale_ok else 0.0
    areas = mesh.area_faces
    degenerate = int(np.count_nonzero(~np.isfinite(areas) | (areas < area_threshold) | (areas == 0)))
    duplicates = triangles - len(np.unique(np.sort(mesh.faces, axis=1), axis=0))
    watertight = bool(triangles and mesh.is_watertight)
    winding = bool(triangles and mesh.is_winding_consistent)
    minimum, maximum = int(settings.get("min_tris", 500)), int(settings.get("max_tris", 200000))
    if minimum < 0 or maximum < minimum:
        raise ValueError("Expected 0 <= min_tris <= max_tris")
    target = float(settings.get("iou_target", .90))
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("iou_target must be between 0 and 1")
    checks = []

    def check(name, passed, value, threshold, message):
        checks.append({"name": name, "passed": bool(passed), "value": value,
                       "threshold": threshold, "message": message, "hard": name in HARD_CHECKS})

    check("single_component", components == 1, components, 1,
          f"{components} connected bodies; {outside:.2%} of faces outside the largest body.")
    checks[-1]["outside_largest_face_share"] = outside
    check("watertight", watertight, watertight, True, "Surface is closed." if watertight else "Surface has open or invalid edges.")
    check("manifold", nonmanifold == 0, nonmanifold, 0, f"{nonmanifold} edges shared by more than two faces.")
    check("winding_consistent", winding, winding, True, "Adjacent faces must have consistent winding.")
    check("no_degenerate_faces", degenerate == 0, degenerate, 0,
          f"{degenerate} degenerate faces; area cutoff {area_threshold:.6g} square asset units.")
    check("no_duplicate_faces", duplicates == 0, duplicates, 0, f"{duplicates} duplicate faces (regardless of winding).")
    check("triangle_count", minimum <= triangles <= maximum, triangles, [minimum, maximum],
          f"{triangles} triangles; expected {minimum} to {maximum}.")
    check("has_uv_and_textures", all(texture_status.values()), texture_status,
          {name: True for name in texture_status}, "Every mesh needs finite UVs, a base color texture and a normal texture.")
    size = {"bbox_diagonal": _number(diagonal), "units": "meters" if settings.get("known_dimension") else "asset units"}
    if settings.get("known_dimension") and scale_ok:
        size["size_meters"] = dict(zip("xyz", extents.tolist()))
    check("scale_sane", scale_ok, size, "finite bbox diagonal > 0", "Size is measured from the exported bounding box.")
    notify(.6, "Checking reconstruction accuracy")
    metrics_path = job_dir / "evaluate" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    # evaluate emits null when there are no held-out views: use the all-view IoU.
    iou = _number(metrics.get("iou_holdout_mean"))
    source = "iou_holdout_mean"
    if iou is None:
        iou, source = _number(metrics.get("iou_mean")), "iou_mean"
    check("silhouette_iou", iou is not None and target <= iou <= 1, iou, target,
          f"Silhouette IoU from {source}." if iou is not None else "No finite silhouette IoU is available; run evaluation.")
    boundary, chamfer = _number(metrics.get("boundary_f_mean")), _number(metrics.get("chamfer_pct_mean"))
    check("boundary_f", boundary is not None and 0 <= boundary <= 1, boundary, None,
          "Boundary F-score (informational; higher is better).")
    check("chamfer_pct", chamfer is not None and 0 <= chamfer <= 1, chamfer, 1.0,
          "Mean surface distance as a percentage of the diagonal (informational).")
    failed = [c["name"] for c in checks if c["hard"] and not c["passed"]]
    topology_ok = all(c["passed"] for c in checks if c["name"] in TOPOLOGY_CHECKS)
    grade = "F" if not topology_ok else (
        "A" if not failed and iou is not None and iou >= .9 else
        "B" if iou is not None and iou >= .8 else "C")
    hints, coverage, dropped = _capture_hints(job_dir)
    if iou is not None and iou < target:
        hints.append("Walk a full circle around the object and include a short look from above.")
    summary = {"passed": not failed, "grade": grade, "failed_checks": failed,
               "silhouette_iou": iou, "components": components, "watertight": watertight,
               "triangles": triangles, "hints": hints}
    report = {**summary, "checks": checks, "outside_largest_face_share": outside,
              "mask_coverage_mean": coverage, "dropped_frames": dropped,
              "psnr_mean": _number(metrics.get("psnr_mean")), "seconds": time.perf_counter() - start}
    out = job_dir / "quality"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    notify(1, f"Quality grade {grade}")
    return summary
