"""Exported-asset quality gate, capture hints and 50k-triangle runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from codemesh.pipeline import quality


def _textured(mesh: trimesh.Trimesh, normal: bool = True) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (4, 4), (180, 100, 40)),
            normalTexture=Image.new("RGB", (4, 4), (128, 128, 255)) if normal else None,
        ),
    )
    return mesh


def _job(path: Path, mesh=None, metrics=None) -> Path:
    mesh = _textured(trimesh.creation.icosphere(subdivisions=3)) if mesh is None else mesh
    (path / "export").mkdir(parents=True, exist_ok=True)
    mesh.export(path / "export" / "model.glb")
    (path / "evaluate").mkdir(exist_ok=True)
    metrics = {"iou_mean": .95, "boundary_f_mean": .93,
               "psnr_mean": 28.5, "chamfer_pct_mean": .4} if metrics is None else metrics
    (path / "evaluate" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return path


def _report(path: Path) -> tuple[dict, dict]:
    report = json.loads((path / "quality" / "report.json").read_text(encoding="utf-8"))
    return report, {c["name"]: c for c in report["checks"]}


def test_textured_icosphere_passes(tmp_path):
    _job(tmp_path)
    progress = []
    result = quality.run(tmp_path, {}, lambda fraction, message: progress.append((fraction, message)))
    assert result == {"passed": True, "grade": "A", "failed_checks": [], "silhouette_iou": .95,
                      "components": 1, "watertight": True, "triangles": 1280, "hints": []}
    report, checks = _report(tmp_path)
    assert all(c["passed"] for c in checks.values())
    assert all({"name", "passed", "value", "threshold", "message"} <= c.keys() for c in checks.values())
    assert report["psnr_mean"] == 28.5
    assert progress[0][0] == 0 and progress[-1][0] == 1


def test_disconnected_sphere_reports_face_share(tmp_path):
    large = _textured(trimesh.creation.icosphere(subdivisions=3))
    small = _textured(trimesh.creation.icosphere(subdivisions=1, radius=.1))
    small.apply_translation([3, 0, 0])
    _job(tmp_path, trimesh.Scene([large, small]))
    result = quality.run(tmp_path, {})
    assert not result["passed"] and result["grade"] == "F"
    assert result["components"] == 2 and result["watertight"]
    assert "single_component" in result["failed_checks"]
    report, _ = _report(tmp_path)
    assert report["outside_largest_face_share"] == pytest.approx(80 / 1360)


def test_hole_fails_watertight(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    _job(tmp_path, _textured(mesh))
    result = quality.run(tmp_path, {})
    assert not result["watertight"] and result["grade"] == "F"
    assert "watertight" in result["failed_checks"]


def test_uv_seams_are_not_disconnected_bodies(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    vertices = sphere.vertices[sphere.faces].reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1, 3), process=False)
    _job(tmp_path, _textured(mesh))
    assert quality.run(tmp_path, {})["passed"]


@pytest.mark.parametrize("iou,grade", [(.92, "A"), (.85, "B"), (.72, "C")])
def test_holdout_iou_controls_accuracy_and_grade(tmp_path, iou, grade):
    _job(tmp_path, metrics={"iou_mean": .99, "iou_holdout_mean": iou})
    result = quality.run(tmp_path, {})
    assert result["silhouette_iou"] == iou and result["grade"] == grade
    assert result["passed"] == (iou >= .9)
    assert ("Walk a full circle around the object and include a short look from above." in result["hints"]) == (iou < .9)


def test_null_holdout_falls_back_and_informational_metrics_do_not_fail(tmp_path):
    _job(tmp_path, metrics={"iou_mean": .95, "iou_holdout_mean": None,
                           "boundary_f_mean": .2, "chamfer_pct_mean": 4.5})
    result = quality.run(tmp_path, {})
    assert result["passed"] and result["grade"] == "A"
    _, checks = _report(tmp_path)
    assert not checks["chamfer_pct"]["passed"] and not checks["chamfer_pct"]["hard"]


def test_missing_accuracy_fails_closed(tmp_path):
    _job(tmp_path)
    (tmp_path / "evaluate" / "metrics.json").unlink()
    result = quality.run(tmp_path, {})
    assert result["silhouette_iou"] is None
    assert result["failed_checks"] == ["silhouette_iou"]


@pytest.mark.parametrize("textured,normal", [(False, False), (True, False)])
def test_missing_uv_or_textures_fails(tmp_path, textured, normal):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    _job(tmp_path, _textured(mesh, normal=normal) if textured else mesh)
    result = quality.run(tmp_path, {})
    assert "has_uv_and_textures" in result["failed_checks"]


def test_every_scene_part_needs_textures(tmp_path):
    good = _textured(trimesh.creation.icosphere(subdivisions=3))
    bare = trimesh.creation.icosphere(subdivisions=1)
    bare.apply_translation([4, 0, 0])
    _job(tmp_path, trimesh.Scene([good, bare]))
    assert "has_uv_and_textures" in quality.run(tmp_path, {})["failed_checks"]


def test_degenerate_duplicate_and_nonmanifold_faces(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    faces = np.vstack([sphere.faces, sphere.faces[0, ::-1], [0, 0, 1]])
    _job(tmp_path, _textured(trimesh.Trimesh(sphere.vertices, faces, process=False)))
    result = quality.run(tmp_path, {})
    _, checks = _report(tmp_path)
    assert checks["no_degenerate_faces"]["value"] == 1
    assert checks["no_duplicate_faces"]["value"] == 1
    assert checks["manifold"]["value"] >= 3
    assert {"manifold", "no_degenerate_faces"} <= set(result["failed_checks"])
    assert result["grade"] == "F"


def test_inconsistent_winding_is_reported(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    sphere.faces[0] = sphere.faces[0, ::-1]
    _job(tmp_path, _textured(sphere))
    quality.run(tmp_path, {})
    _, checks = _report(tmp_path)
    assert not checks["winding_consistent"]["passed"]


def test_triangle_limits_iou_target_and_metric_size(tmp_path):
    _job(tmp_path, metrics={"iou_mean": .85})
    settings = {"min_tris": 1280, "max_tris": 1280, "iou_target": .8,
                "known_dimension": {"axis": "x", "meters": 2}}
    result = quality.run(tmp_path, settings)
    assert result["passed"] and result["grade"] == "B"
    _, checks = _report(tmp_path)
    assert checks["scale_sane"]["value"]["size_meters"] == pytest.approx({"x": 2, "y": 2, "z": 2})
    result = quality.run(tmp_path, {**settings, "max_tris": 1279, "min_tris": 500})
    assert result["failed_checks"] == ["triangle_count"]


def test_mask_coverage_and_dropped_frame_hints(tmp_path):
    _job(tmp_path)
    (tmp_path / "frames").mkdir()
    (tmp_path / "masks").mkdir()
    frames = {"frames": [{"index": i, "file": f"{i:04d}.jpg", "width": 100, "height": 100} for i in range(3)]}
    (tmp_path / "frames" / "frames.json").write_text(json.dumps(frames))
    (tmp_path / "masks" / "valid.json").write_text("[0, 2]")
    pixels = np.zeros((100, 100), np.uint8)
    pixels[:10, :20] = 255
    for i in range(3):
        Image.fromarray(pixels).save(tmp_path / "masks" / f"{i:04d}.png")
    result = quality.run(tmp_path, {})
    assert "Move closer: the object fills only 2.0% of the frame." in result["hints"]
    assert any("1 frame was dropped" in hint for hint in result["hints"])
    report, _ = _report(tmp_path)
    assert report["mask_coverage_mean"] == pytest.approx(.02)
    assert report["dropped_frames"] == 1


def test_scene_transform_is_applied_to_size(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(_textured(trimesh.creation.icosphere(subdivisions=3)),
                       transform=np.diag([2., 3., 4., 1.]))
    _job(tmp_path, scene)
    assert quality.run(tmp_path, {"known_dimension": {"axis": "x", "meters": 4}})["passed"]
    _, checks = _report(tmp_path)
    assert checks["scale_sane"]["value"]["size_meters"] == pytest.approx({"x": 4, "y": 6, "z": 8})


def test_50k_triangle_runtime(tmp_path):
    sphere = _textured(trimesh.creation.uv_sphere(count=[128, 100]))
    assert 49000 <= len(sphere.faces) <= 52000
    _job(tmp_path, sphere)
    start = time.perf_counter()
    result = quality.run(tmp_path, {})
    elapsed = time.perf_counter() - start
    assert result["passed"]
    assert elapsed < 5, f"Quality gate took {elapsed:.3f}s for {len(sphere.faces)} triangles"
    print(f"Quality runtime: {elapsed:.3f}s for {len(sphere.faces)} triangles")
