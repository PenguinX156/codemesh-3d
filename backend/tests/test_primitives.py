"""Evidence-fitted primitives on synthetic hulls (asset frame, +Y up)."""

from __future__ import annotations

import numpy as np
import trimesh

from codemesh.pipeline import primitives


def _mug(handle: bool) -> trimesh.Trimesh:
    body = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=64)
    body.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # axis -> Y
    body.apply_translation([0.01, 0.05, -0.02])
    if not handle:
        return body
    ring = trimesh.creation.torus(0.03, 0.008)  # in the XY plane after rotating about X
    ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    ring.apply_translation([0.01 + 0.06, 0.05, -0.02])
    return trimesh.util.concatenate([body, ring])


def test_lathe_fits_cylinder_and_is_watertight() -> None:
    ev = _mug(False)
    mesh, fit = primitives.lathe(ev, segments=48)
    assert mesh.is_watertight and mesh.volume > 0
    assert abs(fit["radius_max"] - 0.05) < 0.005
    assert np.allclose(fit["axis_xz"], [0.01, -0.02], atol=0.003)
    assert primitives.voxel_iou(mesh, ev) > 0.85
    hollow, _ = primitives.lathe(ev, hollow=True, segments=48)
    assert hollow.is_watertight and 0 < hollow.volume < 0.5 * mesh.volume


def test_lathe_ignores_handle_and_handle_is_fitted() -> None:
    ev = _mug(True)
    mesh, fit = primitives.lathe(ev, segments=48)
    assert abs(fit["radius_max"] - 0.05) < 0.006  # the handle does not inflate the profile
    tube, rep = primitives.handle(ev, fit)
    assert tube is not None and tube.is_watertight, rep
    assert tube.bounds[1][0] > 0.09  # reaches out to the handle (ring outer x = 0.07 + 0.03 + 0.008)
    assert 0.004 < rep["tube_radius"] < 0.015


def test_no_handle_on_plain_body() -> None:
    ev = _mug(False)
    _, fit = primitives.lathe(ev)
    tube, rep = primitives.handle(ev, fit)
    assert tube is None and "skipped" in rep


def test_rounded_box_fits_rotated_box() -> None:
    ev = trimesh.creation.box([0.3, 0.2, 0.1])
    ev.apply_transform(trimesh.transformations.rotation_matrix(0.4, [0, 1, 0]))
    mesh, rep = primitives.rounded_box(ev, bevel=0.1)
    assert mesh.is_watertight
    assert sorted(rep["extent"]) == sorted(np.round(rep["extent"], 5).tolist())
    assert primitives.voxel_iou(mesh, ev) > 0.85


def test_tube_is_closed() -> None:
    t = np.linspace(0, np.pi, 20)
    tube = primitives.tube(np.c_[np.cos(t), np.sin(t), np.zeros_like(t)], 0.1)
    assert tube.is_watertight and tube.volume > 0


def test_agent_primitive_tools() -> None:
    import pytest

    from codemesh.agent import tools

    plan = tools.Plan()
    with pytest.raises(ValueError):
        tools.apply(plan, "add_handle", {})
    assert tools.apply(plan, "build_lathe", {"hollow": True, "segments": 64})[1] == "mesh"
    tools.apply(plan, "add_handle", {})
    assert plan.primitive == {"kind": "lathe", "hollow": True, "segments": 64, "with_handle": True}
    assert "primitive" not in plan.settings()
    tools.apply(plan, "use_reconstruction", {"method": "hull_depth"})
    assert plan.primitive is None and plan.mesh_method == "hull_depth"
