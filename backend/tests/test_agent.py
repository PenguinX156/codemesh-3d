from __future__ import annotations

import json

import numpy as np
import pytest

from codemesh.agent import loop, ops, tools
from codemesh.meta.muse import MuseReply, ToolCall, Usage
from test_pipeline_e2e import run_stage, setup  # noqa: F401  (fixture)


def test_lathe_closes_a_partial_arc_with_outward_normals() -> None:
    rng = np.random.default_rng(0)
    t, y = rng.uniform(0, np.pi / 2, 2000), rng.uniform(0, 1, 2000)  # a quarter of a r=0.3 cylinder
    v = np.c_[0.5 + 0.3 * np.cos(t), y, -0.2 + 0.3 * np.sin(t)]
    centre, r = ops.fit_axis(v[:, [0, 2]])
    assert np.allclose(centre, [0.5, -0.2], atol=1e-6) and r == pytest.approx(0.3)
    mesh = ops.lathe(v, 32)
    mesh.compute_triangle_normals()
    V, F, N = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.triangle_normals)
    side = F[F.max(1) < len(V) - 2]
    out = V[side].mean(1)[:, [0, 2]] - centre
    assert (np.einsum("ij,ij->i", N[F.max(1) < len(V) - 2][:, [0, 2]], out) > 0).all()
    assert mesh.is_watertight()


def test_mirror_makes_vertices_symmetric() -> None:
    v = np.array([[0.5, 0, 0], [-0.3, 0, 0], [0.1, 1, 0], [-0.1, 1, 0]])
    m = ops.mirror(v, 0)
    assert np.allclose(m[0], [0.4, 0, 0]) and np.allclose(m[1], [-0.4, 0, 0])


def test_tools_validate_inputs() -> None:
    plan = tools.Plan()
    with pytest.raises(ValueError):
        tools.apply(plan, "set_triangle_budget", {"tris": 10})
    with pytest.raises(ValueError):
        tools.apply(plan, "set_material", {"roughness": 2, "metallic": 0})
    with pytest.raises(ValueError):
        tools.apply(plan, "rm_rf", {})
    assert tools.apply(plan, "set_material", {"roughness": 0.3, "metallic": 0})[1] == "export"
    assert tools.apply(plan, "set_texture_size", {"size": 4096}) == ("texture size = 4096", "texture")
    assert plan.settings() == {"roughness": 0.3, "metallic": 0.0, "texture_size": 4096}


def _reconstruct(store, job, paths) -> None:
    run_stage(store, job, paths, "reconstruct", "fuse", {"reconstruct": {"preset": "draft"}})
    for name in ("mesh", "texture", "export", "evaluate"):
        run_stage(store, job, paths, "reconstruct", name, {})


def _fake(replies: list):
    calls = []

    def chat(messages, **kw):
        calls.append((messages, kw))
        r = replies[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r
    return chat, calls


def _reply(text: str, *calls: tuple[str, dict]) -> MuseReply:
    tc = [ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)]
    return MuseReply(text, None, tc, Usage(1000, 200, 100, 0), 1.0, "tool_calls",
                     {"role": "assistant", "content": text})


def test_agent_loop_applies_tools_and_keeps_best(setup, monkeypatch) -> None:  # noqa: F811
    store, job, paths = setup
    _reconstruct(store, job, paths)
    iou0 = store.get(job.id).metrics["iou_mean"]
    analysis = {"category": "toy", "hard_surface_or_organic": "hard_surface", "symmetry": "mirror_x",
                "thin_parts": [], "materials": ["plastic"], "issues_seen": [], "plan": "tighten budget"}
    chat, calls = _fake([
        _reply(json.dumps(analysis), ("set_triangle_budget", {"tris": 3000, "rationale": "r"}),
               ("set_material", {"roughness": 0.4, "metallic": 0.0, "rationale": "r"}),
               ("apply_symmetry", {"kind": "rotational_y", "rationale": "r"})),
        _reply("done", ("accept", {"rationale": "r"})),
    ])
    monkeypatch.setattr(loop, "chat", chat)
    m = run_stage(store, job, paths, "reconstruct", "agent", {"refine": {"instruction": "make it shiny"}})
    report = json.loads((paths.root / "agent" / "report.json").read_text())
    assert report["analysis"]["category"] == "toy" and m["agent_category"] == "toy"
    assert "make it shiny" in calls[0][0][1]["content"] and len(calls[0][1]["images"]) >= 5
    assert [c["name"] for c in report["tool_calls"]][:3] == ["set_triangle_budget", "set_material", "apply_symmetry"]
    assert report["rounds"][0]["metrics"]["iou_mean"] == pytest.approx(iou0)
    # never ships a regression: the kept IoU is the best measured one
    ious = [r["metrics"].get("iou_mean") or 0 for r in report["rounds"]]
    assert report["best_iou"] == pytest.approx(max(ious))
    assert store.get(job.id).metrics["iou_mean"] == pytest.approx(max(ious))
    assert m["agent_cost_usd"] > 0 and store.get(job.id).artifacts["agent_report"] == "agent/report.json"


def test_agent_api_error_keeps_result(setup, monkeypatch) -> None:  # noqa: F811
    store, job, paths = setup
    _reconstruct(store, job, paths)
    before = (paths.root / "export" / "model.glb").read_bytes()
    monkeypatch.setattr(loop, "chat", _fake([RuntimeError("HTTP 500")])[0])
    m = run_stage(store, job, paths, "reconstruct", "agent", {})
    assert "HTTP 500" in m["agent_error"] and m["agent_rounds"] == 0
    assert (paths.root / "export" / "model.glb").read_bytes() == before
    assert run_stage(store, job, paths, "reconstruct", "agent", {"agent": False}) == {"skipped": True}
