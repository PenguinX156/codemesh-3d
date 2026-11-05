"""Agent stage: Muse Spark looks at the capture and the measured first reconstruction, decides how
to model the object through `tools.TOOLS`, conventional geometry code executes the decisions, and
every round is measured by the evaluate stage. The best version by silhouette IoU is kept.

Calls: 1 analysis + 1 per further refinement round (MAX_CALLS caps it). Any Muse/API error keeps
the pre-agent result and is reported as metrics.agent_error; the stage never fails the job.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from codemesh import config
from codemesh.jobs.context import StageContext
from codemesh.meta import muse

from . import ops, tools

log = logging.getLogger(__name__)

MAX_ROUNDS = 2
# Keep rule: best silhouette IoU, but a clean primitive wins while within this margin of the best
# (clean topology matters more for game/Blender use than the last few IoU points).
PRIMITIVE_MARGIN = 0.03
MAX_CALLS = 4
VERSIONED = ("mesh", "texture", "export", "evaluate")  # dirs snapshotted per round
chat = muse.chat  # tests monkeypatch this

SYSTEM = """You are the 3D modeling lead of CodeMesh, which turns a phone video of one object into a \
game-ready textured mesh. A segmentation model has isolated the object (masks), a geometry model \
estimated cameras + depth, and conventional code fused and meshed it. You see reference crops of the \
object, its mask cutout, and evaluation renders of the current mesh against the masks (compare images) \
with metrics (silhouette IoU, boundary F, chamfer % of bbox diagonal).
Decide how the object should be modeled and call tools to fix it. Use what you know about the object \
category. Prefer a CLEAN modeled base mesh over the noisy reconstruction whenever the object is a simple hard-surface shape: a mug/cup = build_lathe(hollow=true) + add_handle; a bottle/vase/kettle body = build_lathe; a cabinet/box/book = build_rounded_box. Organic or irregular objects (plants, food, plush toys) keep the reconstruction (use_reconstruction). Shiny/metal objects: set_material with high metallic, low roughness.
In your text answer return ONLY a JSON object: {"category": str, "hard_surface_or_organic": \
"hard_surface"|"organic", "symmetry": str, "thin_parts": [str], "materials": [str], "issues_seen": [str], \
"plan": str}, and in the same reply call the tools that execute the plan. In later rounds answer with one sentence of text plus tool calls. Call accept() when done."""


def _iou(job_dir: Path) -> float | None:
    p = job_dir / "evaluate" / "metrics.json"
    return json.loads(p.read_text()).get("iou_mean") if p.exists() else None


def _summary(job_dir: Path) -> dict[str, Any]:
    e = json.loads((job_dir / "evaluate" / "metrics.json").read_text())
    m = json.loads((job_dir / "mesh" / "metrics.json").read_text()) if (job_dir / "mesh" / "metrics.json").exists() else {}
    keys = ("iou_mean", "boundary_f_mean", "psnr_mean", "ssim_mean", "chamfer_pct_mean")
    return {**{k: e.get(k) for k in keys}, "tris_high": m.get("tris_high"), "tris_low": m.get("tris_low")}


def _snapshot(job_dir: Path, version: int) -> None:
    dst = job_dir / "agent" / f"v{version}"
    shutil.rmtree(dst, ignore_errors=True)
    for d in VERSIONED:
        if (job_dir / d).exists():
            shutil.copytree(job_dir / d, dst / d)


def _restore(job_dir: Path, version: int) -> None:
    src = job_dir / "agent" / f"v{version}"
    for d in VERSIONED:
        if (src / d).exists():
            shutil.rmtree(job_dir / d, ignore_errors=True)
            shutil.copytree(src / d, job_dir / d)


def reference_images(job_dir: Path, n: int = 3) -> list[np.ndarray]:
    """Full best frame (context), crops around the mask of the best + spread frames, mask cutout."""
    masks = sorted((job_dir / "masks").glob("*.png"))
    areas = {p.stem: int((np.asarray(Image.open(p).convert("L")) > 127).sum()) for p in masks}
    frames = {p.stem: p for p in (job_dir / "frames").glob("*") if p.suffix.lower() in (".jpg", ".png")}
    stems = [s for s, a in areas.items() if a > 0 and s in frames]
    if not stems:
        return []
    best = max(stems, key=areas.get)
    spread = [stems[int(i)] for i in np.linspace(0, len(stems) - 1, n + 1)[:-1]]
    picks = [best] + [s for s in spread if s != best][: n - 1]
    out: list[np.ndarray] = []
    for k, s in enumerate(picks):
        rgb = np.asarray(Image.open(frames[s]).convert("RGB"))
        m = np.asarray(Image.open(job_dir / "masks" / f"{s}.png").convert("L").resize(rgb.shape[1::-1])) > 127
        ys, xs = np.nonzero(m)
        pad = int(0.6 * max(np.ptp(ys), np.ptp(xs)) + 16)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, rgb.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, rgb.shape[1])
        if k == 0:
            out.append(rgb)
        out.append(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        if k == 0:
            out.append(np.ascontiguousarray((rgb * m[..., None])[y0:y1, x0:x1]))
    return out


def compare_images(job_dir: Path, n: int = 3) -> list[Path]:
    paths = sorted((job_dir / "evaluate").glob("compare_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    return [paths[int(i)] for i in np.linspace(0, len(paths) - 1, n)] if len(paths) > n else paths


def parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {"raw": text}
    except json.JSONDecodeError:
        return {"raw": text}


def rerun(job_dir: Path, base: dict[str, Any], plan: tools.Plan, first: str, progress) -> None:
    """Re-run `first` and every later stage with the plan applied (symmetry right after mesh)."""
    from codemesh.pipeline import evaluate, export, mesh, primitives, texture

    fns = {"mesh": mesh.run, "texture": texture.run, "export": export.run, "evaluate": evaluate.run}
    settings = {**base, **plan.settings()}
    for name in tools.STAGES[tools.STAGES.index(first):]:
        progress(0.5, f"agent: re-running {name}")
        fns[name](job_dir, settings, None)
        if name == "mesh" and plan.symmetry != "none":
            budget = plan.triangle_budget or (base.get("triangle_budget") if base.get("triangle_budget") != "auto" else None)
            ops.apply_symmetry(job_dir, plan.symmetry, budget)
        if name == "mesh" and plan.primitive:  # low = clean primitive, high = reconstruction (normal-map detail)
            fit = primitives.build(job_dir, **plan.primitive)
            (job_dir / "mesh" / "primitive.json").write_text(json.dumps(fit, indent=1))
            mp = job_dir / "mesh" / "metrics.json"
            if mp.exists():
                mp.write_text(json.dumps({**json.loads(mp.read_text()), "tris_low": fit["triangles"], "primitive": fit}, indent=1))


def _ask(messages: list[dict], images: list, usage: list[muse.Usage]) -> muse.MuseReply:
    if len(usage) >= MAX_CALLS:
        raise RuntimeError("Muse call budget exhausted")
    reply = chat(messages, images=images, tools=tools.TOOLS, reasoning=config.MUSE_REASONING,
                 max_tokens=8000, max_side=config.MUSE_IMAGE_MAX_SIDE)
    usage.append(reply.usage)
    return reply


def run(ctx: StageContext) -> dict[str, Any]:
    from codemesh.pipeline import _publish, _record, reconstruct_settings

    base = reconstruct_settings(ctx.settings)
    if base.get("agent") is False:
        return {"skipped": True}
    root = ctx.paths.root
    if _iou(root) is None:
        return {"skipped": True, "agent_error": "no evaluated reconstruction to refine"}
    instruction = base.get("instruction") or (base.get("refine") or {}).get("instruction")
    report: dict[str, Any] = {"model": config.MUSE_MODEL, "instruction": instruction, "analysis": None,
                              "tool_calls": [], "rounds": [], "usage": []}
    usage: list[muse.Usage] = []
    _snapshot(root, 0)
    report["rounds"].append({"round": 0, "metrics": _summary(root), "plan": None})
    best, best_iou = 0, _iou(root)
    best_score = best_iou  # IoU + PRIMITIVE_MARGIN for primitive rounds
    error = None
    try:
        ctx.progress(0.05, "agent: Muse is analysing the capture")
        text = ("Metrics of the first reconstruction: " + json.dumps(_summary(root))
                + (f"\nUser instruction: {instruction}" if instruction else "")
                + "\nImages: full reference frame, object crop, mask cutout, two more object crops from other "
                  "viewpoints, then evaluation compare images (render vs mask) of the current mesh.")
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        reply = _ask(messages, reference_images(root) + compare_images(root), usage)
        report["analysis"] = parse_json(reply.text)
        if not reply.tool_calls:  # Muse often answers the JSON first and acts only when asked
            messages += [reply.message, {"role": "user", "content": "Now execute your plan: call the tools "
                                         "(several in one reply). Do not repeat the JSON."}]
            reply = _ask(messages, [], usage)
        plan = tools.Plan()
        for rnd in range(1, MAX_ROUNDS + 1):
            calls = reply.tool_calls
            results, first, accepted = [], None, False
            for c in calls:
                try:
                    result, stage = tools.apply(plan, c.name, c.arguments)
                except (ValueError, TypeError) as e:
                    result, stage = f"error: {e}", None
                accepted |= c.name == "accept"
                if stage and (first is None or tools.STAGES.index(stage) < tools.STAGES.index(first)):
                    first = stage
                results.append({"role": "tool", "tool_call_id": c.id, "content": result})
                report["tool_calls"].append({"round": rnd, "name": c.name, "arguments": c.arguments, "result": result})
            if first is None:  # accept() or nothing to change
                break
            ctx.progress(0.2 + 0.35 * (rnd - 1), f"agent round {rnd}: re-running from {first}")
            try:
                rerun(root, base, plan, first, ctx.progress)
                metrics = _summary(root)
            except Exception as e:  # a bad decision (e.g. a degenerate symmetry) must not end the loop
                log.exception("agent round %d failed", rnd)
                metrics = {"error": f"{type(e).__name__}: {e}"}
                _restore(root, best)
            iou = metrics.get("iou_mean")
            _snapshot(root, rnd)
            report["rounds"].append({"round": rnd, "metrics": metrics, "plan": {**plan.settings(), "symmetry": plan.symmetry,
                                                                            "primitive": dict(plan.primitive or {}) or None}})
            score = None if iou is None else iou + (PRIMITIVE_MARGIN if plan.primitive else 0.0)
            improved = score is not None and score > best_score
            if improved:
                best, best_iou, best_score = rnd, iou, score
            if accepted or not improved or rnd == MAX_ROUNDS:
                break
            messages += [reply.message, *results, {"role": "user", "content":
                         f"Round {rnd} result: {json.dumps(metrics)} (best IoU so far {best_iou:.3f}). "
                         "New compare images attached. Call tools to improve further, or accept()."}]
            reply = _ask(messages, compare_images(root), usage)
            report.setdefault("comments", []).append(reply.text)
    except Exception as e:  # Muse/API failure: keep the best measured version
        log.exception("agent stage failed")
        error = f"{type(e).__name__}: {e}"

    _restore(root, best)
    total = sum(usage, muse.Usage())
    report.update(best_round=best, best_iou=best_iou, error=error,
                  keep_rule=f"best silhouette IoU; a clean primitive is preferred within {PRIMITIVE_MARGIN} IoU of the best",
                  usage=[u.__dict__ for u in usage], cost_usd=round(total.cost_usd, 5),
                  tokens={"prompt": total.prompt_tokens, "completion": total.completion_tokens,
                          "reasoning": total.reasoning_tokens})
    (root / "agent" / "report.json").write_text(json.dumps(report, indent=1, default=str))
    ctx.set_artifact("agent_report", "agent/report.json")
    for name in ("export", "evaluate"):
        _publish(ctx, name)
    _record(ctx, "evaluate", json.loads((root / "evaluate" / "metrics.json").read_text()))
    if (root / "mesh" / "metrics.json").exists():
        _record(ctx, "mesh", json.loads((root / "mesh" / "metrics.json").read_text()))
    analysis = report["analysis"] or {}
    out = {"agent_rounds": len(report["rounds"]) - 1, "agent_cost_usd": report["cost_usd"],
           "agent_category": analysis.get("category"), "agent_best_round": best,
           "agent_iou_before": report["rounds"][0]["metrics"]["iou_mean"], "agent_iou_after": best_iou,
           **({"agent_error": error} if error else {})}
    ctx.store.update(ctx.job_id, lambda job: job.metrics.update(out))
    ctx.progress(1.0, f"agent: kept round {best} (IoU {best_iou:.3f})")
    return out
