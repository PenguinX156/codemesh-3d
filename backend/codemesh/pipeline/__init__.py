"""Pipeline registry: which stages each user action runs, and the job status around it.

segment/geometry are `run(ctx)` stages; fuse..evaluate are `run(job_dir, settings, progress)`
functions wrapped by `_job_dir_stage`, which flattens Job.settings["reconstruct"] over the job
settings, publishes the stage's files as artifacts and copies headline numbers into Job.metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from codemesh.jobs.context import Action, Stage, StageContext

from . import evaluate, export, fuse, geometry, ingest, mesh, quality, segment, texture

JobDirFn = Callable[[Path, dict[str, Any], Callable[[float, str], None] | None], dict[str, Any]]

# Stage metrics copied into Job.metrics (prefixed with the stage name) for the UI.
HEADLINE = {
    "geometry": ("views", "engine", "device", "infer_seconds"),
    "fuse": ("points",),
    "mesh": ("tris_high", "tris_low", "chamfer_pct"),
    "texture": ("texture_size",),
    "export": ("triangles", "glb_mb"),
}


def _agent(ctx: StageContext) -> dict[str, Any]:
    from codemesh.agent import loop  # lazy: the agent imports this package

    return loop.run(ctx)


def reconstruct_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Job settings with the reconstruct request (preset, triangle_budget, ...) on top."""
    flat = {k: v for k, v in settings.items() if k not in ("reconstruct", "selection")}
    return {**flat, **{k: v for k, v in (settings.get("reconstruct") or {}).items() if v is not None}}


def _record(ctx: StageContext, name: str, metrics: dict[str, Any]) -> None:
    """Copy headline metrics + stage seconds into Job.metrics (evaluate: its whole summary)."""
    if name == "evaluate":
        picked = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    else:
        picked = {f"{name}_{k}" if k not in ("tris_high", "tris_low") else k: metrics[k]
                  for k in HEADLINE.get(name, ()) if k in metrics}

    def apply(job):
        job.metrics.update(picked)
        if "seconds" in metrics:
            job.metrics.setdefault("stage_seconds", {})[name] = metrics["seconds"]

    ctx.store.update(ctx.job_id, apply)


def _publish(ctx: StageContext, name: str) -> None:
    root = ctx.paths.root
    if name == "export":
        ctx.set_artifact("model_glb", "export/model.glb")
        ctx.set_artifact("preview", "export/preview.png")
    elif name == "evaluate":
        for p in sorted((root / "evaluate").glob("compare_*.png")):
            ctx.set_artifact(p.stem, f"evaluate/{p.name}")
        ctx.set_artifact("evaluate_metrics", "evaluate/metrics.json")


def _job_dir_stage(name: str, fn: JobDirFn) -> Stage:
    def run(ctx: StageContext) -> dict[str, Any]:
        metrics = fn(ctx.paths.root, reconstruct_settings(ctx.settings), ctx.progress)
        _publish(ctx, name)
        _record(ctx, name, metrics)
        return metrics

    return Stage(name, run)


def _geometry(ctx: StageContext) -> dict[str, Any]:
    job = ctx.store.get(ctx.job_id)
    ctx.settings = reconstruct_settings(ctx.settings)
    metrics = geometry.run(ctx, job.mode if job else "walkaround")
    _record(ctx, "geometry", metrics)
    return metrics


ACTIONS: dict[str, Action] = {
    # POST /jobs -> frames ready for the user to pick the object.
    "ingest": Action(running="created", done="ready_to_select", stages=[Stage("ingest", ingest.run)]),
    # POST /jobs/{id}/select -> masks for the selected object.
    "segment": Action(running="segmenting", done="ready_to_reconstruct", stages=[Stage("segment", segment.run)]),
    # POST /jobs/{id}/reconstruct -> model.glb + metrics.
    "reconstruct": Action(
        running="reconstructing",
        done="done",
        stages=[Stage("geometry", _geometry)]
        + [_job_dir_stage(n, m.run) for n, m in
           (("fuse", fuse), ("mesh", mesh), ("texture", texture), ("export", export), ("evaluate", evaluate))]
        + [Stage("agent", _agent)]  # Muse Spark modeling loop; settings.agent = false skips it
        + [_job_dir_stage("quality", quality.run)],  # final game-ready gate on the shipped GLB
    ),
    # POST /jobs/{id}/refine -> the agent loop again, with settings.refine.instruction in the prompt.
    "refine": Action(running="refining", done="done",
                     stages=[Stage("agent", _agent), _job_dir_stage("quality", quality.run)]),
}
