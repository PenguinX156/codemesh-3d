"""Pipeline registry: which stages each user action runs, and the job status around it.

Phase 1 hook point: implement a stage module (e.g. pipeline/segment.py with `run(ctx)`) and
replace its `_not_implemented(...)` entry below with `Stage("segment", segment.run)`.
"""

from __future__ import annotations

from codemesh.jobs.context import Action, Stage, StageContext, StageFn

from . import ingest


def _not_implemented(name: str) -> StageFn:
    def run(ctx: StageContext) -> None:
        raise NotImplementedError(f"stage '{name}' is not implemented yet")

    return run


def _stub(name: str) -> Stage:
    return Stage(name, _not_implemented(name))


ACTIONS: dict[str, Action] = {
    # POST /jobs -> frames ready for the user to pick the object.
    "ingest": Action(running="created", done="ready_to_select", stages=[Stage("ingest", ingest.run)]),
    # POST /jobs/{id}/select -> masks for the selected object.
    "segment": Action(running="segmenting", done="ready_to_reconstruct", stages=[_stub("segment")]),
    # POST /jobs/{id}/reconstruct -> model.glb + metrics.
    "reconstruct": Action(
        running="reconstructing",
        done="done",
        stages=[_stub(n) for n in ("geometry", "fuse", "mesh", "texture", "export", "evaluate")],
    ),
    # POST /jobs/{id}/refine (Phase 3).
    "refine": Action(running="refining", done="done", stages=[_stub("refine")]),
}
