"""HTTP routes of the v1 API (docs/PLAN.md). Thin: validate, update the job, enqueue an action."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from codemesh import __version__, config
from codemesh.jobs.models import Job
from codemesh.jobs.paths import JobPaths, job_paths
from codemesh.jobs.runner import JobBusyError, Runner
from codemesh.jobs.store import JobStore
from codemesh.pipeline.ingest import HEIF_SUPPORTED, PRESET_FRAMES

from .schemas import Capabilities, FrameInfo, Health, ReconstructRequest, RefineRequest, SelectRequest
from .uploads import MAX_IMAGES, UploadError, save_uploads

router = APIRouter(prefix="/api")
_MULTIPART_OVERHEAD = 1024 * 1024


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _runner(request: Request) -> Runner:
    return request.app.state.runner


def _paths(request: Request, job_id: str) -> JobPaths:
    try:
        return job_paths(request.app.state.jobs_dir, job_id)
    except ValueError:
        raise HTTPException(404, "job not found") from None


def _job(request: Request, job_id: str) -> Job:
    _paths(request, job_id)  # validates the id format
    job = _store(request).get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _frames(request: Request, job_id: str) -> list[dict[str, Any]]:
    path = _paths(request, job_id).frames / "frames.json"
    if not path.is_file():
        raise HTTPException(409, "frames are not ready yet")
    return json.loads(path.read_text(encoding="utf-8"))["frames"]


def _submit(request: Request, job_id: str, action: str, settings: dict[str, Any]) -> Job:
    try:
        _runner(request).submit(job_id, action, settings)
    except JobBusyError:
        raise HTTPException(409, "job is busy; wait for the current step to finish") from None
    return _job(request, job_id)


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(ok=True, version=__version__)


@router.get("/capabilities", response_model=Capabilities)
def capabilities(request: Request) -> Capabilities:
    # shortcut: engine details are 'unknown' until the geometry lane registers a real engine probe.
    return Capabilities(
        geometry_engine=config.GEOMETRY_ENGINE,
        device="unknown",
        precision="unknown",
        blender=Path(config.BLENDER_EXE).is_file(),
        meta_api=bool(config.META_API_KEY),
        engines={"torch": find_spec("torch") is not None},
        heic=HEIF_SUPPORTED,
        max_upload_bytes=request.app.state.max_upload_bytes,
    )


@router.post("/jobs", response_model=Job, status_code=201)
async def create_job(request: Request) -> Job:
    """multipart: files[] (one video or 2-300 images), mode, optional preset (default standard)."""
    limit: int = request.app.state.max_upload_bytes
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit + _MULTIPART_OVERHEAD:
        raise HTTPException(413, f"upload exceeds the {limit} byte limit")
    # shortcut: Starlette spools file parts to a temp file before we copy them; stream the
    # multipart body straight into the job dir if 4 GB uploads make the double write hurt.
    form = await request.form(max_files=MAX_IMAGES + 1, max_fields=16)
    try:
        mode = str(form.get("mode") or "walkaround")
        preset = str(form.get("preset") or "standard")
        if mode not in ("walkaround", "turntable"):
            raise HTTPException(422, "mode must be 'walkaround' or 'turntable'")
        if preset not in PRESET_FRAMES:
            raise HTTPException(422, f"preset must be one of {sorted(PRESET_FRAMES)}")
        files = [f for key in ("files[]", "files") for f in form.getlist(key) if isinstance(f, UploadFile)]
        job = Job(mode=mode, settings={"preset": preset})
        paths = job_paths(request.app.state.jobs_dir, job.id)
        paths.create()
        try:
            job.input = await run_in_threadpool(save_uploads, files, paths.input, limit)
        except UploadError as exc:
            paths.remove()
            raise HTTPException(exc.status_code, exc.message) from None
        except BaseException:
            paths.remove()
            raise
    finally:
        await form.close()
    _store(request).create(job)
    return _submit(request, job.id, "ingest", {})


@router.get("/jobs", response_model=list[Job])
def list_jobs(request: Request) -> list[Job]:
    return _store(request).list()


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(request: Request, job_id: str) -> Job:
    return _job(request, job_id)


@router.get("/jobs/{job_id}/frames", response_model=list[FrameInfo])
def list_frames(request: Request, job_id: str) -> list[FrameInfo]:
    _job(request, job_id)
    base = f"/api/jobs/{job_id}/artifacts/frames/"
    return [
        FrameInfo(index=e["index"], url=base + e["file"], thumb_url=base + e["thumb"],
                  sharpness=e["sharpness"], width=e["width"], height=e["height"])
        for e in _frames(request, job_id)
    ]


@router.post("/jobs/{job_id}/select", response_model=Job)
def select(request: Request, job_id: str, body: SelectRequest) -> Job:
    _job(request, job_id)
    count = len(_frames(request, job_id))
    if body.frame_index >= count:
        raise HTTPException(422, f"frame_index must be < {count}")
    return _submit(request, job_id, "segment", {"selection": body.model_dump()})


@router.post("/jobs/{job_id}/reconstruct", response_model=Job)
def reconstruct(request: Request, job_id: str, body: ReconstructRequest) -> Job:
    _job(request, job_id)
    _frames(request, job_id)
    return _submit(request, job_id, "reconstruct", {"reconstruct": body.model_dump()})


@router.post("/jobs/{job_id}/refine", response_model=Job)
def refine(request: Request, job_id: str, body: RefineRequest) -> Job:
    if "model_glb" not in _job(request, job_id).artifacts:
        raise HTTPException(409, "reconstruct the model before refining it")
    return _submit(request, job_id, "refine", {"refine": body.model_dump()})


@router.get("/jobs/{job_id}/artifacts/{rel_path:path}")
def artifact(request: Request, job_id: str, rel_path: str) -> FileResponse:
    _job(request, job_id)
    path = _paths(request, job_id).resolve_artifact(rel_path)
    if path is None:
        raise HTTPException(404, "artifact not found")
    return FileResponse(path)


@router.delete("/jobs/{job_id}")
def delete_job(request: Request, job_id: str) -> dict[str, bool]:
    _job(request, job_id)
    if _runner(request).is_busy(job_id):
        raise HTTPException(409, "job is busy; wait for the current step to finish")
    _store(request).delete(job_id)
    _paths(request, job_id).remove()
    return {"ok": True}
