"""Job store, runner and stage cache tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from codemesh.jobs.cache import cached_metrics, stage_key, write_manifest
from codemesh.jobs.context import Action, Stage
from codemesh.jobs.models import Job
from codemesh.jobs.paths import JobPaths, job_paths
from codemesh.jobs.runner import JobBusyError, Runner
from codemesh.jobs.store import JobStore


@pytest.fixture
def store(tmp_path: Path):
    s = JobStore(tmp_path / "jobs.db")
    yield s
    s.close()


def test_store_crud_and_order(store: JobStore) -> None:
    old = store.create(Job(created_at=1.0))
    new = store.create(Job(created_at=2.0, mode="turntable"))
    assert [j.id for j in store.list()] == [new.id, old.id]
    store.update(old.id, lambda j: setattr(j, "message", "hi"))
    assert store.get(old.id).message == "hi"
    assert store.delete(old.id) and store.get(old.id) is None
    assert store.update("missing", lambda j: None) is None


def test_fail_interrupted(store: JobStore) -> None:
    running = Job(status="reconstructing")
    running.stage_record("geometry").status = "running"
    store.create(running)
    idle = store.create(Job(status="ready_to_select"))
    assert store.fail_interrupted() == 1
    job = store.get(running.id)
    assert job.status == "failed" and "restart" in job.error and job.stages[0].status == "failed"
    assert store.get(idle.id).status == "ready_to_select"


def test_job_paths_reject_bad_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        job_paths(tmp_path, "../etc")
    paths = JobPaths(tmp_path / "j")
    paths.create()
    (paths.frames / "a.jpg").write_bytes(b"x")
    (tmp_path / "secret.txt").write_text("s")
    assert paths.resolve_artifact("frames/a.jpg") == (paths.frames / "a.jpg").resolve()
    for bad in ("../secret.txt", "frames/../../secret.txt", "/etc/passwd", "C:/x", "frames\\a.jpg", "", "frames"):
        assert paths.resolve_artifact(bad) is None, bad


def _runner(tmp_path: Path, store: JobStore, stages: list[Stage]) -> Runner:
    runner = Runner(store, tmp_path, {"act": Action(running="segmenting", done="ready_to_reconstruct", stages=stages)})
    runner.start()
    return runner


def _new_job(tmp_path: Path, store: JobStore) -> Job:
    job = store.create(Job(status="ready_to_select"))
    job_paths(tmp_path, job.id).create()
    return job


def test_runner_records_stages_and_progress(tmp_path: Path, store: JobStore) -> None:
    seen = []

    def a(ctx):
        ctx.progress(0.5, "half")
        seen.append(ctx.settings["x"])
        return {"n": 1}

    runner = _runner(tmp_path, store, [Stage("a", a), Stage("b", lambda ctx: {"cached": True})])
    job = _new_job(tmp_path, store)
    runner.submit(job.id, "act", {"x": 7})
    assert runner.wait_idle()
    runner.stop()
    job = store.get(job.id)
    assert seen == [7] and job.settings["x"] == 7
    assert job.status == "ready_to_reconstruct" and job.progress == 1.0
    assert [(s.name, s.status, s.cached) for s in job.stages] == [("a", "done", False), ("b", "done", True)]
    assert job.stages[0].metrics == {"n": 1} and job.stages[0].seconds is not None


def test_runner_failure_writes_traceback(tmp_path: Path, store: JobStore) -> None:
    def boom(ctx):
        raise RuntimeError("kaputt")

    runner = _runner(tmp_path, store, [Stage("boom", boom), Stage("never", lambda ctx: None)])
    job = _new_job(tmp_path, store)
    runner.submit(job.id, "act")
    assert runner.wait_idle()
    runner.stop()
    job = store.get(job.id)
    assert job.status == "failed" and job.error == "boom failed: kaputt"
    assert [s.name for s in job.stages] == ["boom"] and job.stages[0].status == "failed"
    log = (job_paths(tmp_path, job.id).logs / "boom.log").read_text()
    assert "Traceback" in log and "RuntimeError: kaputt" in log


def test_runner_rejects_busy_job(tmp_path: Path, store: JobStore) -> None:
    runner = _runner(tmp_path, store, [Stage("slow", lambda ctx: time.sleep(0.3))])
    job = _new_job(tmp_path, store)
    runner.submit(job.id, "act")
    with pytest.raises(JobBusyError):
        runner.submit(job.id, "act")
    assert runner.is_busy(job.id)
    assert runner.wait_idle()
    runner.stop()
    assert not runner.is_busy(job.id)


def test_stage_cache(tmp_path: Path) -> None:
    src = tmp_path / "in.bin"
    src.write_bytes(b"abc")
    out = tmp_path / "out"
    out.mkdir()
    key = stage_key("s", [src], {"k": 1})
    assert key != stage_key("s", [src], {"k": 2}) and key != stage_key("t", [src], {"k": 1})
    assert cached_metrics(out, "s", key) is None
    (out / "r.txt").write_text("r")
    write_manifest(out, "s", key, [out / "r.txt"], {"m": 3})
    assert cached_metrics(out, "s", key) == {"m": 3}
    assert cached_metrics(out, "s", "other") is None
    (out / "r.txt").unlink()  # missing output -> miss
    assert cached_metrics(out, "s", key) is None
