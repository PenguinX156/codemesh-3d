"""Background runner: one worker thread, one heavy job at a time, FIFO queue.

Records per-stage status/timings/metrics into the Job. Any exception fails the job with a
readable message; the full traceback goes to jobs/<id>/logs/<stage>.log.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from pathlib import Path

from .context import Action, StageContext
from .paths import job_paths
from .store import JobStore

log = logging.getLogger(__name__)


class JobBusyError(Exception):
    """The job already has an action queued or running."""


class Runner:
    def __init__(self, store: JobStore, jobs_dir: Path, actions: dict[str, Action]) -> None:
        self.store = store
        self.jobs_dir = jobs_dir
        self.actions = actions
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._busy: set[str] = set()  # job ids queued or running
        self._busy_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="codemesh-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Finish the current job, then stop (queued jobs stay unprocessed)."""
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout)
            self._thread = None

    def is_busy(self, job_id: str) -> bool:
        with self._busy_lock:
            return job_id in self._busy

    def active(self) -> bool:
        """True while any job is queued or running."""
        with self._busy_lock:
            return bool(self._busy)

    def submit(self, job_id: str, action: str, settings: dict | None = None) -> None:
        """Queue `action` for the job, first merging `settings` into Job.settings.

        Raises JobBusyError if the job is already queued/running (settings stay untouched).
        """
        spec = self.actions[action]
        with self._busy_lock:
            if job_id in self._busy:
                raise JobBusyError(job_id)
            self._busy.add(job_id)

        def mark_queued(job):
            job.status = spec.running
            job.stage = None
            job.progress = 0.0
            job.message = "Queued"
            job.error = None
            job.settings.update(settings or {})

        self.store.update(job_id, mark_queued)
        self._queue.put((job_id, action))

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no job is queued or running (for tests and scripts)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._busy_lock:
                if not self._busy:
                    return True
            time.sleep(0.02)
        return False

    def _loop(self) -> None:
        while (item := self._queue.get()) is not None:
            job_id, action = item
            try:
                self._run(job_id, self.actions[action])
            except Exception:  # never let the worker die
                log.exception("runner crashed on job %s", job_id)
            finally:
                with self._busy_lock:
                    self._busy.discard(job_id)

    def _run(self, job_id: str, action: Action) -> None:
        job = self.store.get(job_id)
        if job is None:  # deleted while queued
            return
        paths = job_paths(self.jobs_dir, job_id)
        count = len(action.stages)
        for index, stage in enumerate(action.stages):

            def start(j, name=stage.name, index=index):
                j.stage = name
                j.progress = index / count
                j.message = f"Running {name}"
                rec = j.stage_record(name)
                rec.status, rec.seconds, rec.cached, rec.metrics = "running", None, False, {}

            self.store.update(job_id, start)
            ctx = StageContext(self.store, job_id, paths, job.settings, stage.name, index, count)
            t0 = time.perf_counter()
            try:
                metrics = dict(stage.run(ctx) or {})
            except Exception as exc:
                self._fail(job_id, stage.name, exc, time.perf_counter() - t0)
                return
            seconds = round(time.perf_counter() - t0, 3)

            def finish(j, name=stage.name, metrics=metrics, seconds=seconds):
                rec = j.stage_record(name)
                rec.cached = bool(metrics.pop("cached", False))
                rec.status, rec.seconds, rec.metrics = "done", seconds, metrics

            self.store.update(job_id, finish)

        def done(j):
            j.status, j.stage, j.progress, j.message = action.done, None, 1.0, "Done"

        self.store.update(job_id, done)

    def _fail(self, job_id: str, stage: str, exc: Exception, seconds: float) -> None:
        paths = job_paths(self.jobs_dir, job_id)
        try:
            paths.logs.mkdir(parents=True, exist_ok=True)
            with open(paths.logs / f"{stage}.log", "a", encoding="utf-8") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} stage {stage} failed\n")
                f.write(traceback.format_exc())
        except OSError:
            log.exception("could not write log for job %s", job_id)
        message = f"{stage} failed: {exc}" if str(exc) else f"{stage} failed: {type(exc).__name__}"

        def apply(j):
            rec = j.stage_record(stage)
            rec.status, rec.seconds = "failed", round(seconds, 3)
            j.status, j.error, j.message = "failed", message, message

        self.store.update(job_id, apply)
