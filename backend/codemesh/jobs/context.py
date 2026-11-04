"""What a pipeline stage sees (StageContext) and how stages are grouped into actions.

A stage is a plain function `run(ctx) -> metrics dict | None`. It reads its inputs from
`ctx.paths`, its options from `ctx.settings`, writes only into its own job subfolder,
reports progress with `ctx.progress(...)` and publishes files with `ctx.set_artifact(...)`.
Returning `{"cached": True, ...}` marks the stage as a cache hit (see jobs/cache.py).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import JobStatus
from .paths import JobPaths
from .store import JobStore

_PROGRESS_INTERVAL_S = 0.25  # throttle DB writes from chatty stages


class StageContext:
    def __init__(
        self,
        store: JobStore,
        job_id: str,
        paths: JobPaths,
        settings: dict[str, Any],
        stage: str,
        stage_index: int,
        stage_count: int,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.paths = paths
        self.settings = settings  # snapshot of Job.settings when the action started
        self.stage = stage
        self._index = stage_index
        self._count = stage_count
        self._last_write = 0.0

    def progress(self, fraction: float, message: str | None = None) -> None:
        """Report this stage's progress (0..1). Mapped onto the job's overall action progress."""
        now = time.monotonic()
        if fraction < 1.0 and message is None and now - self._last_write < _PROGRESS_INTERVAL_S:
            return
        self._last_write = now
        overall = (self._index + min(max(fraction, 0.0), 1.0)) / self._count

        def apply(job):
            job.progress = round(overall, 4)
            if message is not None:
                job.message = message

        self.store.update(self.job_id, apply)

    def set_artifact(self, key: str, rel_path: str) -> None:
        """Publish a file (path relative to the job dir) under `key` in Job.artifacts."""
        self.store.update(self.job_id, lambda job: job.artifacts.__setitem__(key, rel_path))


StageFn = Callable[[StageContext], dict[str, Any] | None]


@dataclass(frozen=True)
class Stage:
    name: str
    run: StageFn


@dataclass(frozen=True)
class Action:
    """A user-triggered step (ingest, segment, reconstruct, refine) = an ordered list of stages.

    `running` is the job status while queued/running, `done` the status after success.
    """

    running: JobStatus
    done: JobStatus
    stages: list[Stage] = field(default_factory=list)
