"""Job record: the single object the API returns and the store persists (see docs/PLAN.md)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "created",
    "ready_to_select",
    "segmenting",
    "ready_to_reconstruct",
    "reconstructing",
    "refining",
    "done",
    "failed",
]
StageStatus = Literal["pending", "running", "done", "skipped", "failed"]
Mode = Literal["walkaround", "turntable"]

# Statuses in which a worker is (or was, before a crash) processing the job.
ACTIVE_STATUSES: frozenset[str] = frozenset({"created", "segmenting", "reconstructing", "refining"})


class StageRecord(BaseModel):
    name: str
    status: StageStatus = "pending"
    seconds: float | None = None
    cached: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class JobInput(BaseModel):
    kind: Literal["video", "images"]
    files: list[str]  # file names inside jobs/<id>/input/
    bytes: int


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = Field(default_factory=time.time)
    mode: Mode = "walkaround"
    status: JobStatus = "created"
    stage: str | None = None
    progress: float = 0.0
    message: str = ""
    stages: list[StageRecord] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)  # key -> path relative to the job dir
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    input: JobInput | None = None
    # User choices: preset, selection (select), reconstruct settings, refine instruction.
    settings: dict[str, Any] = Field(default_factory=dict)

    def stage_record(self, name: str) -> StageRecord:
        """Return the record for `name`, creating it if this stage never ran."""
        for rec in self.stages:
            if rec.name == name:
                return rec
        rec = StageRecord(name=name)
        self.stages.append(rec)
        return rec
