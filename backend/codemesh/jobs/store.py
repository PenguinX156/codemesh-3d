"""SQLite job store (stdlib sqlite3, WAL). One row per job; the full Job is a JSON column.

Thread-safe: one connection guarded by a lock, shared by API threads and the worker.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from .models import ACTIVE_STATUSES, Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL
)
"""


class JobStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _write(self, job: Job) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO jobs (id, created_at, status, data) VALUES (?, ?, ?, ?)",
            (job.id, job.created_at, job.status, job.model_dump_json()),
        )

    def _read(self, job_id: str) -> Job | None:
        row = self._db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row[0]) if row else None

    def create(self, job: Job) -> Job:
        with self._lock:
            self._write(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._read(job_id)

    def list(self) -> list[Job]:
        """All jobs, newest first."""
        with self._lock:
            rows = self._db.execute("SELECT data FROM jobs ORDER BY created_at DESC").fetchall()
        return [Job.model_validate_json(r[0]) for r in rows]

    def update(self, job_id: str, fn: Callable[[Job], None]) -> Job | None:
        """Atomically read the job, let `fn` mutate it, and save it. Returns None if missing."""
        with self._lock:
            job = self._read(job_id)
            if job is None:
                return None
            fn(job)
            self._write(job)
            return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,)).rowcount > 0

    def fail_interrupted(self) -> int:
        """Mark jobs left in an active status by a crash/restart as failed. Returns the count."""
        with self._lock:
            marks = ",".join("?" * len(ACTIVE_STATUSES))
            rows = self._db.execute(
                f"SELECT data FROM jobs WHERE status IN ({marks})", tuple(ACTIVE_STATUSES)
            ).fetchall()
            for (data,) in rows:
                job = Job.model_validate_json(data)
                job.status = "failed"
                job.error = job.message = "Interrupted by a server restart; retry the last step."
                for rec in job.stages:
                    if rec.status == "running":
                        rec.status = "failed"
                self._write(job)
            return len(rows)
