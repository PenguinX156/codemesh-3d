"""FastAPI app factory. Run: uv run --project backend uvicorn codemesh.main:app --port 8000"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codemesh import __version__, config
from codemesh.api.routes import router
from codemesh.api.uploads import DEFAULT_MAX_UPLOAD_BYTES
from codemesh.jobs.runner import Runner
from codemesh.jobs.store import JobStore
from codemesh.pipeline import ACTIONS


def create_app(jobs_dir: Path | None = None, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> FastAPI:
    """Build the app. Store and worker start in the lifespan (i.e. when the server starts)."""
    jobs_dir = Path(jobs_dir or config.JOBS_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        store = JobStore(jobs_dir / "jobs.db")
        store.fail_interrupted()
        runner = Runner(store, jobs_dir, ACTIONS)
        runner.start()
        app.state.store, app.state.runner = store, runner
        try:
            yield
        finally:
            runner.stop(timeout=10)
            store.close()

    app = FastAPI(title="CodeMesh 3D", version=__version__, lifespan=lifespan)
    app.state.jobs_dir = jobs_dir
    app.state.max_upload_bytes = max_upload_bytes
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
