"""Local GPU geometry engine: TorchEngine on the Radeon 780M, run in a persistent ROCm worker.

The ROCm PyTorch wheels live in their own venv (bench/rocm/.venv) because the backend venv has
CPU torch. This engine keeps ONE worker subprocess (bench/rocm/worker.py) alive so the long
model load and kernel warmup are paid once; requests go through npz files in a temp dir plus a
line-based JSON protocol on the worker's stdin/stdout (see worker.py).

    engine = RocmEngine()          # blocks until the model is loaded (minutes on first run)
    geo = engine.infer(frames)     # frames: list of RGB uint8 arrays
    engine.close()
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from codemesh.config import REPO_ROOT
from codemesh.engines.base import ViewGeometry

ROCM_DIR = REPO_ROOT / "bench" / "rocm"
ROCM_PYTHON = ROCM_DIR / ".venv" / "Scripts" / "python.exe"
WORKER = ROCM_DIR / "worker.py"


def rocm_available() -> bool:
    """True when the ROCm venv and worker script exist (does not check the GPU itself)."""
    return ROCM_PYTHON.exists() and WORKER.exists()


class RocmEngine:
    """GeometryEngine that forwards to a persistent ROCm worker process."""

    name = "rocm"

    def __init__(
        self,
        model: str = "mapanything",
        precision: str | None = None,
        startup_timeout: float = 900.0,
        infer_timeout: float = 1200.0,
    ) -> None:
        if not rocm_available():
            raise FileNotFoundError(f"ROCm venv or worker missing: {ROCM_PYTHON}, {WORKER}")
        self.model, self.precision = model, precision
        self.startup_timeout, self.infer_timeout = startup_timeout, infer_timeout
        self._tmp = Path(tempfile.mkdtemp(prefix="codemesh_rocm_"))
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._info: dict = {}
        self._calls = 0
        self._start()

    # -- worker lifecycle --------------------------------------------------------------------
    def _start(self) -> None:
        args = [str(ROCM_PYTHON), str(WORKER), self.model] + ([self.precision] if self.precision else [])
        log = open(self._tmp / "worker.log", "ab")  # worker stderr (model/loader noise, tracebacks)
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, cwd=str(ROCM_DIR)
        )
        self._lines = q = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, q), daemon=True).start()
        ready = self._read(self.startup_timeout)
        if ready.get("event") != "ready":
            self._kill()
            raise RuntimeError(f"ROCm worker failed to start: {ready}")
        self._info = ready["info"]

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            q.put(line)
        q.put(None)  # EOF: the worker exited

    def _read(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise TimeoutError(f"ROCm worker gave no answer within {timeout:.0f} s (killed; see {self._tmp / 'worker.log'})")
        if line is None:
            raise ConnectionError(f"ROCm worker exited (see {self._tmp / 'worker.log'})")
        return json.loads(line)

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            # The venv python.exe is a launcher that spawns the real interpreter: kill the whole tree,
            # otherwise the worker survives and keeps its GPU memory.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self._proc.pid)], capture_output=True)
            self._proc.kill()
        self._proc = None

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- GeometryEngine ----------------------------------------------------------------------
    def info(self) -> dict:
        return {**self._info, "engine": self.name, "worker_pid": self._proc.pid if self._proc else None}

    def infer(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None) -> ViewGeometry:
        """Run multi-view inference in the worker. `masks` is ignored (as in TorchEngine).

        A dead or hung worker is killed and restarted; a request that hit a dead worker is retried once.
        """
        self._calls += 1
        inp, out = self._tmp / f"in_{self._calls}.npz", self._tmp / f"out_{self._calls}.npz"
        np.savez(inp, **{f"img_{i}": im for i, im in enumerate(images)})
        try:
            for attempt in (1, 2):
                if not self._alive():
                    self._start()
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "infer", "inp": str(inp), "out": str(out)}) + "\n")  # type: ignore[union-attr]
                    self._proc.stdin.flush()  # type: ignore[union-attr]
                    resp = self._read(self.infer_timeout)
                    break
                except (ConnectionError, BrokenPipeError, OSError):
                    self._kill()
                    if attempt == 2:
                        raise
            if not resp.get("ok"):
                raise RuntimeError(f"ROCm worker error:\n{resp.get('error')}")
            with np.load(out) as z:
                return ViewGeometry(
                    intrinsics=z["intrinsics"], extrinsics=z["extrinsics"], depth=z["depth"],
                    depth_conf=z["depth_conf"], pixel_to_original=z["pixel_to_original"],
                    points=z["points"], points_conf=z["points_conf"],
                    metric_scale=bool(z["metric_scale"]), info=json.loads(str(z["info_json"])),
                )
        finally:
            inp.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    def close(self) -> None:
        if self._alive():
            try:
                self._proc.stdin.write('{"cmd": "quit"}\n')  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._kill()
        shutil.rmtree(self._tmp, ignore_errors=True)
