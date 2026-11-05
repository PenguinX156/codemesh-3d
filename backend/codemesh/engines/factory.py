"""Pick and cache the process-wide GeometryEngine from config.GEOMETRY_ENGINE.

modal -> ModalEngine (deployed app 'codemesh', class GeometryWorker), torch -> TorchEngine
MapAnything on CUDA if available else CPU, rocm -> RocmEngine, auto -> rocm when the ROCm venv exists
(local Windows), else modal when the deployed class resolves within ~10 s, else torch. Loading
weights is slow, so the engine is a singleton; warmup() loads it (and runs one dummy inference) in a
background thread so the first job does not pay for it.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from codemesh import config
from codemesh.engines.base import GeometryEngine

log = logging.getLogger(__name__)
MODAL_PROBE_TIMEOUT_S = 10.0

WARMUP_VIEWS = 16  # the draft view count, so the kernels compiled here are the ones jobs reuse
WARMUP_SIZE = (640, 480)  # (w, h)

_engine: GeometryEngine | None = None
_lock = threading.Lock()
_resolved: str | None = None
_state = "cold"  # cold | warming | ready (set by warmup())
_state_lock = threading.Lock()


def modal_available(timeout: float = MODAL_PROBE_TIMEOUT_S) -> bool:
    """True when `modal` is importable and the deployed GeometryWorker class hydrates in time."""
    try:
        import modal
    except ImportError:
        return False
    from codemesh.engines.modal_engine import APP_NAME, WORKER_CLASS

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(lambda: modal.Cls.from_name(APP_NAME, WORKER_CLASS).hydrate()).result(timeout)
        return True
    except (FutureTimeout, Exception) as exc:  # noqa: BLE001 - any failure means "not deployed / no auth"
        log.info("modal engine unavailable: %s", exc)
        return False
    finally:
        pool.shutdown(wait=False)


def resolve_name(name: str | None = None) -> str:
    """Concrete engine name for `name` (default config.GEOMETRY_ENGINE); resolves 'auto'."""
    name = (name or config.GEOMETRY_ENGINE).lower()
    if name == "auto":
        from codemesh.engines.rocm_engine import ROCM_PYTHON

        return "rocm" if ROCM_PYTHON.exists() else "modal" if modal_available() else "torch"
    if name not in ("modal", "torch", "rocm"):
        raise ValueError(f"unknown geometry engine {name!r}; expected auto|modal|torch|rocm")
    return name


def create_engine(name: str) -> GeometryEngine:
    """Build a new engine by concrete name (modal|torch|rocm)."""
    if name == "modal":
        from codemesh.engines.modal_engine import ModalEngine

        return ModalEngine()
    if name == "rocm":
        from codemesh.engines.rocm_engine import RocmEngine  # written by the ROCm lane

        return RocmEngine()
    import torch

    from codemesh.engines.torch_engine import TorchEngine

    return TorchEngine("mapanything", device="cuda" if torch.cuda.is_available() else "cpu")


def get_engine(name: str | None = None) -> GeometryEngine:
    """The cached engine; the first call picks and loads it. Pass `name` to override the config."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = create_engine(resolve_name(name))
        return _engine


def loaded_engine_info() -> dict | None:
    """info() of the cached engine, or None when no engine has been loaded yet (never loads one)."""
    return _engine.info() if _engine is not None else None


def resolved_engine() -> str:
    """resolve_name() for the configured engine, cached (the modal probe can take ~10 s)."""
    global _resolved
    if _resolved is None:
        _resolved = resolve_name()
    return _resolved


def engine_label() -> str:
    """Engine name for /capabilities; never probes: the loaded or resolved name, else the config value."""
    return (_engine.name if _engine is not None else _resolved) or config.GEOMETRY_ENGINE


def warmup_state() -> str:
    if _state == "cold" and _engine is not None and _engine.name != "modal":
        return "ready"  # a job already loaded a local engine
    return _state


def _noise(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (WARMUP_SIZE[1], WARMUP_SIZE[0], 3), dtype=np.uint8) for _ in range(n)]


def _warm() -> None:
    global _state
    try:
        engine = get_engine(resolved_engine())
        if engine.name == "rocm":
            engine.infer(_noise(WARMUP_VIEWS))  # compiles the kernels for the draft view count
        elif engine.name == "modal":
            engine.worker.warmup.spawn().get()  # loads the model on a GPU container without inference
        _state = "ready"
    except Exception:  # noqa: BLE001 - warming is best effort; the first job reports real errors
        log.exception("engine warmup failed")
        _state = "cold"


def warmup() -> str:
    """Start the background warmup unless already running or done (idempotent); returns the state."""
    global _state
    with _state_lock:
        if warmup_state() != "cold":
            return warmup_state()
        _state = "warming"
    threading.Thread(target=_warm, name="engine-warmup", daemon=True).start()
    return "warming"
