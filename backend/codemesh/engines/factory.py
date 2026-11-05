"""Pick and cache the process-wide GeometryEngine from config.GEOMETRY_ENGINE.

modal -> ModalEngine (deployed app 'codemesh', class GeometryWorker), torch -> TorchEngine
MapAnything on CUDA if available else CPU, rocm -> RocmEngine, auto -> modal when the deployed
class resolves within ~10 s, else torch. Loading weights is slow, so the engine is a singleton.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from codemesh import config
from codemesh.engines.base import GeometryEngine

log = logging.getLogger(__name__)
MODAL_PROBE_TIMEOUT_S = 10.0

_engine: GeometryEngine | None = None
_lock = threading.Lock()


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
        return "modal" if modal_available() else "torch"
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
