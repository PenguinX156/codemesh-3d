"""Stage cache: skip a stage when its outputs exist and were built from the same inputs/settings.

Usage inside a stage:

    key = stage_key("ingest", input_files, settings)
    if (hit := cached_metrics(out_dir, "ingest", key)) is not None:
        return {**hit, "cached": True}
    ... do the work, produce `outputs` ...
    write_manifest(out_dir, "ingest", key, outputs, metrics)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _manifest_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f".cache-{stage}.json"


def stage_key(stage: str, inputs: Iterable[Path], settings: dict[str, Any]) -> str:
    """Hash of (stage name, input file name/size/mtime, settings JSON).

    shortcut: inputs are fingerprinted by size + mtime_ns, not content, so multi-GB videos are
    not re-read; switch to content hashes if inputs are ever rewritten in place with equal size.
    """
    h = hashlib.sha256(stage.encode())
    for path in sorted(Path(p) for p in inputs):
        st = path.stat()
        h.update(f"|{path.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()


def cached_metrics(out_dir: Path, stage: str, key: str) -> dict[str, Any] | None:
    """Metrics of the previous run if its manifest matches `key` and all outputs still exist."""
    try:
        manifest = json.loads(_manifest_path(out_dir, stage).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("key") != key:
        return None
    if not all((out_dir / rel).is_file() for rel in manifest.get("outputs", [])):
        return None
    return dict(manifest.get("metrics", {}))


def write_manifest(
    out_dir: Path, stage: str, key: str, outputs: Iterable[Path], metrics: dict[str, Any]
) -> None:
    """Record a finished stage. Write this last, so a crash mid-stage never looks cached."""
    rels = [Path(p).resolve().relative_to(out_dir.resolve()).as_posix() for p in outputs]
    manifest = {"stage": stage, "key": key, "outputs": rels, "metrics": metrics}
    _manifest_path(out_dir, stage).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def invalidate(out_dir: Path, stage: str) -> None:
    """Forget the cached result (call before rewriting a stage's outputs)."""
    _manifest_path(out_dir, stage).unlink(missing_ok=True)
