"""Re-run the tail of the pipeline on an existing job dir.

uv run --project backend python bench/quality/rerun.py <job_dir> [--from fuse|mesh|texture|export|evaluate]
       [--set key=json ...]   e.g. --set mesh_method='"poisson"' --set preset='"high"'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from codemesh.pipeline import evaluate, export, fuse, mesh, texture

STAGES = {"fuse": fuse, "mesh": mesh, "texture": texture, "export": export, "evaluate": evaluate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--from", dest="start", default="mesh", choices=list(STAGES))
    ap.add_argument("--set", action="append", default=[], help="key=json setting")
    a = ap.parse_args()
    settings = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    names = list(STAGES)[list(STAGES).index(a.start):]
    for name in names:
        t = time.perf_counter()
        m = STAGES[name].run(a.job_dir, settings, lambda f, msg: None)
        brief = {k: v for k, v in (m or {}).items() if not isinstance(v, (list, dict))}
        print(f"[{name}] {time.perf_counter() - t:.1f}s {json.dumps(brief)}", flush=True)


if __name__ == "__main__":
    main()
