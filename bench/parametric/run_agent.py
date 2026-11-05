"""Run the Muse agent loop on an existing evaluated job dir (no job store / server needed).

uv run --project backend python bench/parametric/run_agent.py <job_dir> [settings-json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from codemesh.agent import loop


def main() -> None:
    root = Path(sys.argv[1])
    settings = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    ctx = SimpleNamespace(settings=settings, paths=SimpleNamespace(root=root), job_id="bench",
                          progress=lambda f, m=None: print(f"[{f:.2f}] {m}", flush=True),
                          set_artifact=lambda k, p: None,
                          store=SimpleNamespace(update=lambda *a, **k: None))
    print(json.dumps(loop.run(ctx), indent=1, default=str))


if __name__ == "__main__":
    main()
