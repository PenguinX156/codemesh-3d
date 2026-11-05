"""Persistent ROCm worker: loads TorchEngine once, then serves `infer` requests.

Run with the ROCm venv: bench/rocm/.venv/Scripts/python.exe bench/rocm/worker.py [model] [precision]

Protocol (one JSON object per line):
  stdout -> {"event": "ready", "info": {...}}                      after the model is loaded
  stdin  <- {"cmd": "infer", "inp": "<in.npz>", "out": "<out.npz>"}
  stdout -> {"ok": true, "info": {...}} | {"ok": false, "error": "..."}
  stdin  <- {"cmd": "quit"}                                         (or EOF) -> exit
`in.npz` holds `img_0..img_{N-1}` (RGB uint8 HxWx3). `out.npz` holds the ViewGeometry arrays
plus `info_json`. Anything the libraries print goes to stderr so stdout stays clean.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # otherwise attention is ~30x slower

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "backend", ROOT / "third_party" / "map-anything", ROOT / "third_party" / "vggt"):
    sys.path.insert(0, str(p))

_stdout = sys.stdout
sys.stdout = sys.stderr  # protect the protocol channel from stray prints


def send(obj: dict) -> None:
    _stdout.write(json.dumps(obj) + "\n")
    _stdout.flush()


def main() -> None:
    import numpy as np

    from codemesh.engines.torch_engine import TorchEngine

    model = sys.argv[1] if len(sys.argv) > 1 else "mapanything"
    precision = sys.argv[2] if len(sys.argv) > 2 else None  # None -> engine default for the device
    engine = TorchEngine(model, device="cuda", precision=precision)
    send({"event": "ready", "info": engine.info()})

    for line in sys.stdin:
        req = json.loads(line)
        if req["cmd"] == "quit":
            break
        try:
            with np.load(req["inp"]) as z:
                images = [z[f"img_{i}"] for i in range(len(z.files))]
            geo = engine.infer(images)
            np.savez(
                req["out"],
                intrinsics=geo.intrinsics, extrinsics=geo.extrinsics, depth=geo.depth,
                depth_conf=geo.depth_conf, pixel_to_original=geo.pixel_to_original,
                points=geo.points, points_conf=geo.points_conf,
                metric_scale=geo.metric_scale, info_json=json.dumps(geo.info),
            )
            send({"ok": True, "info": geo.info})
        except Exception:  # keep serving; the client decides whether to restart
            send({"ok": False, "error": traceback.format_exc()})
    engine.close()


if __name__ == "__main__":
    main()
