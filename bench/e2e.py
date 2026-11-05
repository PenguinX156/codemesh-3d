"""Run the real CodeMesh pipeline on one capture through the registered stage functions (no HTTP).

    uv run --project backend python bench/e2e.py capture.mp4 --text "lego bulldozer" --preset draft
    uv run --project backend python bench/e2e.py --from-images third_party/vggt/examples/kitchen/images \\
        data/captures/kitchen.mp4 --text "lego bulldozer" --engine torch --preset draft

Prints per-stage seconds + metrics and the GLB path; the job dir stays under --jobs-dir.
Meta API cost: one SAM 3.1 video call (billed >= 50 frames, $0.01).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from codemesh.jobs.context import StageContext  # noqa: E402
from codemesh.jobs.models import Job  # noqa: E402
from codemesh.jobs.paths import job_paths  # noqa: E402
from codemesh.jobs.store import JobStore  # noqa: E402
from codemesh.pipeline import ACTIONS  # noqa: E402


def images_to_video(src_dir: Path, dst: Path, fps: int = 2) -> Path:
    """Encode the sorted images of `src_dir` as an H.264 MP4 (even width/height)."""
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(dst), "w") as out:
        stream = None
        for f in files:
            rgb = np.asarray(Image.open(f).convert("RGB"))
            if stream is None:
                stream = out.add_stream("libx264", rate=fps)
                stream.width, stream.height = rgb.shape[1] // 2 * 2, rgb.shape[0] // 2 * 2
                stream.pix_fmt, stream.options = "yuv420p", {"crf": "16"}
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[: stream.height, : stream.width]), "rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return dst


def run_action(store: JobStore, jobs_dir: Path, job: Job, action: str, settings: dict) -> dict[str, dict]:
    """Run every stage of `action` like the job runner does; return {stage: metrics + seconds}."""
    store.update(job.id, lambda j: j.settings.update(settings))
    paths = job_paths(jobs_dir, job.id)
    stages = ACTIONS[action].stages
    results = {}
    for i, stage in enumerate(stages):
        ctx = StageContext(store, job.id, paths, store.get(job.id).settings, stage.name, i, len(stages))
        t0 = time.perf_counter()
        metrics = dict(stage.run(ctx) or {})
        results[stage.name] = {"wall_seconds": round(time.perf_counter() - t0, 2), **metrics}
        print(f"[{stage.name}] {results[stage.name]['wall_seconds']} s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path, help="input video (written first when --from-images is given)")
    ap.add_argument("--from-images", type=Path, help="encode this image folder into `video` first (2 fps)")
    ap.add_argument("--text", help="SAM concept prompt, e.g. 'lego bulldozer'")
    ap.add_argument("--point", help="x,y normalized click on frame 0")
    ap.add_argument("--engine", default=None, help="auto|modal|torch|rocm (default: config)")
    ap.add_argument("--preset", default="draft", choices=("draft", "standard", "high"))
    ap.add_argument("--mode", default="walkaround", choices=("walkaround", "turntable"))
    ap.add_argument("--jobs-dir", type=Path, default=REPO / "jobs" / "bench")
    args = ap.parse_args()
    if not args.text and not args.point:
        ap.error("give --text and/or --point")
    if args.from_images:
        images_to_video(args.from_images, args.video)

    store = JobStore(args.jobs_dir / "bench.db")
    job = store.create(Job(mode=args.mode, settings={"preset": args.preset}))
    paths = job_paths(args.jobs_dir, job.id)
    paths.create()
    shutil.copy2(args.video, paths.input / args.video.name)

    points = []
    if args.point:
        x, y = (float(v) for v in args.point.split(","))
        points = [{"x": x, "y": y, "label": 1}]
    report = {"job_dir": str(paths.root)}
    report |= run_action(store, args.jobs_dir, job, "ingest", {})
    report |= run_action(store, args.jobs_dir, job, "segment", {"selection": {"frame_index": 0, "points": points, "text": args.text}})
    report |= run_action(store, args.jobs_dir, job, "reconstruct", {"reconstruct": {"preset": args.preset, "triangle_budget": "auto",
                                                                      "texture_size": 2048, "engine": args.engine}})
    final = store.get(job.id)
    report["job_metrics"] = final.metrics
    report["artifacts"] = final.artifacts
    report["glb"] = str(paths.root / "export" / "model.glb")
    (paths.root / "e2e_report.json").write_text(json.dumps(report, indent=1, default=str))
    for name in ("ingest", "segment", "geometry", "fuse", "mesh", "texture", "export", "evaluate"):
        r = report.get(name, {})
        print(f"{name:9s} {r.get('wall_seconds', '-'):>8} s")
    print(json.dumps(final.metrics, indent=1, default=str))
    print("GLB:", report["glb"])


if __name__ == "__main__":
    main()
