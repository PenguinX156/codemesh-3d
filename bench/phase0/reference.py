"""Phase 0 reference runs: PyTorch FP32 outputs that other runtimes (ggml, ROCm) must match.

For every (model, image set, N) it writes, under bench/phase0/out/:
  ref_<model>_<set>_<N>.npz   inputs + outputs (layout: bench/phase0/REFERENCE_FORMAT.md)
  ref_<model>_<set>_<N>.ply   colored point cloud from depth + cameras (confident half)
  reference_results.json      timings, peak RSS and geometry self-consistency per run

Each run is a separate subprocess so peak RSS is per run. Usage (repo root):
  uv run --project backend python bench/phase0/reference.py [--models mapanything vggt]
      [--sets kitchen llff_flower] [--n 8 16 24] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "phase0" / "out"
EXAMPLES = ROOT / "third_party" / "vggt" / "examples"


def pick_frames(image_dir: Path, n: int) -> list[Path]:
    """n images spread evenly over the sorted set (all of them if n >= set size)."""
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    idx = np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int))
    return [files[i] for i in idx]


def consistency(geo) -> dict:
    """Project predicted world points with the predicted cameras and compare with depth/pixels.

    Errors are over pixels with confidence >= the per-view median and depth > 0.
    """
    n, h, w = geo.depth.shape
    R, t = geo.extrinsics[:, :3, :3], geo.extrinsics[:, :3, 3]
    cam = np.einsum("nij,nhwj->nhwi", R, geo.points.astype(np.float64)) + t[:, None, None]
    z = cam[..., 2]
    uv = np.einsum("nij,nhwj->nhwi", geo.intrinsics, cam)
    uv = uv[..., :2] / np.where(np.abs(uv[..., 2:]) < 1e-9, 1e-9, uv[..., 2:])
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)[None]
    conf = geo.depth_conf
    valid = (conf >= np.median(conf.reshape(n, -1), axis=1)[:, None, None]) & (geo.depth > 0)
    rel_z = (np.abs(z - geo.depth) / geo.depth)[valid]
    px = np.linalg.norm(uv - grid, axis=-1)[valid]
    px_half = np.linalg.norm(uv - (grid + 0.5), axis=-1)[valid]
    return {
        "depth_rel_err_median": float(np.median(rel_z)),
        "depth_rel_err_p90": float(np.percentile(rel_z, 90)),
        "pixel_err_median": float(np.median(px)),
        "pixel_err_p90": float(np.percentile(px, 90)),
        # Which pixel-center convention the intrinsics follow (lower = that convention).
        "pixel_err_median_if_centers_at_half": float(np.median(px_half)),
    }


def unproject(geo) -> np.ndarray:
    """World points (N, H, W, 3) from depth + intrinsics + extrinsics (integer pixel centers)."""
    n, h, w = geo.depth.shape
    grid = np.stack([*np.meshgrid(np.arange(w), np.arange(h)), np.ones((h, w))], -1)
    rays = np.einsum("nij,hwj->nhwi", np.linalg.inv(geo.intrinsics), grid)
    cam = rays * geo.depth[..., None]
    cam_to_world = np.linalg.inv(geo.extrinsics)
    return np.einsum("nij,nhwj->nhwi", cam_to_world[:, :3, :3], cam) + cam_to_world[:, None, None, :3, 3]


def run_one(model: str, set_name: str, n: int, device: str) -> dict:
    """Load model, infer, save npz/ply, return the result record (runs in a child process)."""
    import psutil
    import torch
    import trimesh
    from PIL import Image

    from codemesh.engines.torch_engine import TorchEngine

    frames = pick_frames(EXAMPLES / set_name / "images", n)
    images = [np.asarray(Image.open(p).convert("RGB")) for p in frames]
    engine = TorchEngine(model, device=device, precision="fp32")
    t0 = time.perf_counter()
    geo = engine.infer(images)
    wall = time.perf_counter() - t0

    stem = f"ref_{model}_{set_name}_{len(frames)}"
    from_depth = unproject(geo).astype(np.float32)
    np.savez_compressed(
        OUT / f"{stem}.npz",
        frame_names=np.array([p.name for p in frames]),
        original_hw=np.array([im.shape[:2] for im in images]),
        inputs=engine.last_inputs,
        intrinsics=geo.intrinsics,
        extrinsics=geo.extrinsics,
        depth=geo.depth,
        depth_conf=geo.depth_conf,
        points=geo.points,
        points_conf=geo.points_conf,
        points_from_depth=from_depth,
        pixel_to_original=geo.pixel_to_original,
        metric_scale=np.array(geo.metric_scale),
        **engine.last_extras,
    )

    # Colors: the engine-resolution input images (undo VGGT [0,1] / MapAnything DINOv2 normalization).
    rgb = engine.last_inputs.transpose(0, 2, 3, 1)
    if model == "mapanything":
        rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    keep = (geo.depth > 0) & (geo.depth_conf >= np.median(geo.depth_conf.reshape(len(frames), -1), 1)[:, None, None])
    colors = (np.clip(rgb[keep], 0, 1) * 255).astype(np.uint8)
    trimesh.PointCloud(from_depth[keep], colors=colors).export(OUT / f"{stem}.ply")

    vm = psutil.Process().memory_info()
    record = {
        "model": model,
        "set": set_name,
        "n": len(frames),
        "frames": [p.name for p in frames],
        **{k: v for k, v in geo.info.items() if k not in {"num_views"}},
        "wall_seconds": round(wall, 2),
        "torch_threads": torch.get_num_threads(),
        "peak_rss_gb": round(getattr(vm, "peak_wset", vm.rss) / 2**30, 2),
        "npz": f"{stem}.npz",
        "points_vs_cameras": consistency(geo),
        "points_vs_depth_unprojection_median": float(
            np.median(np.linalg.norm(geo.points - from_depth, axis=-1)[keep])
        ),
        "scene_extent_median": float(np.median(np.linalg.norm(from_depth[keep] - from_depth[keep].mean(0), axis=-1))),
    }
    engine.close()
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["mapanything", "vggt"])
    ap.add_argument("--sets", nargs="+", default=["kitchen", "llff_flower"])
    ap.add_argument("--n", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--one", nargs=3, metavar=("MODEL", "SET", "N"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.one:  # child process: print one JSON record on the last line
        print(json.dumps(run_one(args.one[0], args.one[1], int(args.one[2]), args.device)))
        return

    results_path = OUT / "reference_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for model in args.models:
        for set_name in args.sets:
            for n in args.n:
                cmd = [sys.executable, __file__, "--one", model, set_name, str(n), "--device", args.device]
                print(f"== {model} {set_name} N={n}", flush=True)
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
                if proc.returncode != 0:
                    print(proc.stderr[-3000:], flush=True)
                    continue
                rec = json.loads(proc.stdout.strip().splitlines()[-1])
                results = [r for r in results if (r["model"], r["set"], r["n"]) != (rec["model"], rec["set"], rec["n"])]
                results.append(rec)
                results_path.write_text(json.dumps(results, indent=2))
                c = rec["points_vs_cameras"]
                print(
                    f"   infer {rec['infer_seconds']}s  load {rec['load_seconds']}s  peak {rec['peak_rss_gb']} GB  "
                    f"hw {rec['engine_hw']}  depth rel err {c['depth_rel_err_median']:.2e}  px err {c['pixel_err_median']:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
