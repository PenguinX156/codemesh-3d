"""Benchmark RocmEngine (MapAnything on the 780M) against the CPU fp32 reference.

    uv run --project backend python bench/rocm/bench_mapanything.py [--precision bf16] [--n 8 16 24] [--repeats 2]

Accuracy is compared on N=8 (the only MapAnything reference that exists, ref_mapanything_kitchen_8.npz);
other N report timing only. Timing = engine.infer() wall time incl. file IPC (median of repeats, after
one warmup run); the worker's own infer_seconds is reported next to it. Results go to out/bench_<precision>.json.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from codemesh.engines.rocm_engine import RocmEngine  # noqa: E402

IMG_DIR = ROOT / "third_party" / "vggt" / "examples" / "kitchen" / "images"
REF = ROOT / "bench" / "phase0" / "out" / "ref_mapanything_kitchen_8.npz"


def peak_rss_gb(pid: int) -> float:
    """Peak working set of a process (Windows)."""

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    c = Counters()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
        raise OSError(f"GetProcessMemoryInfo failed for pid {pid}")
    k32.CloseHandle(h)
    return c.PeakWorkingSetSize / 2**30


def worker_pid(launcher_pid: int) -> int:
    """The venv python.exe is a launcher; the model lives in its child process."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={launcher_pid}').ProcessId"],
        capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else launcher_pid


def load_frames(n: int) -> list[np.ndarray]:
    files = sorted(IMG_DIR.glob("*.png"))
    idx = np.unique(np.round(np.linspace(0, len(files) - 1, n)).astype(int))
    return [np.asarray(Image.open(files[i]).convert("RGB")) for i in idx]


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.trace(Ra @ Rb.transpose(0, 2, 1), axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def centers(ext: np.ndarray) -> np.ndarray:
    return -np.einsum("nij,nj->ni", ext[:, :3, :3].transpose(0, 2, 1), ext[:, :3, 3])


def accuracy(geo, ref) -> dict:
    """Pose and depth error vs the CPU fp32 reference (same world frame: view 0 = origin)."""
    rot = rot_err_deg(geo.extrinsics[:, :3, :3], ref["extrinsics"][:, :3, :3])
    cdist = np.linalg.norm(centers(geo.extrinsics) - centers(ref["extrinsics"]), axis=1)
    extent = np.linalg.norm(centers(ref["extrinsics"]) - centers(ref["extrinsics"])[0], axis=1).max()
    d, dr, conf = geo.depth, ref["depth"], ref["depth_conf"]
    keep = np.stack([(dr[i] > 0) & (conf[i] >= np.median(conf[i])) for i in range(len(dr))])
    absrel = np.abs(d - dr)[keep] / dr[keep]
    scale = np.median(dr[keep] / d[keep])  # depth error after removing a global scale difference
    absrel_aligned = np.abs(d * scale - dr)[keep] / dr[keep]
    return {
        "rotation_err_deg_mean": float(rot.mean()), "rotation_err_deg_max": float(rot.max()),
        "center_err_m_mean": float(cdist.mean()), "center_err_m_max": float(cdist.max()),
        "baseline_extent_m": float(extent),
        "depth_absrel_mean": float(absrel.mean()), "depth_absrel_median": float(np.median(absrel)),
        "depth_absrel_p90": float(np.percentile(absrel, 90)),
        "depth_absrel_scale_aligned_mean": float(absrel_aligned.mean()), "global_scale_ratio": float(scale),
        "intrinsics_focal_rel_err_max": float(np.abs(geo.intrinsics[:, 0, 0] / ref["intrinsics"][:, 0, 0] - 1).max()),
        "finite": bool(np.isfinite(d).all() and np.isfinite(geo.extrinsics).all()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default=None)
    ap.add_argument("--n", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    t0 = time.perf_counter()
    engine = RocmEngine(precision=a.precision)
    out: dict = {"engine_info": engine.info(), "startup_seconds": round(time.perf_counter() - t0, 1), "runs": []}
    print("started", out["engine_info"], out["startup_seconds"], "s", flush=True)
    ref = np.load(REF)
    try:
        for n in a.n:
            frames = load_frames(n)
            t = time.perf_counter()
            engine.infer(frames)  # warmup (kernel compile for this shape)
            warm = time.perf_counter() - t
            times, worker_times = [], []
            for _ in range(a.repeats):
                t = time.perf_counter()
                geo = engine.infer(frames)
                times.append(time.perf_counter() - t)
                worker_times.append(geo.info["infer_seconds"])
            rec = {"n": n, "warmup_seconds": round(warm, 1), "wall_seconds": [round(x, 1) for x in times],
                   "wall_median": round(statistics.median(times), 1), "worker_infer_seconds": worker_times,
                   "engine_hw": geo.info["engine_hw"], "peak_rss_gb": round(peak_rss_gb(worker_pid(engine._proc.pid)), 2)}
            if n == 8:
                rec["accuracy"] = accuracy(geo, ref)
            out["runs"].append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        engine.close()
    dest = Path(__file__).parent / "out"
    dest.mkdir(exist_ok=True)
    (dest / f"bench_{out['engine_info'].get('precision', 'x')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
