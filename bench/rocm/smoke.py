"""ROCm smoke tests for the 780M. Each test runs in its own subprocess with a timeout.

Usage (from bench/rocm):  .venv/Scripts/python smoke.py [test_name ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

TIMEOUT_S = 300


def _sync(torch):
    torch.cuda.synchronize()


def _time(torch, fn, iters=5):
    fn()  # warmup / kernel compile
    _sync(torch)
    t = time.perf_counter()
    for _ in range(iters):
        out = fn()
    _sync(torch)
    return (time.perf_counter() - t) / iters, out


def _rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item()


def t_device() -> dict:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0),
        "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"),
        "mem_free_gb": round(free / 2**30, 2),
        "mem_total_gb": round(total / 2**30, 2),
    }


def _matmul(dtype_name: str) -> dict:
    import torch

    dt = getattr(torch, dtype_name)
    n = 4096
    g = torch.Generator().manual_seed(0)
    a = torch.randn(n, n, generator=g).to(dt)
    b = torch.randn(n, n, generator=g).to(dt)
    ag, bg = a.cuda(), b.cuda()
    sec, out = _time(torch, lambda: ag @ bg)
    ref = a.float() @ b.float()
    return {"tflops": round(2 * n**3 / sec / 1e12, 2), "ms": round(sec * 1e3, 1), "max_rel_err_vs_cpu_fp32": _rel(out, ref)}


def t_matmul_fp16() -> dict:
    return _matmul("float16")


def t_matmul_fp32() -> dict:
    return _matmul("float32")


def t_conv2d() -> dict:
    import torch

    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, 128, 128, generator=g)
    w = torch.randn(128, 64, 3, 3, generator=g)
    xg, wg = x.cuda(), w.cuda()
    sec, out = _time(torch, lambda: torch.nn.functional.conv2d(xg, wg, padding=1))
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    return {"ms": round(sec * 1e3, 1), "max_rel_err_vs_cpu": _rel(out, ref)}


def t_sdpa() -> dict:
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    shp = (1, 16, 4096, 64)
    q, k, v = (torch.randn(shp, generator=g).half() for _ in range(3))
    qg, kg, vg = q.cuda(), k.cuda(), v.cuda()
    sec, out = _time(torch, lambda: F.scaled_dot_product_attention(qg, kg, vg))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    flops = 4 * 16 * 4096 * 4096 * 64
    return {"ms": round(sec * 1e3, 1), "tflops": round(flops / sec / 1e12, 2), "max_rel_err_vs_cpu_fp32": _rel(out, ref)}


TESTS = {f.__name__[2:]: f for f in (t_device, t_matmul_fp32, t_matmul_fp16, t_conv2d, t_sdpa)}


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print("RESULT " + json.dumps(TESTS[sys.argv[2]]()))
        return
    for name in sys.argv[1:] or TESTS:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, "--child", name], capture_output=True, text=True, timeout=TIMEOUT_S)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            res = line[7:] if line else f"FAILED rc={p.returncode} stderr={p.stderr.strip()[-600:]}"
        except subprocess.TimeoutExpired:
            res = f"TIMEOUT after {TIMEOUT_S}s"
        print(f"{name:14s} [{time.time() - t0:5.1f}s] {res}", flush=True)


if __name__ == "__main__":
    main()
