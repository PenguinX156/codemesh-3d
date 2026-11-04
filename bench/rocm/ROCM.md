# PyTorch ROCm on Windows, Radeon 780M (gfx1103)

**Verdict: usable for transformer inference.** AMD TheRock nightly wheels run on the 780M under Windows with fp16/fp32 matmul, conv2d and SDPA numerically matching CPU. Two caveats: set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (otherwise attention is 20-30x slower), and expect a long first-run warmup (30-130 s per fresh process while kernels load).

Measured 2026-11-04 on Ryzen 7 7840HS / 780M, AMD driver 26.8.1, Windows 11. Other lanes were running at the same time, so absolute numbers are indicative.

## Install (exact)

```powershell
cd bench/rocm
uv venv --python 3.12 .venv
# own cache dir avoids "Access is denied" on the shared uv cache when other uv processes run in parallel
$env:UV_CACHE_DIR = "$env:TEMP\uvcache_rocm"
uv pip install --python .venv/Scripts/python.exe --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1103]" "torchvision[device-gfx1103]"
```

Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md ("Installing multi-arch releases"); gfx1103 is Build/Sanity/Release-ready on Windows in SUPPORTED_GPUS.md. Extra: `device-gfx1103` pulls `amd-torch-device-gfx1103` + `amd-torch-device-gfx110x` (the latter holds the AOTriton attention kernels). Download is ~300 MB total (about 10+ min on this link; the first attempt hit the cache rename error above).

Installed versions: torch 2.15.0+rocm10.3.0a20261104, torchvision 0.29.2+rocm10.3.0a20261104, rocm-sdk-core/libraries/device-gfx1103 10.3.0a20261104, numpy 2.5.3, Python 3.12. HIP 7.18.26441. No system ROCm install needed; the pip packages are self-contained.

Note: this is a nightly (`whl-next`), so versions drift daily. Pin the exact versions above for reproducibility.

## Results (`smoke.py`, each test in its own subprocess, 300 s timeout)

| Test | Result | Max rel err vs CPU fp32 |
|---|---|---|
| `torch.cuda.is_available()` | True, "AMD Radeon 780M Graphics", arch gfx1103 | |
| Memory | 14.3 GB free of 14.45 GB visible to torch (shared with system RAM) | |
| matmul 4096^2 fp32 | 0.33 TFLOPS (423 ms) | 2.6e-6 |
| matmul 4096^2 fp16 | 1.87 TFLOPS (73.6 ms) | 3.5e-4 |
| conv2d (8x64x128x128, 3x3, 128 out) fp32 | 20.0 ms | 4.8e-7 |
| SDPA fp16 (1x16x4096x64), default, no env var | 810 ms = 0.08 TFLOPS (math fallback) | 3.1e-4 |

Attention backends (`sdpa_backends.py`, SDPA fp16 seq 4096, 16 heads, dim 64):

| Backend | env var unset | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
|---|---|---|
| math | 563 ms (0.12 TFLOPS) | 512 ms (0.13 TFLOPS) |
| mem-efficient | "No available kernel" | 29.0 ms (2.37 TFLOPS) |
| flash | "No available kernel" | 24.2 ms (2.84 TFLOPS) |

End-to-end (`vit_block.py`, env var set): ViT-L-like 24 layers, d=1024, 16 heads, 1024 tokens, fp16 on GPU = 433 ms, peak 0.61 GB, max rel err 2.4e-3 vs the fp32 reference. The CPU reference took 70.9 s in this venv; that CPU number is not representative (ROCm wheel CPU path under load; CPU 4096^2 fp32 matmul measured 0.02 TFLOPS here), so do not quote the 164x speedup as a real-world figure. Compare against the backend venv's CPU torch if a ratio is needed.

## What failed / gotchas

- `uv pip install` error `failed to rename file ... archive-v0\...: Access is denied. (os error 5)`: concurrent uv processes in the shared cache. Fix: private `UV_CACHE_DIR`.
- Without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`: warning "Mem Efficient attention on Current AMD GPU is still experimental", flash/efficient SDP raise "No available kernel", default SDPA silently uses the math path (about 30x slower). Set the env var in the backend process before importing torch.
- Startup latency: first process run costs 35-130 s per test (kernel/library load and first-use compile); see per-test wall times in `smoke.py` output. Plan for a persistent worker process, not one process per request.
- fp32 is slow (0.33 TFLOPS); use fp16 (bf16 not tested).
- VRAM is shared system memory: torch sees ~14.4 GB of the 28 GB.

## Reproduce

```bash
cd bench/rocm
.venv/Scripts/python.exe smoke.py                       # 5 subprocess tests
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/Scripts/python.exe sdpa_backends.py
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/Scripts/python.exe vit_block.py
```

Not tested: actually running VGGT / MapAnything on this stack (next step: install their requirements into `bench/rocm/.venv` and set the env var).
