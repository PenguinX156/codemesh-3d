# MapAnything on the Radeon 780M (PyTorch ROCm) - results

Measured 2026-11-04 on Ryzen 7 7840HS / Radeon 780M (gfx1103, 14.3 GB visible, shared with system RAM), Windows 11,
TheRock nightly torch 2.15.0+rocm10.3.0a20261104 (see ROCM.md), `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
Model: `models/map-anything-apache` via `TorchEngine('mapanything', device='cuda')`, run inside the persistent worker
(`bench/rocm/worker.py`, driven by `backend/codemesh/engines/rocm_engine.py`). Input: `third_party/vggt/examples/kitchen`
frames, `unique(round(linspace(0, 24, N)))`; engine resolution 336x518. Other lanes (an `e2e.py` CPU run, Modal deploys) were
running on the same machine during the benchmark, so absolute times are indicative.

Reproduce: `uv run --project backend python bench/rocm/bench_mapanything.py [--precision fp16] [--n 8 16 24]`
(raw JSON in `bench/rocm/out/bench_<precision>.json`, logs `run_<precision>.log`).

## Accuracy vs CPU fp32 reference (`bench/phase0/out/ref_mapanything_kitchen_8.npz`, N=8)

The only MapAnything reference that exists is N=8; N=16/24 are timing only. Same world frame (view 0 = origin, metric).
Depth AbsRel is over pixels with reference depth > 0 and reference confidence >= per-view median.

| metric | bf16 autocast (engine default on this GPU) | fp16 autocast |
|---|---|---|
| rotation error, mean / max (deg) | 0.038 / 0.070 | 0.028 / 0.058 |
| camera-center error, mean / max (m; trajectory extent 6.4 m) | 0.0030 / 0.0046 | 0.0006 / 0.0010 |
| depth AbsRel, mean / median / p90 | 3.6e-4 / 3.7e-4 / 6.3e-4 | 4.7e-5 / 4.1e-5 / 1.0e-4 |
| global depth scale ratio (ref / GPU) | 1.0003 | 0.99996 |
| max focal-length relative error | 4.2e-4 | 8.1e-5 |
| all outputs finite | yes | yes |

Both are far inside any tolerance that matters for fusion (rotation < 0.1 deg, depth < 0.1 %). fp16 is about 8x closer to
the fp32 reference than bf16 and showed no overflow on this scene; bf16 is the safer choice for scenes with large activations.

## Timing (engine.infer() wall time including npz file IPC; median of 2 after a warmup run of the same shape)

| frames | bf16 wall (s) | first call of a new shape (s) | fp16 wall (s) | CPU fp32 reference (s) |
|---|---|---|---|---|
| 8 | 11.3 (10.6, 12.0) | 14.8 (112 s when the process was fresh) | 13.4 (12.8, 14.1) | 103 (infer only, 8 threads) |
| 16 | 34.4 (31.0, 37.8) | 205.6 | not run | not measured |
| 24 | 54.1 (50.4, 57.9) | 286.2 | not run | not measured |

- Speed-up vs the CPU reference at N=8: about 9x (103 s -> 11.3 s). The fp16/bf16 difference (11 vs 13 s) is inside the noise.
- Worker start (model load) 34-56 s, plus the first inference of a process 112 s (kernel load/compile), i.e. about 2.5 min
  cold start; the CPU engine needs 126 s to load alone.
- **Every new frame count pays a one-off warmup of 3.5-5 min** (205 s at N=16, 286 s at N=24 observed; the kernels are
  compiled per tensor shape). With the worker kept alive this is paid once per distinct N, so a pipeline using the GPU engine
  should pad/choose a small set of frame counts (e.g. always 8, 16 or 24) rather than arbitrary N.
- Cost scales roughly linearly to slightly super-linearly with frames (about 1.4 s/frame at 8, 2.2 s/frame at 16, 2.3 s/frame at 24).

## Peak RAM

Peak working set of the worker's python process: **10.0 GB at N=8 (fp16)**. This includes the shared-memory GPU
allocations (VRAM is system RAM on this APU) and the 4.9 GB fp32 checkpoint load. The CPU reference peaked at 5.65 GB at N=8.
Peak for N=16/24 was not captured (the bf16 run recorded the launcher process instead of the worker; fixed afterwards, rerun
`bench_mapanything.py` to capture it). The machine has 28 GB, so the GPU worker should not run next to another large model.

## Verdict

Yes, make `RocmEngine` the default local engine when `bench/rocm/.venv` exists and a GPU is present, falling back to the CPU
`TorchEngine`: accuracy is practically identical to the fp32 CPU reference and it is about 9x faster at N=8 once warm.
Caveats: (1) minutes of cold start and per-shape warmup, so start the worker at server boot, run a warmup, and keep the frame
count in a small set; (2) ~10 GB RAM; (3) it depends on a nightly ROCm wheel (pin the versions in ROCM.md); (4) VGGT was not
benchmarked on ROCm (MapAnything only).

## Implementation notes / gotchas

- The ROCm venv's `python.exe` is a launcher that spawns the real interpreter as a child. `RocmEngine` therefore kills the
  worker with `taskkill /F /T` (a plain `Popen.kill()` would orphan the worker and leave its GPU memory allocated).
- The worker keeps the protocol on the original stdout and redirects `sys.stdout` to stderr; stderr goes to `worker.log` in the
  engine's temp dir. A hung worker is killed after `infer_timeout`; a dead worker is restarted on the next call and a request
  that hit a dead worker is retried once.
- Extra packages installed into `bench/rocm/.venv` only: python-dotenv, einops, huggingface-hub, hydra-core, jaxtyping, omegaconf,
  opencv-python-headless, safetensors, scipy, timm, tqdm, trimesh, natsort, orjson, pillow-heif, plyfile, python-box, requests,
  pytest, uniception==0.1.7 (`--no-deps`). MapAnything itself is imported from `third_party/map-anything` via `sys.path`.
- Test: `backend/tests/test_rocm_engine.py` (skips when the venv is absent; the GPU test needs `CODEMESH_TEST_ROCM=1` because
  it takes minutes).
