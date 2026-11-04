"""SDPA backend comparison + CPU baseline. Run: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python sdpa_backends.py"""
import os, time, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
print("aotriton_experimental env:", os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"), flush=True)
q, k, v = (torch.randn(1, 16, 4096, 64, dtype=torch.half, device="cuda") for _ in range(3))
flops = 4 * 16 * 4096 * 4096 * 64
for name, be in [("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION)]:
    try:
        ctx = sdpa_kernel(be) if be else sdpa_kernel(list(SDPBackend))
        with ctx:
            F.scaled_dot_product_attention(q, k, v); torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(3): F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize(); s = (time.perf_counter() - t) / 3
        print(f"{name:10s} {s*1e3:8.1f} ms  {flops/s/1e12:.2f} TFLOPS", flush=True)
    except Exception as e:
        print(f"{name:10s} FAILED {str(e)[:160]}", flush=True)
a = torch.randn(4096, 4096); b = torch.randn(4096, 4096)
t = time.perf_counter(); a @ b; s = time.perf_counter() - t
print(f"CPU fp32 matmul 4096: {s*1e3:.0f} ms = {2*4096**3/s/1e12:.2f} TFLOPS", flush=True)
