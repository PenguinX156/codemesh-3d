"""ViT-L-like transformer stack (24 layers, d=1024, 16 heads, 1024 tokens, fp16) on GPU vs CPU fp32 reference.
Run with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1."""
import time, torch
torch.manual_seed(0)
layer = torch.nn.TransformerEncoderLayer(1024, 16, 4096, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
model = torch.nn.TransformerEncoder(layer, 24, enable_nested_tensor=False).eval()
x = torch.randn(1, 1024, 1024)
with torch.no_grad():
    t = time.perf_counter(); ref = model(x); cpu_s = time.perf_counter() - t
    mg = model.half().cuda(); xg = x.half().cuda()
    mg(xg); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3): out = mg(xg)
    torch.cuda.synchronize(); gpu_s = (time.perf_counter() - t) / 3
rel = ((out.float().cpu() - ref).abs().max() / ref.abs().max()).item()
print(f"ViT-L 24L 1024tok  GPU fp16 {gpu_s*1e3:.0f} ms | CPU fp32 {cpu_s*1e3:.0f} ms | speedup {cpu_s/gpu_s:.1f}x | max_rel_err {rel:.2e} | peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
