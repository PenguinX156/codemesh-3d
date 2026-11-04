# Phase 0 reference outputs (accuracy ground truth)

Produced by `bench/phase0/reference.py` with `TorchEngine` (`backend/codemesh/engines/torch_engine.py`):
official PyTorch model code, **FP32, CPU**, no autocast. Other runtimes (ggml/Vulkan, ROCm, ONNX)
compare against these files.

Regenerate (repo root):

```
uv run --project backend python bench/phase0/reference.py --models mapanything vggt --sets kitchen llff_flower --n 8 16 24
```

## Runs

| model | weights (models/) | image sets (third_party/vggt/examples/<set>/images) | N |
|---|---|---|---|
| `mapanything` | `map-anything-apache/model.safetensors` (4,914,062,480 B, Apache-2.0) | `kitchen` (25 × 779×520 PNG), `llff_flower` (25 × 706×529 PNG) | 8, 16, 24 |
| `vggt` | `VGGT-1B/model.safetensors` (5,026,367,224 B, CC-BY-NC) | same | 8, 16, 24 |

N = 32 is not possible: both sets have 25 images.

**Frame selection:** files sorted by name, indices `unique(round(linspace(0, 24, N)))`. The chosen
names, in input order, are in the npz key `frame_names` (view 0 = first name). Images are read with
`PIL.Image.open(p).convert("RGB")` (no EXIF rotation; the PNGs have none).

## Files in `bench/phase0/out/` (gitignored)

- `ref_<model>_<set>_<N>.npz` — `np.savez_compressed`, keys below.
- `ref_<model>_<set>_<N>.ply` — colored point cloud from `points_from_depth`, pixels with
  `depth > 0` and `depth_conf >= per-view median`, colors from the engine-resolution input.
- `reference_results.json` — one record per run: timings (`load_seconds`, `preprocess_seconds`,
  `infer_seconds`), `peak_rss_gb` (Windows peak working set of the run's own process),
  `torch_threads`, `engine_hw`, and geometry self-consistency (`points_vs_cameras`).

## npz keys

N views, H × W engine resolution (`kitchen`: MapAnything 336×518, VGGT 350×518).

| key | shape | dtype | meaning |
|---|---|---|---|
| `frame_names` | (N,) | str | input image file names, in view order |
| `original_hw` | (N, 2) | int64 | original image (height, width) |
| `inputs` | (N, 3, H, W) | float32 | **exact model input tensor** (feed this to other runtimes). MapAnything: DINOv2-normalized `(x/255 - mean) / std`, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). VGGT: `x/255` in [0, 1] (the model normalizes internally). RGB channel order. |
| `intrinsics` | (N, 3, 3) | float64 | pinhole K in engine pixels |
| `extrinsics` | (N, 4, 4) | float64 | world → camera, OpenCV axes (x right, y down, z forward) |
| `depth` | (N, H, W) | float32 | z-depth in the camera frame |
| `depth_conf` | (N, H, W) | float32 | confidence, higher = better. MapAnything: `raw_conf` × `non_ambiguous_mask` (ambiguous pixels → 0). VGGT: `depth_conf` head. |
| `points` | (N, H, W, 3) | float32 | world points from the model. MapAnything: `pts3d` (= rays × depth, posed). VGGT: point head `world_points` (independent of the depth head). |
| `points_conf` | (N, H, W) | float32 | MapAnything: same as `depth_conf`. VGGT: `world_points_conf`. |
| `points_from_depth` | (N, H, W, 3) | float32 | `depth` unprojected with `intrinsics` + `extrinsics` (what the fuse stage uses) |
| `pixel_to_original` | (N, 3, 3) | float64 | engine pixel (u, v, 1) → original-image pixel |
| `metric_scale` | () | bool | True for MapAnything (metric, meters), False for VGGT (up to scale) |
| `raw_conf` | (N, H, W) | float32 | MapAnything only: `conf` before ambiguity masking |
| `non_ambiguous_mask` | (N, H, W) | bool | MapAnything only: model output `non_ambiguous_mask` (> 0.5) |

## Conventions

- **Pixels:** integer coordinates are pixel centers; engine pixel (u, v) = column u, row v of
  `depth`. Both models' own unprojection uses `arange(W)`/`arange(H)` grids with these K.
- **pixel_to_original:** for resize W→w then crop at offset (ox, oy):
  `x_orig = (u + ox + 0.5) * W / w - 0.5` (same for y). Matches PIL resize sampling.
- **World frame:** whatever the model outputs (both: approximately the first camera, so
  `extrinsics[0] ≈ I`). Compare cameras across runtimes after aligning, or directly if the
  runtime reproduces the same network.
- **Preprocessing** (`backend/codemesh/engines/preprocess.py`, checked equal to the official
  loaders to 1e-5 by `backend/tests/test_torch_engine.py`):
  - MapAnything = `load_images(resize_mode="fixed_mapping", resolution_set=518)`: target from the
    mean aspect ratio via its table (kitchen 1.50 → 518×336), scale by `max(tw/W, th/H) + 1e-8`
    to `floor` size (Lanczos down, bicubic up), center crop `(w - tw) // 2`.
  - VGGT = `load_and_preprocess_images(mode="crop")`: width → 518 bicubic, height
    `round(H * 518 / W / 14) * 14`, center-crop to 518 if taller.
- **MapAnything inference call:** `model.infer(views, memory_efficient_inference=True,
  use_amp=False, apply_mask=False, mask_edges=False)` — raw dense outputs, no edge masking.
- **VGGT inference call:** `model(images[None])`, cameras from `pose_encoding_to_extri_intri`.

## Suggested comparison metrics for other runtimes

Feed `inputs` directly, then report per key: `depth` relative error (median, p90) on pixels with
`depth_conf >= median`, `points` L2 error / `scene_extent_median` from the JSON, camera rotation
error (degrees) and translation error relative to the scene extent, and Pearson r of `depth_conf`.
