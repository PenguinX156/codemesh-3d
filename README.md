# CodeMesh 3D

Turn an ordinary phone video of an object into a textured, game-ready 3D asset (GLB).

Upload a walk-around video (or a set of photos), type what the object is, and CodeMesh finds it in every frame with **SAM 3.1**, recovers the cameras and depth with **MapAnything** (Meta, Apache-2.0 weights), builds a clean mesh, projects the original 4K photography onto it, bakes a normal map, and checks its own work by re-rendering the model from the real camera positions and comparing against your footage.

Built for the Meta Global AI Developer Hackathon (Cerebral Valley, Nov 2026).

## How it works

| Step | What happens | Tech |
|---|---|---|
| Ingest | Picks the sharpest, non-redundant frames from the video at full resolution (handles iPhone HDR/HEVC and rotation) | PyAV, OpenCV |
| Segment | One text prompt or click → the object is tracked through every frame (one video request, not per image) | **SAM 3.1** (Meta Model API) |
| Geometry | Camera poses, intrinsics, depth and confidence from all views at once | **MapAnything** (Meta) on Modal L4 GPU, AMD Radeon 780M via ROCm, or CPU |
| Fuse + mesh | Masked, confidence-filtered point cloud → Poisson surface → silhouette + free-space carving → error-bounded decimation to an object-dependent triangle budget | Open3D, trimesh |
| Texture | UV atlas, multi-view photo projection with visibility, view-angle weighting and exposure balancing; tangent-space normal map baked from the dense surface | xatlas, Open3D ray casting |
| Export | GLB with PBR material (albedo + normal), +Y up, meters when a real dimension is given | glTF 2.0 |
| Evaluate | Re-renders the model from every source camera: silhouette IoU vs SAM masks, PSNR/SSIM vs photos, side-by-side compare images | Open3D, NumPy |

Measured detail (from the reconstruction) is kept separate from estimated detail (unseen areas filled by the surface reconstruction); the UI says which is which.

## Run locally (Windows)

Requirements: Python via [uv](https://docs.astral.sh/uv/), Node 20+, a Meta Model API key.

```powershell
copy .env.example .env   # then set META_API_KEY
uv sync --project backend --extra torch
cd frontend; npm install; cd ..
.\scripts\demo_backend.ps1    # API on http://localhost:8010
.\scripts\demo_frontend.ps1   # UI  on http://localhost:5180
```

Geometry engine (`CODEMESH_GEOMETRY_ENGINE`): `auto` (default), `rocm` (AMD GPU via PyTorch ROCm, see `bench/rocm/ROCM.md`), `modal` (cloud GPU), `torch` (CPU / CUDA).

## Hosted (Modal)

`deploy/modal_app.py` serves the UI and API from one URL, runs MapAnything on an L4 GPU that scales to zero, and keeps weights and jobs on Modal volumes. See `deploy/README.md`.

## Capture tips

See `docs/CAPTURE_GUIDE.md`: even indoor light, 1× lens, 30–45 s slow loop around the object plus a short look from above. A single quick loop also works; unseen areas are estimated.

## Repository layout

`backend/codemesh/` (FastAPI app, job runner, pipeline stages, engines, Meta API clients) · `frontend/` (React + TypeScript + Vite + React Three Fiber; `src/core/` is platform-agnostic for a future Expo app) · `deploy/` (Modal) · `bench/` (benchmarks and their results) · `docs/` (plan, capture guide).

## Licenses and data use

- MapAnything weights: `facebook/map-anything-apache` (Apache-2.0). VGGT-1B was benchmarked only (CC-BY-NC).
- SAM 3.1 and Muse Spark are used through the Meta Model API. **The Muse Spark Contributor tier may use inputs to train Meta models — do not upload personal or sensitive content.**
- Code license: not chosen yet.
