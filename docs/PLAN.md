# CodeMesh 3D — Plan and Architecture

Single source of truth for humans and AI agents. Keep it current; record decisions in `docs/DECISIONS.md`.

## Goal

Desktop web app: upload a walk-around (or turntable) video or photo set of one object → game-ready, textured GLB with clean, efficient geometry. Runs fully on the local Windows mini PC (Ryzen 7840HS, Radeon 780M, ~28 GB visible RAM). Hackathon deadline: 2026-11-04 night. Built so it can be grown afterwards (mobile app, commercial use).

## Fixed decisions

| Area | Decision | Why |
|---|---|---|
| Segmentation | SAM 3.1 via Meta API (`sam-3.1`, `https://api.meta.ai/v1`) | Local SAM needs CUDA. |
| Reasoning agent | `muse-spark-1.3-contributor`, reasoning effort high, via Meta API | User choice. Contributor inputs may train Meta models → UI disclosure, no personal content. |
| API key | `.env` → `META_API_KEY` (Meta docs call it `MODEL_API_KEY`; we pass it explicitly). Never sent to the browser. | |
| Geometry model | Multi-view feed-forward model behind `GeometryEngine` interface. Candidates: MapAnything (`facebook/map-anything-apache`, Apache-2.0, commercial-safe) and VGGT-1B (CC-BY-NC, hackathon only). Pick by Phase 0 benchmark. | |
| GPU path | Try in order: ggml + Vulkan (`Asher-1/map-anything-ggml`), PyTorch ROCm (TheRock gfx1103), ONNX DirectML. If none works: PyTorch CPU on the mini PC + PyTorch CUDA for NVIDIA machines. | |
| Backend | Python 3.12, FastAPI, managed with `uv`. Heavy compute lives in native libs (ggml/Vulkan, Open3D, Blender, NumPy). | Python is orchestration only; ecosystem (Open3D, trimesh, xatlas, Blender) is Python-first. |
| Blender | Portable Blender run as a subprocess (`blender -b --python …`), never imported. | Keeps GPL code out of our process and decouples Python versions. |
| Frontend | React + TypeScript + Vite, React Three Fiber (three.js) viewer, plain CSS with variables. | Same code wraps into Tauri later; `@react-three/fiber/native` + Expo can reuse viewer logic. |
| Mobile readiness | `frontend/src/core/` = platform-agnostic TS (API client, types, job state). No DOM/browser APIs there. Progress via polling (works in Expo; no EventSource). | |
| Hosting | Local: backend + frontend on the mini PC. Vercel: static frontend build with `VITE_API_BASE_URL` pointing at a backend, plus a "Showcase" mode with real pre-processed results bundled as static files. Backend cannot run on Vercel (no GPU, no Blender, function time limits). | |
| Job system | One worker process, one heavy job at a time, SQLite (stdlib `sqlite3`) for job state, files under `jobs/<id>/`. Each stage caches its outputs keyed by a hash of inputs + settings, so changing texture settings reruns only texture stages. | No Redis/Celery. |

## Repo layout

```
backend/
  pyproject.toml            uv project (Python 3.12)
  codemesh/
    config.py               settings from env (.env at repo root)
    api/                    FastAPI routes (thin; no pipeline logic)
    jobs/                   job store (SQLite), runner, stage cache
    pipeline/               one module per stage: ingest, segment, geometry, fuse, mesh, texture, export, evaluate, refine
    engines/                GeometryEngine implementations (base.py = interface)
    meta/                   Meta API clients: sam.py, muse.py
    agent/                  Muse tool API (Phase 3)
  tests/
frontend/                   React + Vite app
  src/core/                 platform-agnostic: api client, types, job state
  src/features/             upload, select, progress, viewer, settings, showcase
bench/                      benchmark scripts + results (bench/<phase>/RESULTS.md)
docs/                       PLAN.md, DECISIONS.md, ARCHITECTURE notes
third_party/                cloned repos, builds (gitignored)
models/                     weights (gitignored)
data/captures/              test videos/photos (gitignored)
jobs/                       job working dirs (gitignored)
```

## Pipeline (core)

1. **ingest** — decode video with PyAV, score sharpness (variance of Laplacian), pick frames spread over time (preset: 24 / 40 / 64), drop near-duplicates. Keep full-resolution originals; never downscale saved frames.
2. **segment** — SAM 3.1 tracks the user-selected object (click points and/or text) through the video; full-res masks per selected frame. Photos: per-image text/point prompts.
3. **geometry** — engine infers cameras, depth, confidence, points on full frames (walk-around) or background-blacked frames (turntable mode). Originals stay full-res; engine resolution is whatever the model wants (~518 px).
4. **fuse** — unproject depth inside eroded masks with confidence filtering; colors from full-res frames; outlier removal; normals oriented to cameras.
5. **mesh** — dense surface (screened Poisson or TSDF, trimmed by density and by mask visual-hull consistency) = high-res source; then retopology/decimation to an object-dependent triangle budget (QuadriFlow / decimate in Blender, meshoptimizer), UVs (xatlas).
6. **texture** — project full-res photos onto the UV atlas (visibility + view-angle weighted blending, seam fill), 2K/4K; bake normal map from high-res surface to low-poly (Blender).
7. **export** — GLB (PBR metallic-roughness, base color + normal), optional real-world scale from one user-given dimension.
8. **evaluate** — held-out views: silhouette IoU vs SAM masks at full res, masked PSNR/SSIM; report in job metrics.
9. **refine** (Phase 3) — silhouette-distance + point-distance vertex optimization (CPU, no differentiable renderer needed); Muse reviews renders vs references and picks high-level tool actions within a time/API budget.

## API contract (v1, JSON, base `/api`)

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | `{ok, version}` |
| GET | `/capabilities` | `{geometry_engine, device, precision, blender: bool, meta_api: bool}` |
| POST | `/jobs` | multipart: `files[]` (one video or many images), `mode` = `walkaround`\|`turntable` → `Job` |
| GET | `/jobs` | `Job[]` (newest first) |
| GET | `/jobs/{id}` | `Job` (poll ~1 s for progress) |
| GET | `/jobs/{id}/frames` | `[{index, url, thumb_url, sharpness}]` |
| POST | `/jobs/{id}/select` | `{frame_index, points:[{x,y,label:1\|0}] (normalized 0..1), text?}` → starts segmentation |
| POST | `/jobs/{id}/reconstruct` | `{preset: draft\|standard\|high, triangle_budget: "auto"\|int, texture_size: 2048\|4096, known_dimension?: {axis: x\|y\|z, meters}}` |
| POST | `/jobs/{id}/refine` | `{instruction?: string}` → Muse-assisted refinement (Phase 3) |
| GET | `/jobs/{id}/artifacts/{path}` | files: masks, previews, `model.glb` |
| DELETE | `/jobs/{id}` | delete job + files |

`Job` = `{id, created_at, mode, status: created|ready_to_select|segmenting|ready_to_reconstruct|reconstructing|refining|done|failed, stage, progress: 0..1, message, stages: [{name, status, seconds, metrics}], artifacts: {model_glb?, masks_preview?, ...}, metrics: {...}, error?}`

## Phases (tonight)

| Phase | Content | User test |
|---|---|---|
| 0 | Geometry inference on the 780M (ggml Vulkan / ROCm / CPU), Meta API clients (SAM 3.1, Muse), backend + frontend skeletons on the contract | Benchmark table; app shell runs locally |
| 1 | Core pipeline 1–8 as CLI + job runner, on the user's two captures | Open the GLBs in Blender / viewer |
| 2 | Web app end to end: upload → select → mask preview → reconstruct → progress → viewer → export | Full flow in browser |
| 3 | Muse analysis + refine loop, silhouette optimization, normal maps, presets | Compare before/after |
| 4 | Vercel deploy (frontend + showcase), README, disclosure, polish | Hosted URL |

Later: LODs, collision meshes, sandboxed Muse code tool, Tauri desktop build, Expo capture app, metric scale from MapAnything.

## Conventions

- Small modules, one responsibility, type hints, docstrings on public functions. No pipeline logic in API routes.
- Every stage is a pure-ish function `run(ctx, inputs, settings) -> outputs` writing into its own folder in the job dir.
- Coordinates: OpenCV camera convention (x right, y down, z forward), extrinsics world→camera 4×4, meters when scale is known.
- Tests: `pytest` in `backend/tests`; run via `uv run pytest`. Frontend: `npm run build` + `npm run lint` must pass.
- Never commit weights, captures, jobs, `.env`.
