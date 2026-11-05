# CodeMesh 3D

Turn an ordinary phone video of an object into a textured, game-ready 3D asset (GLB).

Upload a walk-around video (or a set of photos), type what the object is, and CodeMesh finds it in every frame with **SAM 3.1**, recovers the cameras and depth with **MapAnything** (Meta, Apache-2.0 weights), builds a clean mesh, projects the original 4K photography onto it, bakes a normal map, and checks its own work by re-rendering the model from the real camera positions and comparing against your footage.

Built for the Meta Global AI Developer Hackathon (Cerebral Valley, Nov 2026).

## Try it live (no install)

**https://fjziemski--codemesh-web.us-east.modal.run**

1. Open the link in any desktop or phone browser. The first visit after it has been idle takes a few seconds to wake up ("Connecting to the backend…").
2. Drop in a short walk-around video of an object (or 2-300 photos), tick the data-use notice, and press **Upload and start**.
3. Type what the object is (for example "electric kettle"), press **Find object**, check the masks, then **Build 3D model**.
4. In a few minutes you get the 3D viewer, the accuracy score, the quality grade and Muse's modeling decisions. **Download GLB** to open it in Blender, Unity or Unreal.

Tips: walk one slow circle (30-45 s) with the whole object in frame, then add a short look from above. Don't upload people, faces or private spaces: inputs go to Meta's Muse Spark Contributor tier. Finished jobs are listed under Recent jobs on the home page.

## The idea

Making a 3D asset of a real object still means a scanning rig, or photogrammetry followed by hours of cleanup: scans come out as lumpy, holey, million-triangle meshes that no game engine wants. CodeMesh 3D treats a phone video as **evidence** rather than as the model. Meta's models extract what is in the footage (which pixels are the object, where every camera was, how far away each surface is), and **Muse Spark acts like a 3D artist**: it looks at the photos, decides what the object is made of (a lathe-turned body, a box, a handle), builds a clean, simple mesh with modeling tools, and checks every change against the real footage. The goal is a model that looks made, not scanned: one watertight piece, sensible topology, UVs, a texture and a normal map.

## How it works

| Step | What happens | Tech |
|---|---|---|
| Ingest | Picks the sharpest, non-redundant frames from the video at full resolution (handles iPhone HDR/HEVC and rotation) | PyAV, OpenCV |
| Segment | One text prompt or click → the object is tracked through every frame (one video request, not per image) | **SAM 3.1** (Meta Model API) |
| Geometry | Camera poses, intrinsics, depth and confidence from all views at once | **MapAnything** (Meta) on Modal L4 GPU, AMD Radeon 780M via ROCm, or CPU |
| Fuse + mesh | Visual hull carved from the full-resolution SAM silhouettes, refined with depth (free-space carving opens cavities such as a mug's inside), repaired to one watertight piece, decimated to an object-dependent triangle budget. Poisson surface is the alternative | NumPy, Open3D, trimesh |
| Texture | UV atlas, multi-view photo projection with visibility, view-angle weighting and exposure balancing; tangent-space normal map baked from the dense surface | xatlas, Open3D ray casting |
| Export | GLB with PBR material (albedo + normal), +Y up, meters when a real dimension is given | glTF 2.0 |
| Evaluate | Re-renders the model from every source camera: silhouette IoU vs SAM masks, PSNR/SSIM vs photos, side-by-side compare images | Open3D, NumPy |
| Model (agent) | Muse Spark reviews reference frames, masks and render-vs-photo comparisons, then calls modeling tools: `build_lathe` (bodies of revolution), `build_rounded_box`, `add_handle` (swept tube), `use_reconstruction`, `set_mesh_method`, `set_triangle_budget`, `apply_symmetry`, material and texture settings. Each call is executed, re-textured and re-measured; the best version is kept and a clean primitive wins over a messy scan when it scores about as well | **Muse Spark 1.3** (Meta Model API), reasoning high |
| Quality gate | Checks game-readiness: single connected piece, watertight, manifold, no degenerate faces, triangle count, UVs and textures, silhouette accuracy. Gives a grade A-F and concrete recapture tips | trimesh |

Measured detail (from the reconstruction) is kept separate from estimated detail (unseen areas filled by the surface reconstruction); the UI says which is which.

## Architecture

```
Browser (React + TypeScript + React Three Fiber)
   │  upload video ─ select object ─ review masks ─ 3D result + GLB download
   ▼
FastAPI  (/api: jobs, frames, select, reconstruct, refine, artifacts)
   │  SQLite job store · thread runner · per-stage cache (a rerun skips unchanged stages)
   ▼
Pipeline stages: ingest → segment → geometry → fuse → mesh → texture → export → evaluate → agent → quality
   │                 │           │                                                    │
   │                 │           └─ GeometryEngine: Modal GPU worker (L4/A10G/T4)       │
   │                 │              · AMD ROCm worker · CPU/CUDA torch                  │
   │                 └─ Meta Model API: SAM 3.1 (one video tracking request)            │
   └──────────────────────────────── Meta Model API: Muse Spark 1.3 (tool calls) ───────┘
```

On Modal (`deploy/modal_app.py`) one web function serves both the built UI and the API at a single public URL; MapAnything runs in a separate GPU class with a memory snapshot and scales to zero; model weights and job files live on Modal volumes. Locally the same code runs with the AMD laptop GPU (ROCm) or the CPU.

## How it uses Meta models

- **SAM 3.1** (Meta Model API): text-prompted video segmentation. The app builds a short clip from only the frames it keeps and sends one tracking request (about 25x cheaper than sending the whole video). The full-resolution masks drive everything after: the shape carving, texture visibility and the accuracy score.
- **MapAnything** (Meta open weights, `facebook/map-anything-apache`, Apache-2.0): camera poses, intrinsics and depth for all views in one feed-forward pass, on a Modal GPU or on an AMD Radeon 780M through PyTorch ROCm.
- **Muse Spark 1.3 Contributor** (Meta Model API, reasoning high, images at up to 2048 px): the modeling agent described above. It classifies the object, chooses how to build it, calls the modeling tools, and its decisions and rationale are shown in the app. Typical cost is well under a cent per job.
- In development, Muse Spark was also used as a helper for two extensions that are built and unit-tested in this repo but not yet switched on in the live pipeline: **frame review** (Muse rates every frame and keeps blurry or unusual ones as references instead of dropping them blindly, `backend/codemesh/agent/frames.py`) and **procedural mesh code** (Muse writes mesh-building code that runs in a restricted sandbox, `backend/codemesh/agent/codegen.py`; on the mug it produced a 704-triangle watertight model at IoU 0.81).

## How it was built

Built during the hackathon with **Claude Code running Claude Opus 5.5** as the lead engineer: it planned the architecture, wrote and integrated the pipeline, and orchestrated parallel subagents (Claude Sonnet for scoped coding, UI polish and object test runs, Claude Haiku for code search) with a review round per phase. **Muse Spark 1.3** was used both inside the product (the modeling agent) and as a helper during development (frame review and mesh-code experiments in `bench/muse_tools/`). OpenAI Codex served as an independent third-party QA reviewer of the results. Every stage is covered by tests (`backend/tests`, 120+ tests).

## Current results

| Object | Silhouette IoU | Notes |
|---|---|---|
| Kettle (shiny steel) | 0.89 | Lathe body + handle; logo and water window visible in the texture |
| Mug | 0.81 | Clean lathe body + swept handle, about 13k triangles, watertight |
| Water-filter pitcher (transparent) | 0.82 | Recognisable shape; lid and spout not yet modeled as parts |
| Cabinet (large, fills the frame) | 0.20 | Fails today: needs more views and box-with-drawers tools |

Known limits and next steps: spout and lid tools, de-lit textures (highlights and colour cast are currently baked in), metric scale from the table plane, running the mesh stage on the GPU (it is the slowest step on Modal), and switching on Muse frame review and code modeling in the live pipeline.

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

`deploy/modal_app.py` serves the UI and API from one URL, runs MapAnything on a GPU (L4, falling back to A10G or T4) that scales to zero, and keeps weights and jobs on Modal volumes. See `deploy/README.md`.

## Capture tips

See `docs/CAPTURE_GUIDE.md`: even indoor light, 1× lens, 30–45 s slow loop around the object plus a short look from above. A single quick loop also works; unseen areas are estimated.

## Repository layout

`backend/codemesh/` (FastAPI app, job runner, pipeline stages, engines, Meta API clients) · `frontend/` (React + TypeScript + Vite + React Three Fiber; `src/core/` is platform-agnostic for a future Expo app) · `deploy/` (Modal) · `bench/` (benchmarks and their results) · `docs/` (plan, capture guide).

## Licenses and data use

- MapAnything weights: `facebook/map-anything-apache` (Apache-2.0). VGGT-1B was benchmarked only (CC-BY-NC).
- SAM 3.1 and Muse Spark are used through the Meta Model API. **The Muse Spark Contributor tier may use inputs to train Meta models — do not upload personal or sensitive content.**
- Code license: not chosen yet.
