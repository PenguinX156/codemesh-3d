# Hackathon submission (Meta Global AI Developer Hackathon, Cerebral Valley)

Deadline: Nov 4, 2026, 9:00 PM ET (22:00 machine-local).
Form: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit (you submit it yourself).

## Suggested choices
- Track: **Indie Developer Track** (solo; judged on creativity, technical ambition, whether it works)
- Problem statement: **Push the Frontier** (gives a machine a new capability: turning ordinary video into usable 3D objects)
- Public GitHub repository: (fill after pushing)
- 3-minute demo video: (fill after uploading, e.g. YouTube unlisted)
- Live app: https://fjziemski--codemesh-web.us-east.modal.run

## Project Description (≤ 3,000 chars)

CodeMesh 3D turns an ordinary phone video of an object into a textured, game-ready 3D asset (GLB) that drops into Blender, Unity, Unreal, Godot or a video editor.

You upload a casual walk-around video (or a set of photos), type what the object is (for example "black coffee mug"), and the pipeline does the rest automatically:

1. Ingest picks the sharpest, non-redundant frames from the full-resolution video (iPhone 4K HEVC/HDR and rotation handled).
2. SAM 3.1 tracks the object through all frames from that one text prompt, in a single video request.
3. MapAnything (Meta's open-weights multi-view model, Apache-2.0) estimates camera poses, intrinsics and depth for all views at once. It runs on a Modal L4 GPU in the cloud, or locally on an AMD Radeon 780M through PyTorch ROCm (about 9x faster than CPU, accuracy within 0.07 degrees of the fp32 reference), with a CPU fallback.
4. A camera-consistency filter rejects views whose estimated lens disagrees with the rest of the capture.
5. The shape is carved from the full-resolution SAM silhouettes (visual hull) and refined with depth (free-space carving recovers openings such as the inside of a mug), then decimated to an error-bounded, object-dependent triangle budget.
6. The original 4K photos are projected into a UV atlas with visibility and view-angle weighting, and a tangent-space normal map is baked from the dense surface.
7. The model is re-rendered from every real camera and scored against the footage (silhouette IoU, boundary F-score, PSNR/SSIM, chamfer), with side-by-side comparison images.
8. Muse Spark 1.3 acts as the modeling agent: it looks at the reference frames, masks and the comparison renders, classifies the object (category, hard-surface vs organic, symmetry, thin parts, materials) and calls restricted modeling tools (meshing method, triangle budget, symmetry, material, texture size). The pipeline executes each call, re-measures, and keeps the best version, so it never ships a regression.
9. A quality gate checks game-readiness (single connected piece, watertight, manifold, no degenerate faces, UVs and textures present, silhouette accuracy) and gives the user a grade and concrete recapture tips instead of a silent bad result.

Real result: on a 45-second iPhone video of a mug, naive depth fusion produced a torn fragment (silhouette IoU 0.28). Muse, shown the comparison renders, diagnosed sparse depth on the small dark object and switched to mask-based meshing on its own (0.69); adding the camera-consistency filter raised it to 0.82. An independent review (OpenAI Codex as third-party QA) found what the score hides (thin handle, texture color cast, open boundaries), which drove the quality gate. Cost per job: about $0.01 of SAM and $0.002 of Muse.

Stack: FastAPI + Python job runner, React + TypeScript + React Three Fiber (platform-agnostic core for a future Expo app), Open3D, trimesh, xatlas, Modal (one URL serves UI + API, GPU scales to zero).

## Describe your use of Meta products (≤ 3,000 chars)

- SAM 3.1 (Meta Model API): text-prompted video segmentation and tracking. The app builds a short clip from only the frames it keeps (padded to the 50-frame minimum) and sends one tracking request, which is about 25x cheaper than sending the whole video and far cheaper than per-image calls. The full-resolution masks drive everything downstream: point filtering, the visual hull that defines the shape, texture visibility and the accuracy score.
- Muse Spark 1.3 Contributor (Meta Model API), reasoning high: the modeling agent. It receives reference frames, mask cut-outs and render-vs-photo comparison images with metrics, returns a structured analysis, and calls typed tools (set_mesh_method, set_triangle_budget, apply_symmetry, set_material, set_texture_size, accept). Conventional geometry code executes each call; the loop re-measures and keeps the best result. Every decision and its rationale is logged in an agent report shown in the app. Typical cost: about $0.002 per job.
- MapAnything (Meta open weights, facebook/map-anything-apache): multi-view camera and depth estimation, deployed on Modal L4 (bf16, about 4 s for 8 views) and on an AMD Radeon 780M via PyTorch ROCm. VGGT-1B was also benchmarked.

## Why are Meta's models important to your solution? (≤ 3,000 chars)

Turning a casual video into a usable 3D asset needs three things that used to require separate specialist tools or manual work: knowing which pixels belong to the object, knowing where every camera was, and deciding how to model what was seen.

SAM 3.1 makes the first one a single sentence. Because it tracks the object through the whole video from a text prompt, the user never draws masks frame by frame, and its full-resolution silhouettes are precise enough to carve the shape directly. That is what makes small objects and imperfect captures work, where depth alone produced a torn fragment.

MapAnything replaces a slow, fragile structure-from-motion step with one feed-forward pass that gives cameras and depth for all views, and its Apache-2.0 weights mean the same model can run in the cloud and locally on a consumer AMD laptop GPU.

Muse Spark closes the loop. Geometry algorithms are good at executing a plan but bad at choosing one: a mug, a rock and a chair need different methods, budgets and symmetry assumptions. Muse looks at the evidence and the measured errors, picks the strategy through restricted tools, and the system verifies every change numerically. Its low cost per call is what makes running this agent on every job practical.

## Feedback on partner technologies (≤ 150 chars)

Modal made GPU hosting one Python file; snapshots + scale-to-zero fit a hackathon budget. Image needed extra libs for Open3D.
