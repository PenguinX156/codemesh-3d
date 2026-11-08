# CodeMesh 3D on Modal

`deploy/modal_app.py` defines the Modal app `codemesh`:

| Object | What | Resources |
|---|---|---|
| `web` | FastAPI app (`codemesh.main.create_app`) + `frontend/dist` at `/`: one URL for UI and API | 8 CPU, 16 GiB, `max_containers=1`, `min_containers=0`, scales down after 600 s idle |
| `GeometryWorker` | `TorchEngine("mapanything")`: weights load to CPU at snapshot time (`@modal.enter(snap=True)`), move to CUDA on every start (`snap=False`); `infer` + cheap `warmup` methods | L4 GPU, 16 GiB, `enable_memory_snapshot=True`, scales down after 120 s idle, 900 s timeout |
| `download_weights` | snapshots `facebook/map-anything-apache` into the `codemesh-models` volume, warms the DINOv2 torch hub cache | 4 CPU, run once |

Volumes: `codemesh-models` at `/models` (weights, torch hub, HF cache), `codemesh-jobs` at `/jobs` (SQLite + job files).
The backend selects `ModalEngine` (`backend/codemesh/engines/modal_engine.py`) when `CODEMESH_GEOMETRY_ENGINE=modal`, which the web container sets.

All commands run from the repo root. `modal run` of `bench` imports `codemesh` locally, so run it through the backend venv with `--with modal`.

## 1. Secret (Meta API key)

The key must never appear on a command line or in logs. This writes a temp dotenv holding only `META_API_KEY`, creates the secret from it, and deletes it:

```bash
uv run --project backend --with modal python - <<'EOF'
import os, subprocess, tempfile
from dotenv import dotenv_values
key = dotenv_values(".env")["META_API_KEY"]
fd, path = tempfile.mkstemp(suffix=".env")
try:
    with os.fdopen(fd, "w") as f:
        f.write(f"META_API_KEY={key}\n")
    subprocess.run(["modal", "secret", "create", "codemesh-meta", "--from-dotenv", path, "--force"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
finally:
    os.remove(path)
print("secret codemesh-meta created")
EOF
```

## 2. Weights (once, idempotent)

```bash
modal run deploy/modal_app.py::download_weights
```

## 3. GPU smoke test (optional, ~2 GPU calls)

```bash
uv run --project backend --with modal modal run deploy/modal_app.py::bench
```

Prints the GPU name, output shapes, cold-start and warm-call seconds, and the rotation / depth AbsRel error against `bench/phase0/out/ref_mapanything_kitchen_8.npz` (CPU reference).

## 4. Deploy / redeploy

```bash
(cd frontend && npm run build)        # optional; without dist/ only the API is served
modal deploy deploy/modal_app.py
curl https://<workspace>--codemesh-web.modal.run/api/health
```

Redeploying after code changes only rebuilds the thin source / UI layers. Stop everything with `modal app stop codemesh`.

## Cold start

Measured before memory snapshots (`modal run deploy/modal_app.py::bench`, L4 bf16): model load 47 s, inference 4.1 s for 8 views, first call ever 347 s including the image build.
With `enable_memory_snapshot=True` the first container after each deploy loads the weights and writes the snapshot; later cold starts restore CPU memory and only pay the CPU->GPU copy (`warmup()` returns `to_gpu_seconds`). Not yet measured after the snapshot change: the redeploy was not run.

Measure (two calls spaced more than `scaledown_window` apart so the second one is a restore):

```bash
uv run --project backend --with modal python -c "import modal,time; w=modal.Cls.from_name('codemesh','GeometryWorker')(); t=time.time(); print(w.warmup.remote(), round(time.time()-t,1))"
```

`POST /api/warmup` on the web app spawns one 8-view `infer` call (`codemesh.engines.factory.warmup`), so the GPU container is up by the time segmentation finishes.

## Costs (Modal list prices, check modal.com/pricing)

- L4: ~$0.80/h. A geometry call costs cold start (~1-2 min: container + model load) + inference + 120 s idle tail: roughly **$0.03-0.05 per job**.
- Web container: 8 CPU + 16 GiB ~ $0.50/h while up; a job keeps it up for its runtime + 600 s idle tail: roughly **$0.10-0.20 per job**.
- Idle: $0 (both scale to zero). Volumes: storage only, a few GB.
- Estimates, not measured; set a workspace spending limit in the Modal dashboard.

## Checking that it scales to zero

The web container stays up while requests keep arriving, then stops 600 s after the last one (`scaledown_window`); the GPU worker stops 120 s after its last call. `min_containers=0`, so nothing runs when nobody uses the app. Until Nov 7 2026 the page re-checked `/api/health` every 10 s forever, so any open tab kept the 8-CPU container up around the clock (~$12/day). Now an open page only makes requests while a job is being processed and its tab is visible: it stops once the job is done, failed or waiting for the user, and hidden tabs make no requests.

1. **Page behaviour** (no real job is started; about 1.5 min):
   ```bash
   node bench/ui/visibility_check.mjs
   ```
   All three lines must say PASS (running job + hidden tab, waiting job, finished job).
2. **Nothing running when idle:** leave the app alone for 11 minutes (close or reload any tab opened before the fix: old tabs keep the old code), then
   ```bash
   modal container list
   ```
   prints an empty table (or `[]` with `--json`). The dashboard shows the same under Apps → codemesh → Containers.
3. **Who kept it awake** (if a container is still up):
   ```bash
   modal app logs codemesh --timestamps --since 2026-11-08T00:00:00Z
   ```
   A steady stream of `GET /api/health` lines means an old tab is polling; `GET /api/jobs/<id>` every second means a job is running.
4. **Cost per hour:**
   ```bash
   modal billing report --for today --resolution h --show-resources
   ```
   Idle hours show no CPU or memory cost for `codemesh`.
5. **Hard stop** when the app is no longer needed: `modal app stop codemesh` (the URL stops working; `modal deploy deploy/modal_app.py` brings it back). Also set a workspace spending limit under Modal Settings → Usage & Billing.

## Limits

- One web container (`max_containers=1`): SQLite and the in-process runner thread are not shared across containers. Jobs run one at a time, FIFO. To scale out, move the job store to a shared DB and the runner to separate Modal functions.
- SQLite lives on a Modal Volume: fine for one writer; data written in the last seconds before a container is killed may be lost. Jobs that were running when the container stopped are marked failed on restart (`fail_interrupted`).
- Upload limit 600 MB (`CODEMESH_MAX_UPLOAD_BYTES`). Modal web endpoints also cap request time; very large uploads on slow links may time out.
- No queue length limit in the backend today; a busy queue just waits.
- The cold UI load after idle takes a few seconds (web container start).

## Switching GPU type

Change `GPU = "L4"` in `deploy/modal_app.py` (e.g. `"A10G"`, `"L40S"`, `"A100"`), then `modal deploy deploy/modal_app.py`. bf16 is used on Ampere and newer (L4, A10G, A100); fp16 on older cards.
