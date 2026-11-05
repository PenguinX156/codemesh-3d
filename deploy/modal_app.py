"""CodeMesh 3D on Modal: one URL serving UI + API, geometry on an L4 GPU that scales to zero.

    modal run deploy/modal_app.py::download_weights   # once: weights + DINOv2 hub code into the volume
    modal run deploy/modal_app.py::bench               # GPU smoke test on 8 kitchen frames
    modal deploy deploy/modal_app.py                   # public web endpoint

See deploy/README.md for costs and limits.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MA_DIR = ROOT / "third_party" / "map-anything"
DIST = ROOT / "frontend" / "dist"
GPU = "L4"
MAX_UPLOAD_BYTES = 600 * 1024**2

app = modal.App("codemesh")
models_vol = modal.Volume.from_name("codemesh-models", create_if_missing=True)
jobs_vol = modal.Volume.from_name("codemesh-jobs", create_if_missing=True)

ENV = {
    "CODEMESH_MODELS_DIR": "/models",
    "CODEMESH_JOBS_DIR": "/jobs",
    "CODEMESH_GEOMETRY_ENGINE": "modal",
    "CODEMESH_MAX_UPLOAD_BYTES": str(MAX_UPLOAD_BYTES),
    "TORCH_HOME": "/models/torch_hub",  # DINOv2 hub code is fetched once into the volume
    "HF_HOME": "/models/hf",
    "PYTHONPATH": "/app/backend",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libegl1", "libglib2.0-0", "libgomp1", "libusb-1.0-0", "libx11-6", "libxext6", "libxrender1", "libsm6", "git")  # open3d links these even headless
    # PyPI Linux wheels of torch bundle CUDA; versions match the local CPU reference venv.
    .pip_install("torch==2.15.0", "torchvision==0.29.2")
    .pip_install(
        "fastapi>=0.115", "uvicorn[standard]>=0.30", "python-multipart>=0.0.9", "pydantic>=2.7",
        "httpx>=0.27", "openai>=1.50", "python-dotenv>=1.0", "numpy>=1.26,<2.3", "scipy>=1.13",
        "av>=12", "pillow>=10.4", "open3d>=0.19", "trimesh>=4.4", "xatlas>=0.0.9", "huggingface-hub>=0.25",
        "meta-sam-parser==0.1.0",  # SAM 3.1 mask decoding (codemesh.meta.sam)
        # MapAnything runtime deps (third_party/map-anything/pyproject.toml at 3d10cf7)
        "hydra-core", "natsort", "opencv-python-headless==4.10.0.84", "orjson", "pillow-heif", "plyfile",
        "python-box", "requests", "rerun-sdk~=0.24.1", "safetensors", "tensorboard", "tqdm", "uniception==0.1.7",
    )
    .add_local_dir(MA_DIR, "/opt/map-anything", copy=True, ignore=[".git", "**/__pycache__", "*.ipynb"])
    .run_commands("pip install --no-deps /opt/map-anything")
    .env(ENV)
)
# Source and UI last so code edits rebuild only these thin layers.
image = image.add_local_dir(ROOT / "backend" / "codemesh", "/app/backend/codemesh", ignore=["**/__pycache__"])
if DIST.exists():
    image = image.add_local_dir(DIST, "/app/dist")


@app.function(image=image, volumes={"/models": models_vol}, cpu=4.0, memory=16384, timeout=1800)
def download_weights() -> None:
    """Idempotent: snapshot the weights and warm the torch hub cache by loading the model on CPU."""
    from huggingface_hub import snapshot_download

    target = Path("/models/map-anything-apache")
    if not (target / "config.json").exists():
        snapshot_download("facebook/map-anything-apache", local_dir=target)
    if not Path("/models/torch_hub/hub").exists():
        from codemesh.engines.torch_engine import TorchEngine

        TorchEngine("mapanything", device="cpu").close()
    models_vol.commit()
    print(sorted(p.name for p in target.iterdir()))


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": models_vol},
    memory=16384,
    scaledown_window=120,
    timeout=900,
    enable_memory_snapshot=True,  # CPU memory snapshot: weights load once, restores skip the volume read
)
class GeometryWorker:
    @modal.enter(snap=True)
    def load(self) -> None:
        """Runs once at snapshot time (no GPU attached): model weights into CPU memory."""
        from codemesh.engines.torch_engine import TorchEngine

        t0 = time.perf_counter()
        self.engine = TorchEngine("mapanything", device="cpu")
        self.snapshot_load_seconds = round(time.perf_counter() - t0, 2)

    @modal.enter(snap=False)
    def to_gpu(self) -> None:
        """Runs on every container start, including restores from the snapshot."""
        import torch

        from codemesh.engines.torch_engine import default_precision

        t0 = time.perf_counter()
        self.engine.device = torch.device("cuda")
        self.engine.precision = default_precision("cuda")
        self.engine.model = self.engine.model.to(self.engine.device)
        self.gpu = torch.cuda.get_device_name(0)
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.started = time.time()

    @modal.method()
    def warmup(self) -> dict:
        """Cheap no-op call that only forces a container up; reports how long the GPU move took."""
        return {"gpu": self.gpu, "to_gpu_seconds": self.load_seconds,
                "snapshot_load_seconds": self.snapshot_load_seconds, "container_age_s": round(time.time() - self.started, 1)}

    @modal.method()
    def infer(self, images: list[bytes], masks: list[bytes] | None = None) -> bytes:
        """JPEG frames (+ optional PNG masks) in, npz of ViewGeometry out (codec in modal_engine.py)."""
        from codemesh.engines.modal_engine import decode_image, decode_mask, geometry_to_npz

        geo = self.engine.infer([decode_image(b) for b in images], [decode_mask(b) for b in masks] if masks else None)
        geo.info = {**geo.info, "gpu": self.gpu, "worker_load_seconds": self.load_seconds}
        return geometry_to_npz(geo)


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={"/jobs": jobs_vol, "/models": models_vol},
    secrets=[modal.Secret.from_name("codemesh-meta")],
    min_containers=0,
    max_containers=1,  # shortcut: SQLite + in-process runner thread need one container; move to a shared DB to scale out
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    from fastapi.staticfiles import StaticFiles

    from codemesh.main import create_app

    api = create_app(max_upload_bytes=MAX_UPLOAD_BYTES)
    if Path("/app/dist").exists():
        api.mount("/", StaticFiles(directory="/app/dist", html=True), name="ui")  # after /api routes
    return api


@app.local_entrypoint()
def bench(n: int = 8) -> None:
    """Run GeometryWorker on n kitchen frames; compare with the CPU reference npz if present."""
    import json

    import numpy as np

    from codemesh.engines.modal_engine import encode_image, geometry_from_npz

    from PIL import Image

    img_dir = ROOT / "third_party" / "vggt" / "examples" / "kitchen" / "images"
    runs = json.loads((ROOT / "bench" / "phase0" / "out" / "reference_results.json").read_text())
    ref_run = next((r for r in runs if r["model"] == "mapanything" and r["set"] == "kitchen" and r["n"] == n), None)
    # Use the reference's exact frames, otherwise the accuracy comparison is meaningless.
    paths = [img_dir / f for f in ref_run["frames"]] if ref_run else sorted(img_dir.iterdir())[:n]
    frames = [encode_image(np.asarray(Image.open(p).convert("RGB"))) for p in paths]
    worker = GeometryWorker()
    t0 = time.perf_counter()
    geo = geometry_from_npz(worker.infer.remote(frames))
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    geometry_from_npz(worker.infer.remote(frames))
    warm = time.perf_counter() - t0
    print(json.dumps(geo.info, indent=1))
    print("shapes", {k: getattr(geo, k).shape for k in ("intrinsics", "extrinsics", "depth", "depth_conf", "points")})
    print(f"first call {first:.1f}s (cold start ~{first - warm:.1f}s), warm call {warm:.1f}s")

    ref_path = ROOT / "bench" / "phase0" / "out" / f"ref_mapanything_kitchen_{n}.npz"
    if ref_path.exists():
        ref = np.load(ref_path)
        # Relative rotation of each camera to camera 0, so the comparison ignores the world frame.
        def rel(e: np.ndarray) -> np.ndarray:
            return e[:, :3, :3] @ e[0, :3, :3].T

        r = np.einsum("nij,nkj->nik", rel(geo.extrinsics), rel(ref["extrinsics"]))
        rot_deg = np.degrees(np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        valid = ref["depth"] > 0
        absrel = np.abs(geo.depth[valid] - ref["depth"][valid]) / ref["depth"][valid]
        print(f"vs CPU reference: rotation err max {rot_deg.max():.3f} deg, depth AbsRel {absrel.mean():.4f}")
