"""PyTorch reference engine for MapAnything and VGGT (CPU now, CUDA by switching `device`).

Uses the official model code from third_party/ (installed editable into the backend venv)
and local weights under models/. Inputs are preprocessed exactly like the official loaders
(see preprocess.py); outputs follow the `ViewGeometry` contract in base.py.

    engine = TorchEngine("mapanything", device="cuda")   # bf16 autocast on CUDA by default
    geo = engine.infer(frames)                           # frames: list of RGB uint8 arrays
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from codemesh.config import MODELS_DIR
from codemesh.engines.base import ViewGeometry
from codemesh.engines.preprocess import preprocess_mapanything, preprocess_vggt

ModelName = Literal["mapanything", "vggt"]
Precision = Literal["fp32", "bf16", "fp16"]

WEIGHTS = {"mapanything": "map-anything-apache", "vggt": "VGGT-1B"}
_TORCH_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def default_precision(device: str) -> Precision:
    """What the official demos use: bf16 on Ampere+ CUDA, fp16 on older CUDA, fp32 on CPU."""
    if device.startswith("cuda"):
        return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    return "fp32"


class TorchEngine:
    """GeometryEngine backed by the official PyTorch MapAnything / VGGT implementations."""

    name = "torch"

    def __init__(
        self,
        model: ModelName = "mapanything",
        device: str = "cpu",
        precision: Precision | None = None,
        weights_dir: Path | None = None,
    ) -> None:
        if model not in WEIGHTS:
            raise ValueError(f"Unknown model {model!r}; expected one of {list(WEIGHTS)}")
        self.model_name = model
        self.device = torch.device(device)
        self.precision: Precision = precision or default_precision(device)
        self.weights_dir = Path(weights_dir or MODELS_DIR / WEIGHTS[model])
        if not (self.weights_dir / "config.json").exists():
            raise FileNotFoundError(f"Weights not found in {self.weights_dir} (see bench/phase0/reference.py)")

        t0 = time.perf_counter()
        if model == "mapanything":
            from mapanything.models import MapAnything

            # Needs the DINOv2 code from torch hub on first run (cached afterwards); weights are local.
            self.model = MapAnything.from_pretrained(str(self.weights_dir))
        else:
            from vggt.models.vggt import VGGT

            self.model = VGGT.from_pretrained(str(self.weights_dir))
        self.model = self.model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0
        # Last infer() inputs and model-specific extra outputs, for benchmarks and debugging.
        self.last_inputs: np.ndarray | None = None
        self.last_extras: dict[str, np.ndarray] = {}

    def info(self) -> dict:
        return {
            "engine": self.name,
            "model": self.model_name,
            "weights": self.weights_dir.name,
            "device": str(self.device),
            "precision": self.precision,
            "load_seconds": round(self.load_seconds, 2),
        }

    def infer(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None) -> ViewGeometry:
        """Run multi-view inference on full-res RGB uint8 frames. `masks` is ignored."""
        t0 = time.perf_counter()
        if self.model_name == "mapanything":
            inputs, p2o = preprocess_mapanything(images)
        else:
            inputs, p2o = preprocess_vggt(images)
        self.last_inputs, self.last_extras = inputs, {}
        t1 = time.perf_counter()
        with torch.inference_mode():
            run = self._run_mapanything if self.model_name == "mapanything" else self._run_vggt
            geo = run(torch.from_numpy(inputs))
        geo.pixel_to_original = p2o
        geo.info = {**self.info(), "num_views": len(images), "engine_hw": list(inputs.shape[-2:]),
                    "preprocess_seconds": round(t1 - t0, 3), "infer_seconds": round(time.perf_counter() - t1, 3)}
        return geo

    def _run_mapanything(self, imgs: torch.Tensor) -> ViewGeometry:
        views = [
            {"img": imgs[i : i + 1], "true_shape": np.int32([imgs.shape[-2:]]), "idx": i, "instance": str(i),
             "data_norm_type": ["dinov2"]}
            for i in range(len(imgs))
        ]
        # Raw dense outputs (no masking/edge removal): filtering is the fuse stage's job.
        preds = self.model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=self.precision != "fp32",
            amp_dtype=self.precision,
            apply_mask=False,
            mask_edges=False,
        )

        def cat(key: str) -> np.ndarray:
            return torch.cat([p[key] for p in preds]).float().cpu().numpy()

        cam_to_world = cat("camera_poses").astype(np.float64)
        ambiguous_free = cat("non_ambiguous_mask") > 0.5
        # Pixels the model marks ambiguous (sky, reflections) get zero confidence.
        conf = cat("conf") * ambiguous_free
        self.last_extras = {"raw_conf": cat("conf"), "non_ambiguous_mask": ambiguous_free}
        return ViewGeometry(
            intrinsics=cat("intrinsics").astype(np.float64),
            extrinsics=np.linalg.inv(cam_to_world),
            depth=cat("depth_z")[..., 0],
            depth_conf=conf,
            pixel_to_original=np.empty(0),
            points=cat("pts3d"),
            points_conf=conf,
            metric_scale=True,  # MapAnything predicts a metric scale factor and applies it
        )

    def _run_vggt(self, imgs: torch.Tensor) -> ViewGeometry:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        imgs = imgs.to(self.device)[None]  # (1, N, 3, H, W)
        with torch.autocast(self.device.type, dtype=_TORCH_DTYPES[self.precision], enabled=self.precision != "fp32"):
            preds = self.model(imgs)
        extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], imgs.shape[-2:])

        def get(t: torch.Tensor) -> np.ndarray:
            return t[0].float().cpu().numpy()

        n = imgs.shape[1]
        extrinsics = np.tile(np.eye(4), (n, 1, 1))
        extrinsics[:, :3, :] = get(extri)
        return ViewGeometry(
            intrinsics=get(intri).astype(np.float64),
            extrinsics=extrinsics,
            depth=get(preds["depth"])[..., 0],
            depth_conf=get(preds["depth_conf"]),
            pixel_to_original=np.empty(0),
            points=get(preds["world_points"]),
            points_conf=get(preds["world_points_conf"]),
            metric_scale=False,  # VGGT is up to scale (first camera = world origin, normalized scene)
        )

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
