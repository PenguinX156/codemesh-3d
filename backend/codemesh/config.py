"""Settings loaded from environment / the repo-root .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

META_API_KEY = os.getenv("META_API_KEY", "")
META_API_BASE = os.getenv("META_API_BASE", "https://api.meta.ai/v1")
MUSE_MODEL = os.getenv("MUSE_MODEL", "muse-spark-1.3-contributor")
MUSE_REASONING = os.getenv("MUSE_REASONING", "high")
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")

JOBS_DIR = Path(os.getenv("CODEMESH_JOBS_DIR", REPO_ROOT / "jobs"))
MODELS_DIR = Path(os.getenv("CODEMESH_MODELS_DIR", REPO_ROOT / "models"))
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
BLENDER_EXE = os.getenv("BLENDER_EXE", str(THIRD_PARTY_DIR / "blender" / "blender.exe"))
GEOMETRY_ENGINE = os.getenv("CODEMESH_GEOMETRY_ENGINE", "auto")  # auto | modal | torch | rocm
CORS_ORIGINS = [o for o in os.getenv("CODEMESH_CORS_ORIGINS", "http://localhost:5173").split(",") if o]
