"""Texture + export the Muse-coded mesh: mesh/low.ply = Muse mesh, mesh/high.ply = hull (normal bake).

uv run --project backend python bench/muse_tools/texture_mug.py jobs/cases/mug_code/agent/code_1.ply
"""

import json
import shutil
import sys
from pathlib import Path

from codemesh.pipeline import export, texture

JOB = Path(__file__).resolve().parents[2] / "jobs/cases/mug_code"
shutil.copy(sys.argv[1], JOB / "mesh" / "low.ply")
print("TEXTURE", json.dumps(texture.run(JOB, {}), default=str)[:600])
print("EXPORT", json.dumps(export.run(JOB, {}), default=str)[:600])
shutil.copy(JOB / "export" / "preview.png", Path(__file__).parent / "mug_code_export_preview.png")
