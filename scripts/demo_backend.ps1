# Local demo backend on :8010 with its own jobs folder (dev/test instances use :8000 + jobs/).
$root = Split-Path $PSScriptRoot -Parent
$env:CODEMESH_JOBS_DIR = "$root\jobs_demo"
$env:CODEMESH_CORS_ORIGINS = "http://localhost:5180"
$env:PYTHONUTF8 = "1"
uv run --project "$root\backend" uvicorn codemesh.main:app --port 8010
