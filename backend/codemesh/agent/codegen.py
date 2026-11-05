"""Run Muse-written `build(api, evidence)` code in a sandbox and score the mesh it returns.

Layers: an AST whitelist (no imports, exec/eval/open, dunder or file-writing attributes, global/
nonlocal), restricted builtins, and a child process killed after `TIMEOUT` seconds. The child sends
the mesh back as PLY bytes; the parent validates it and scores silhouette IoU against the SAM masks.
shortcut: no memory rlimit (Windows has no `resource`); the timeout and the 60k-triangle cap bound it.
"""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

TIMEOUT = 20
MAX_TRIANGLES = 60_000
SAFE_BUILTINS = ("range", "len", "min", "max", "abs", "float", "int", "list", "dict", "tuple", "zip",
                 "enumerate", "sum", "round", "bool", "str", "True", "False", "None")
BANNED_NAMES = {"exec", "eval", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals",
                "vars", "input", "breakpoint", "help", "dir", "type", "object", "memoryview"}
# Methods on trimesh/numpy objects that touch the filesystem.
BANNED_ATTRS = {"export", "save", "savez", "load", "loadtxt", "tofile", "dump", "dumps", "fromfile",
                "write", "read", "system", "popen", "f_globals", "gi_frame", "cr_frame"}
ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.If, ast.Pass, ast.Break, ast.Continue, ast.Expr, ast.Call, ast.keyword, ast.Name, ast.Load,
    ast.Store, ast.Attribute, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.ListComp, ast.comprehension,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context, ast.Starred, ast.While,
)


class SandboxError(ValueError):
    pass


def check(code: str) -> None:
    """Raise SandboxError unless `code` only defines build(api, evidence) with whitelisted syntax."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise SandboxError(f"forbidden syntax: {type(node).__name__} (line {getattr(node, 'lineno', '?')})")
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else \
            node.arg if isinstance(node, ast.arg) else None
        if name and (name.startswith("_") or name in BANNED_NAMES or name in BANNED_ATTRS):
            raise SandboxError(f"forbidden name: {name} (line {node.lineno})")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        raise SandboxError("code must define build(api, evidence)")


def _child() -> None:
    """Child process: stdin {code, job_dir} -> stdout PLY bytes (or a JSON error on stderr, exit 1)."""
    import builtins

    from codemesh.agent import meshapi

    req = json.loads(sys.stdin.read())
    check(req["code"])
    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    env = {"__builtins__": safe}
    exec(compile(req["code"], "<muse>", "exec"), env)  # noqa: S102 - AST-whitelisted above
    ev = meshapi.evidence_namespace(meshapi.Evidence(Path(req["job_dir"])))
    mesh = env["build"](meshapi.API, ev)
    if isinstance(mesh, (list, tuple)):
        mesh = meshapi.combine(mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        raise SandboxError(f"build() returned {type(mesh).__name__}, expected a mesh")
    sys.stdout.buffer.write(mesh.export(file_type="ply"))


def execute(code: str, job_dir: Path) -> trimesh.Trimesh:
    """Run build() in a child process; return the mesh or raise SandboxError with the reason."""
    check(code)
    backend = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run([sys.executable, "-c", "from codemesh.agent.codegen import _child; _child()"],
                              input=json.dumps({"code": code, "job_dir": str(job_dir)}).encode(),
                              capture_output=True, timeout=TIMEOUT, cwd=backend)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"build() exceeded {TIMEOUT} s") from exc
    if proc.returncode:
        raise SandboxError(proc.stderr.decode(errors="replace").strip().splitlines()[-1][:500])
    return trimesh.load(io.BytesIO(proc.stdout), file_type="ply", force="mesh", process=False)


def validate(mesh: trimesh.Trimesh, bbox: dict) -> dict:
    """Problems that make a candidate unusable, plus body/triangle stats."""
    problems = []
    if not len(mesh.faces):
        problems.append("empty mesh")
    if len(mesh.faces) > MAX_TRIANGLES:
        problems.append(f"{len(mesh.faces)} triangles > {MAX_TRIANGLES}")
    if not np.isfinite(mesh.vertices).all():
        problems.append("non-finite vertices")
    bodies = mesh.split(only_watertight=False) if len(mesh.faces) else []
    open_bodies = sum(not b.is_watertight for b in bodies)
    if open_bodies:
        problems.append(f"{open_bodies}/{len(bodies)} bodies not watertight")
    lo, hi = np.asarray(bbox["min"]), np.asarray(bbox["max"])
    c, half = (lo + hi) / 2, (hi - lo) / 2 * 1.5
    if len(mesh.vertices) and (np.abs(mesh.vertices - c) > half + 1e-6).any():
        problems.append("mesh extends beyond 1.5x the evidence bbox")
    return {"problems": problems, "bodies": len(bodies), "triangles": int(len(mesh.faces)),
            "watertight": not open_bodies and bool(bodies)}


def score(mesh: trimesh.Trimesh, job_dir: Path, long_side: int = 384, max_views: int = 12,
          preview_views: int = 4) -> tuple[dict, np.ndarray]:
    """Silhouette IoU per camera (mask reprojection via the evaluator's ray caster) and a preview image.

    Preview rows: photo with SAM mask outline (green) and the candidate silhouette outline (red).
    """
    from codemesh.pipeline import evaluate
    from codemesh.pipeline.cameras import load_cameras

    cams, _ = load_cameras(job_dir / "geometry" / "cameras.json")
    inv = np.linalg.inv(evaluate._transform(job_dir))
    scene = evaluate._scene(mesh.vertices, mesh.faces)
    pick = np.unique(np.linspace(0, len(cams) - 1, min(max_views, len(cams))).astype(int))
    show = set(pick[np.unique(np.linspace(0, len(pick) - 1, min(preview_views, len(pick))).astype(int))])
    views, tiles = [], []
    for i in pick:
        cam = cams[i]
        photo_path, mask_path = evaluate._evidence(job_dir, cam)
        photo, mask = cv2.imread(str(photo_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if photo is None or mask is None:
            continue
        s = long_side / max(cam.width, cam.height)
        size = (max(1, round(cam.width * s)), max(1, round(cam.height * s)))
        photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0
        hit = np.isfinite(scene.cast_rays(evaluate._rays(cam, inv, size), nthreads=4)["t_hit"].numpy())
        iou, boundary_f = evaluate._silhouette(hit, mask)
        views.append({"index": cam.index, "iou": round(iou, 4), "boundary_f": round(boundary_f, 4)})
        if i in show:
            tile = photo.copy()
            for m, colour in ((mask, (0, 255, 0)), (hit, (0, 0, 255))):
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, contours, -1, colour, 2)
            cv2.putText(tile, f"#{cam.index} IoU {iou:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            tiles.append(tile)
    h = max(t.shape[0] for t in tiles)
    preview = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])
    ious = [v["iou"] for v in views]
    return {"iou_mean": round(float(np.mean(ious)), 4), "iou_min": round(float(np.min(ious)), 4),
            "views": views}, preview[:, :, ::-1].copy()  # RGB for Muse / PIL


def build_with_code(job_dir: Path, code: str, name: str = "candidate") -> dict:
    """Execute Muse code, validate and score it; write agent/<name>.ply and agent/<name>_preview.png.

    Returns {"ok", "error" | ("validation", "metrics", "mesh_path", "preview_path")}.
    """
    from codemesh.agent.meshapi import Evidence

    job_dir = Path(job_dir)
    out = job_dir / "agent"
    out.mkdir(parents=True, exist_ok=True)
    try:
        mesh = execute(code, job_dir)
    except SandboxError as exc:
        return {"ok": False, "error": str(exc)}
    validation = validate(mesh, Evidence(job_dir).bbox)
    mesh_path = out / f"{name}.ply"
    mesh.export(mesh_path)
    result = {"ok": not validation["problems"], "validation": validation, "mesh_path": str(mesh_path)}
    if len(mesh.faces) and np.isfinite(mesh.vertices).all():
        metrics, preview = score(mesh, job_dir)
        preview_path = out / f"{name}_preview.png"
        cv2.imwrite(str(preview_path), preview[:, :, ::-1])
        result.update(metrics=metrics, preview_path=str(preview_path))
    return result
