"""Export stage: texture/ outputs -> export/model.glb (+ export/preview.png).

The GLB has one PBR metallic-roughness material (baseColorTexture = albedo, normalTexture =
normal, metallic 0, roughness from settings) and POSITION, NORMAL, TEXCOORD_0 and TANGENT
attributes. Coordinates are the asset frame of mesh/transform.json (glTF: meters, +Y up).
trimesh only writes custom attributes with a leading underscore, so `_TANGENT` is renamed to
the standard `TANGENT` in the JSON chunk after export.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

from codemesh.pipeline.texture import camera_rays, cast, raycaster, sample_bilinear, uv_to_pixel

Progress = Callable[[float, str], None] | None


def glb_json(data: bytes) -> tuple[dict, bytes]:
    """Split a GLB into (json dict, remaining chunks after the JSON chunk)."""
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB file")
    length, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON":
        raise ValueError("GLB first chunk is not JSON")
    return json.loads(data[20:20 + length]), data[20 + length:]


def build_glb(doc: dict, rest: bytes) -> bytes:
    body = json.dumps(doc, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 4)
    chunk = struct.pack("<I4s", len(body), b"JSON") + body + rest
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


def render(mesh: dict, albedo: np.ndarray, normal: np.ndarray | None, rays: np.ndarray,
           light: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ray cast the textured mesh (uv_mesh.npz arrays). Returns (RGB uint8 (h, w, 3), hit mask).
    With `light` (unit direction toward the light) the normal map is applied with Lambert +
    ambient shading; without it the raw albedo is returned (used for texture PSNR checks)."""
    h, w = rays.shape[:2]
    V, F, uv = mesh["vertices"], mesh["faces"], mesh["uv"]
    t, prim, b = cast(raycaster(V, F), rays.reshape(-1, 6))
    hit = np.isfinite(t)
    f = F[prim[hit]]
    bh = b[hit]
    px = uv_to_pixel(np.einsum("mk,mkd->md", bh, uv[f]), albedo.shape[0])
    rgb = np.zeros((h * w, 3), np.float32)
    rgb[hit] = sample_bilinear(albedo, px[:, 0], px[:, 1])
    if light is not None:
        n = np.einsum("mk,mkd->md", bh, mesh["normals"][f])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if normal is not None:
            tan = np.einsum("mk,mkd->md", bh, mesh["tangents"][f])
            T = tan[:, :3] - n * (n * tan[:, :3]).sum(1, keepdims=True)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            B = np.cross(n, T) * np.sign(tan[:, 3:4] + 1e-9)
            ts = sample_bilinear(normal, px[:, 0], px[:, 1]).astype(np.float32) / 127.5 - 1
            n = ts[:, :1] * T + ts[:, 1:2] * B + ts[:, 2:3] * n
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        view = -rays.reshape(-1, 6)[hit, 3:]
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        n *= np.where((n * view).sum(1, keepdims=True) < 0, -1, 1)  # two-sided
        shade = 0.3 + 0.7 * np.clip(n @ light, 0, 1)
        rgb[hit] *= shade[:, None]
        rgb[~hit] = 235  # light background for the showcase
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(h, w, 3), hit.reshape(h, w)


def preview_rays(vertices: np.ndarray, width: int = 1024, height: int = 768) -> tuple[np.ndarray, np.ndarray]:
    """3/4 view (azimuth 45 deg, elevation 25 deg, +Y up) framing the mesh. Returns (rays, light)."""
    lo, hi = vertices.min(0), vertices.max(0)
    center, radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    a, e = np.radians(45), np.radians(25)
    d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    f = 0.5 * height / np.tan(np.radians(20))  # 40 deg vertical field of view
    dist = radius / np.sin(np.radians(20)) * 1.05
    z = -d  # OpenCV look-at: x right, y down, z forward
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    E = np.eye(4)
    E[:3, :3] = np.stack([x, np.cross(z, x), z])
    E[:3, 3] = -E[:3, :3] @ (center + dist * d)
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
    rays = camera_rays(K, E, width, height)
    light = d + np.array([0.3, 0.6, 0.0])
    return rays, light / np.linalg.norm(light)


def run(job_dir: Path, settings: dict, progress: Progress = None) -> dict[str, Any]:
    """Write export/model.glb and export/preview.png. Settings: roughness (0.8), metallic (0)."""
    t0 = time.perf_counter()
    report = progress or (lambda f, m: None)
    src, out = job_dir / "texture", job_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    mesh = dict(np.load(src / "uv_mesh.npz"))
    albedo = np.asarray(Image.open(src / "albedo.png").convert("RGB"))
    normal = np.asarray(Image.open(src / "normal.png").convert("RGB"))

    report(0.1, "write GLB")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(albedo), normalTexture=Image.fromarray(normal),
        metallicFactor=float(settings.get("metallic", 0.0)), roughnessFactor=float(settings.get("roughness", 0.8)))
    tm = trimesh.Trimesh(mesh["vertices"], mesh["faces"], vertex_normals=mesh["normals"],
                         visual=trimesh.visual.TextureVisuals(uv=mesh["uv"], material=material), process=False)
    tm.vertex_attributes["TANGENT"] = mesh["tangents"].astype(np.float32)
    doc, rest = glb_json(tm.export(file_type="glb", include_normals=True))
    for m in doc["meshes"]:
        for p in m["primitives"]:
            if "_TANGENT" in p["attributes"]:
                p["attributes"]["TANGENT"] = p["attributes"].pop("_TANGENT")
    path = out / "model.glb"
    path.write_bytes(build_glb(doc, rest))

    report(0.5, "validate GLB")
    check = validate_glb(path, len(mesh["faces"]))

    report(0.6, "render preview")
    rays, light = preview_rays(mesh["vertices"])
    img, _ = render(mesh, albedo, normal, rays, light)
    cv2.imwrite(str(out / "preview.png"), img[..., ::-1])
    report(1.0, "export done")
    return {**check, "seconds": round(time.perf_counter() - t0, 2)}


def validate_glb(path: Path, expected_faces: int) -> dict[str, Any]:
    """Reload the GLB and check UVs, both textures, TANGENT and the triangle count. Raises on failure."""
    doc, _ = glb_json(path.read_bytes())
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    mat = doc["materials"][0]
    loaded = trimesh.load(path, force="mesh", process=False)
    problems = [msg for ok, msg in (
        ({"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"} <= attrs.keys(), f"attributes {sorted(attrs)}"),
        ("baseColorTexture" in mat.get("pbrMetallicRoughness", {}), "no baseColorTexture"),
        ("normalTexture" in mat, "no normalTexture"),
        (len(loaded.faces) == expected_faces, f"faces {len(loaded.faces)} != {expected_faces}"),
        (getattr(loaded.visual, "uv", None) is not None and len(loaded.visual.uv) == len(loaded.vertices), "no UVs"),
    ) if not ok]
    if problems:
        raise ValueError(f"invalid GLB {path}: {'; '.join(problems)}")
    return {"glb_bytes": path.stat().st_size, "glb_mb": round(path.stat().st_size / 2**20, 2),
            "triangles": expected_faces, "vertices": len(loaded.vertices), "textures": len(doc.get("images", []))}
