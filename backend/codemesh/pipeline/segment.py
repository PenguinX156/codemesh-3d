"""Segment stage: the user's selection -> full-res object masks for the ingested frames.

Input: Job.settings["selection"] = {frame_index, points: [{x, y, label}] (normalized)?, text?}
(stored by POST /jobs/{id}/select; at least a point or text). The frames ingest kept are encoded as
a short H.264 clip (<= 1920 px, padded to 50 frames) and tracked by ONE SAM 3.1 video request;
the prompt frame is the selected frame's position in the clip. Masks map 1:1 back to the kept
frames and are upscaled to full frame resolution. Photo sets use the same clip path.

Outputs (masks/): <index>.png (255 = object, full frame resolution), valid.json (frames usable
by later stages: mask not empty and touching <= 40% of the image border), preview.jpg (overlay
grid for the UI, artifact "masks_preview").
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image

from codemesh.jobs.context import StageContext
from codemesh.meta import sam

MIN_VIEWS = 8  # fewer usable views cannot give a closed model
MAX_BORDER_FRACTION = 0.4  # masks hugging the border = background/table picked, not the object
CLIP_MAX_SIDE = 1920  # long side of the clip uploaded to SAM
SAM_MIN_FRAMES = 50  # the API bills at least 50 frames per request
PREVIEW_TILES, PREVIEW_TILE_PX = 8, 480


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)  # imencode + write_bytes: cv2.imwrite fails on non-ASCII paths
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    path.write_bytes(buf.tobytes())


def border_fraction(mask: np.ndarray) -> float:
    """Fraction of the image border pixels covered by the mask."""
    border = np.concatenate([mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]])
    return float(border.mean())


def build_clip(files: list[Path], dst: Path) -> tuple[int, int]:
    """Encode the kept frames, in order, as a small H.264 MP4 for SAM; returns its (width, height).

    Long side <= CLIP_MAX_SIDE (even dimensions for yuv420p, so the aspect is off by < 1 px), padded
    to the API's SAM_MIN_FRAMES minimum by repeating the last frame. SAM bills per clip frame, so
    this costs len(files) frames instead of every frame of the original video.
    """
    with av.open(str(dst), "w") as out:
        stream = None
        for k in range(max(len(files), SAM_MIN_FRAMES)):
            if k < len(files):
                with Image.open(files[k]) as im:
                    rgb = im.convert("RGB")
                if stream is None:
                    s = min(1.0, CLIP_MAX_SIDE / max(rgb.size))
                    size = (max(2, round(rgb.width * s) // 2 * 2), max(2, round(rgb.height * s) // 2 * 2))
                    stream = out.add_stream("libx264", rate=10)
                    stream.width, stream.height = size
                    stream.pix_fmt, stream.options = "yuv420p", {"crf": "20"}
                vf = av.VideoFrame.from_ndarray(np.asarray(rgb.resize(size, Image.LANCZOS)), format="rgb24")
            out.mux(stream.encode(vf))  # k >= len(files): vf is still the last frame
        out.mux(stream.encode())
    return size


def upscale_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bool mask -> bool mask of `size` (w, h). SAM returns no soft mask, so smooth the binary one:
    bilinear upscale of 0/1 then threshold 0.5 rounds the stair-steps of the clip resolution."""
    if mask.shape == (size[1], size[0]):
        return mask
    return cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= 0.5


def segment_masks(ctx: StageContext, frames: dict[str, Any], sel: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict]:
    """Run ONE SAM 3.1 video request on a clip of the kept frames (photos too: tracking keeps the
    identity and is cheaper than per-image). Returns {frame position: bool mask at full frame size}."""
    entries = frames["frames"]
    points = [sam.Point(float(p["x"]), float(p["y"]), int(p.get("label", 1))) for p in sel.get("points", [])]
    text = (sel.get("text") or "").strip() or None
    if not points and not text:
        raise ValueError("selection needs at least one point or a text prompt")
    ctx.progress(0.05, "building the SAM clip")
    clip = ctx.paths.masks / "sam_input.mp4"
    clip_size = build_clip([ctx.paths.frames / e["file"] for e in entries], clip)
    ctx.progress(0.1, "tracking the object with SAM 3.1")
    res = sam.segment_video(clip, points=points, text=text, prompt_frame=int(sel["frame_index"]),
                            frame_indices=range(len(entries)), size=clip_size)
    masks = {i: upscale_mask(m.mask, (int(entries[i]["width"]), int(entries[i]["height"])))
             for i, m in res.masks.items() if 0 <= i < len(entries)}
    return masks, {"usage": res.usage, "timings": res.timings, "candidates": len(res.candidates),
                   "clip_size": list(clip_size)}


def preview(ctx: StageContext, entries: list[dict], masks: dict[int, np.ndarray], first: int) -> Path:
    """Grid of up to PREVIEW_TILES frames (selected frame first) with the mask tinted and outlined."""
    order = [first] + [i for i in np.linspace(0, len(entries) - 1, PREVIEW_TILES).round().astype(int).tolist()
                       if i != first]
    order = list(dict.fromkeys(order))[:PREVIEW_TILES]
    tiles = []
    for i in order:
        bgr = cv2.imdecode(np.fromfile(ctx.paths.frames / entries[i]["file"], np.uint8), cv2.IMREAD_COLOR)
        m = masks.get(i, np.zeros(bgr.shape[:2], bool)).astype(np.uint8)
        s = PREVIEW_TILE_PX / max(bgr.shape[:2])
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        small[~ms] = (small[~ms] * 0.35).astype(np.uint8)
        contours, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(small, contours, -1, (60, 220, 255), 2)
        tiles.append(small)
    th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    cols = min(4, len(tiles))
    rows = -(-len(tiles) // cols)
    grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for n, t in enumerate(tiles):
        r, c = divmod(n, cols)
        grid[r * th : r * th + t.shape[0], c * tw : c * tw + t.shape[1]] = t
    path = ctx.paths.masks / "preview.jpg"
    write_image(path, grid)
    return path


def run(ctx: StageContext) -> dict[str, Any]:
    """Write masks/<index>.png, masks/valid.json and masks/preview.jpg. Returns metrics."""
    t0 = time.perf_counter()
    sel = ctx.settings.get("selection")
    if not sel:
        raise ValueError("no selection: POST /jobs/{id}/select first")
    frames = json.loads((ctx.paths.frames / "frames.json").read_text(encoding="utf-8"))
    entries = frames["frames"]
    if not 0 <= int(sel["frame_index"]) < len(entries):
        raise ValueError(f"frame_index must be < {len(entries)}")
    shutil.rmtree(ctx.paths.masks, ignore_errors=True)
    ctx.paths.masks.mkdir(parents=True)

    masks, info = segment_masks(ctx, frames, sel)
    ctx.progress(0.8, "writing masks")
    coverage, empty, border, valid = {}, [], [], []
    for e in entries:
        i = int(e["index"])
        m = masks.get(i)
        if m is None or m.shape != (e["height"], e["width"]):  # no mask or wrong size -> unusable frame
            m = np.zeros((e["height"], e["width"]), bool)
        write_image(ctx.paths.masks / f"{Path(e['file']).stem}.png", m.astype(np.uint8) * 255)
        coverage[i] = round(float(m.mean()), 4)
        if not m.any():
            empty.append(i)
        elif border_fraction(m) > MAX_BORDER_FRACTION:
            border.append(i)
        else:
            valid.append(i)
    (ctx.paths.masks / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    preview(ctx, entries, masks, int(sel["frame_index"]))
    ctx.set_artifact("masks_preview", "masks/preview.jpg")
    metrics = {"frames": len(entries), "frames_valid": len(valid), "frames_empty": empty,
               "frames_border": border, "coverage": coverage,
               "coverage_mean": round(float(np.mean([coverage[i] for i in valid])), 4) if valid else 0.0,
               "seconds": round(time.perf_counter() - t0, 2), **info}
    if len(valid) < MIN_VIEWS:
        raise RuntimeError(f"only {len(valid)} frames have a usable mask (need {MIN_VIEWS}); "
                           "select the object again (more points or a clearer text) or capture more views")
    return metrics
