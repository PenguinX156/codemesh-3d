"""Segment stage: the user's selection -> full-res object masks for the ingested frames.

Input: Job.settings["selection"] = {frame_index, points: [{x, y, label}] (normalized), text?}
(stored by POST /jobs/{id}/select). Video jobs: SAM 3.1 tracks the object through the original
upload (re-encoded to H.264 MP4 only when the API cannot read it) and we keep the masks of the
frames ingest selected. Photo jobs: SAM 3.1 per image; the clicked points apply only to the
selected image, the others use the text prompt (fallback "object").

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

from codemesh.jobs.context import StageContext
from codemesh.meta import sam
from codemesh.pipeline.ingest import rotate_upright

MIN_VIEWS = 8  # fewer usable views cannot give a closed model
MAX_BORDER_FRACTION = 0.4  # masks hugging the border = background/table picked, not the object
API_CODECS = frozenset({"h264", "hevc", "mpeg4"})
API_CONTAINERS = frozenset({".mp4", ".mov"})
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


def api_video(src: Path, work: Path, rotation: float) -> Path:
    """`src` if SAM accepts it, else an upright H.264 MP4 re-encode in `work` (same frame order)."""
    with av.open(str(src)) as c:
        codec = c.streams.video[0].codec_context.name
    if src.suffix.lower() in API_CONTAINERS and codec in API_CODECS:
        return src
    dst = work / "sam_input.mp4"
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        stream = None
        for frame in inp.decode(video=0):
            rgb = rotate_upright(frame.to_ndarray(format="rgb24"), rotation)
            if stream is None:
                stream = out.add_stream("libx264", rate=30)
                stream.width, stream.height = rgb.shape[1] // 2 * 2, rgb.shape[0] // 2 * 2
                stream.pix_fmt, stream.options = "yuv420p", {"crf": "18"}
            vf = av.VideoFrame.from_ndarray(rgb[: stream.height, : stream.width], format="rgb24")
            out.mux(stream.encode(vf))
        out.mux(stream.encode())
    return dst


def segment_masks(ctx: StageContext, frames: dict[str, Any], sel: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict]:
    """Run SAM 3.1; return {frame index: bool mask at full frame size} and call info."""
    entries = frames["frames"]
    chosen = entries[int(sel["frame_index"])]
    points = [sam.Point(float(p["x"]), float(p["y"]), int(p.get("label", 1))) for p in sel.get("points", [])]
    text = (sel.get("text") or "").strip() or None
    if not points and not text:
        raise ValueError("selection needs at least one point or a text prompt")
    size = (int(chosen["width"]), int(chosen["height"]))
    if frames["kind"] == "video":
        video = sorted(p for p in ctx.paths.input.iterdir() if p.is_file())[0]
        src = api_video(video, ctx.paths.masks, float(frames.get("rotation") or 0))
        ctx.progress(0.1, "tracking the object with SAM 3.1")
        res = sam.segment_video(src, points=points, text=text, prompt_frame=int(chosen["source_index"]),
                                frame_indices=[int(e["source_index"]) for e in entries], size=size)
        by_source = {int(e["source_index"]): int(e["index"]) for e in entries}
        masks = {by_source[f]: m.mask for f, m in res.masks.items() if f in by_source}
        return masks, {"usage": res.usage, "timings": res.timings, "candidates": len(res.candidates)}
    ctx.progress(0.1, "segmenting photos with SAM 3.1")
    files = [ctx.paths.frames / e["file"] for e in entries]
    sel_idx = int(chosen["index"])
    first = sam.segment_images([files[sel_idx]], points=points, text=text)[0]
    others = [i for i in range(len(entries)) if i != sel_idx]
    rest = sam.segment_images([files[i] for i in others], text=text or "object") if others else []
    masks = {i: r.masks[0].mask for i, r in zip([sel_idx, *others], [first, *rest]) if 0 in r.masks}
    return masks, {"images": len(entries)}


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
