"""SAM 3.1 segmentation through the Meta Model API (`POST /v1/responses`, model `sam-3.1`).

Public entry points:
    segment_video(video_path, points=..., text=..., prompt_frame=...) -> SegResult
    segment_images(images, points=..., text=...) -> list[SegResult]

Wire format (see bench/phase0/META_API.md): the answer is special-token text, one line per frame,
`<Nf>id<|box;x1;y1;x2;y2;w;h;c|><|mask;x=0;y=0;c;data=H,W,<enc>payload|>,id<|box..|><|mask..|>`.
The mask raster covers exactly the inclusive box, so a full-resolution mask is that raster pasted
into an empty frame at (x1, y1). Mask payload decoding (a custom range coder) is delegated to the
official `meta-sam-parser` package; everything else is parsed here.

Prompts take normalized 0..1 coordinates (resolution independent) and are converted to the media's
pixel space here. Meta's rule: a request carries either a text concept or object prompts, never
both. With points we send an object prompt; with text only a concept prompt; with both we send the
concept and use the points to choose which match the user meant.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import av
import cv2
import numpy as np
from meta_sam_parser import (
    FramePrompt,
    PromptObject,
    PromptPoint,
    SegmentationMask,
    build_object_prompt,
)
from PIL import Image, ImageOps

from codemesh import config
from codemesh.meta._maskdec import decode_one
from codemesh.meta.client import MetaAPIError, MetaClient

log = logging.getLogger(__name__)

Direction = Literal["both", "forward", "backward"]
OBJECT_ID = 1  # id we give the user's clicked object in object prompts
PARALLEL_DECODE_MIN = 6  # below this many masks, a process pool costs more than it saves

_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_FRAME = re.compile(r"<(\d+)f>")


@dataclass(frozen=True)
class Point:
    """A click in normalized image coordinates (0..1, origin top-left). label 1 = on the object."""

    x: float
    y: float
    label: int = 1


@dataclass
class ObjectMask:
    """One object on one frame, in full-resolution source pixels."""

    object_id: int
    frame: int
    mask: np.ndarray  # (H, W) bool, same size as the source frame
    box: tuple[int, int, int, int]  # x1, y1, x2, y2, inclusive pixels
    score: float | None  # model confidence 0..1

    def contains(self, x: int, y: int) -> bool:
        h, w = self.mask.shape
        return 0 <= x < w and 0 <= y < h and bool(self.mask[y, x])


@dataclass
class SegResult:
    """Outcome of one segmentation request, reduced to the object the user meant."""

    size: tuple[int, int]  # (width, height) of the media in pixels
    selected_id: int | None  # None = nothing matched
    masks: dict[int, ObjectMask] = field(default_factory=dict)  # frame index -> selected object
    candidates: list[ObjectMask] = field(default_factory=list)  # all matches on the prompt frame
    usage: dict[str, Any] = field(default_factory=dict)  # images_processed / video_frames_processed
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)  # upload / api / decode seconds
    raw_text: str = ""

    @property
    def frames_with_mask(self) -> list[int]:
        return sorted(self.masks)


# --- parsing -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    """A parsed, not yet decoded, box+mask record."""

    frame: int
    object_id: int
    box: tuple[int, int, int, int]
    frame_size: tuple[int, int]  # (w, h)
    score: float | None
    mask: SegmentationMask


def _fields(body: str) -> dict[str, str]:
    return dict(part.partition("=")[::2] for part in body.split(";") if "=" in part)


def parse_output(text: str) -> list[RawRecord]:
    """Parse the special-token output text (all lines) into records, in source order.

    Frame markers are explicit (`<Nf>`), never inferred from line position; ids are per-object and
    may be sparse. Fields are read by key so tokens parse the same with or without `c`.
    """
    records: list[RawRecord] = []
    for line in map(str.strip, text.splitlines()):
        marker = _FRAME.match(line)
        if not marker:
            continue
        frame = int(marker[1])
        for rec in _RECORD.finditer(line, marker.end()):
            box_f, mask_f = _fields(rec[2]), _fields(rec[3])
            height, width, payload = mask_f["data"].split(",", 2)
            score = box_f.get("c")
            records.append(
                RawRecord(
                    frame=frame,
                    object_id=int(rec[1]),
                    box=tuple(int(box_f[k]) for k in ("x1", "y1", "x2", "y2")),  # type: ignore[arg-type]
                    frame_size=(int(box_f["w"]), int(box_f["h"])),
                    score=float(score) if score is not None else None,
                    mask=SegmentationMask(
                        encoding="one_bit" if payload.startswith("!") else "lossless",
                        payload=payload,
                        width=int(width),
                        height=int(height),
                    ),
                )
            )
    return records


def to_object_masks(records: Sequence[RawRecord]) -> list[ObjectMask]:
    """Decode records and paste each box-local raster into a full-resolution frame mask.

    Decoding is pure Python (~0.2-0.4 s per 100k pixels), so larger batches go to a process pool.
    """
    jobs = [(r.mask.encoding, r.mask.payload, r.mask.width, r.mask.height) for r in records]
    if len(jobs) >= PARALLEL_DECODE_MIN:
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            rasters = list(pool.map(decode_one, jobs))
    else:
        rasters = [decode_one(j) for j in jobs]

    out: list[ObjectMask] = []
    for rec, raster in zip(records, rasters):
        (w, h), (x1, y1, x2, y2) = rec.frame_size, rec.box
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if raster.shape != (bh, bw):  # should not happen; stay robust to off-by-one rasters
            raster = cv2.resize(raster.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((h, w), dtype=bool)
        full[y1 : y2 + 1, x1 : x2 + 1] = raster
        out.append(ObjectMask(rec.object_id, rec.frame, full, (x1, y1, x2, y2), rec.score))
    return out


# --- choosing the match ------------------------------------------------------------------------


def choose_object(objects: Sequence[ObjectMask], point_px: tuple[int, int] | None = None) -> ObjectMask | None:
    """Pick the match the user meant: the one containing the click, else the highest score.

    Among several containing the click the highest score wins; ties and missing scores fall back to
    the larger mask. Returns None for an empty list.
    """
    if not objects:
        return None
    pool = [o for o in objects if point_px and o.contains(*point_px)] or list(objects)
    return max(pool, key=lambda o: (o.score if o.score is not None else -1.0, int(o.mask.sum())))


# --- requests ----------------------------------------------------------------------------------


def _to_pixels(points: Sequence[Point], size: tuple[int, int]) -> list[tuple[int, int, int]]:
    w, h = size
    return [(min(int(p.x * w), w - 1), min(int(p.y * h), h - 1), p.label) for p in points]


def _prompt_text(
    points: Sequence[Point], text: str | None, size: tuple[int, int], frame: int
) -> str:
    """Object prompt when there are points and no text, concept prompt for text."""
    if text:
        return text.strip()
    if not points:
        raise ValueError("need points and/or text to segment")
    pts = [PromptPoint(x, y, "positive" if label else "negative") for x, y, label in _to_pixels(points, size)]
    return build_object_prompt(
        size=size, objects=[PromptObject(id=OBJECT_ID, prompts=[FramePrompt(frame=frame, points=pts)])]
    )


def _run(client: MetaClient, content: list[dict[str, Any]], metadata: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """One streaming /responses call. Returns (output text lines joined by \\n, usage)."""
    payload = {
        "model": config.SAM_MODEL,
        "input": [{"type": "message", "role": "user", "content": content}],
        # one_bit: ~8x smaller payloads and faster to decode than lossless.
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true", **metadata},
    }
    lines: list[str] = []
    usage: dict[str, Any] = {}
    for event in client.stream_events("/responses", payload):
        kind = event.get("type")
        if kind == "response.output_text.delta" and event.get("delta"):
            lines.append(event["delta"])
        elif kind == "response.completed":
            usage = event["response"].get("usage") or {}
        elif kind in ("response.failed", "response.incomplete", "error"):
            raise MetaAPIError(f"SAM request failed: {str(event)[:300]}")
    return "\n".join(lines), usage


def _select(
    result: SegResult,
    records: list[RawRecord],
    prompt_frame: int,
    point_px: tuple[int, int] | None,
    frame_filter: set[int] | None,
) -> None:
    """Fill result.candidates / selected_id / masks from parsed records (decodes lazily)."""
    if not records:
        return
    on_prompt = [r for r in records if r.frame == prompt_frame]
    if on_prompt:
        result.candidates = to_object_masks(on_prompt)
        chosen = choose_object(result.candidates, point_px)
        result.selected_id = chosen.object_id if chosen else None
    else:  # nothing on the prompt frame: follow the id that appears on most frames
        ids = [r.object_id for r in records]
        result.selected_id = max(set(ids), key=ids.count)
    wanted = [
        r
        for r in records
        if r.object_id == result.selected_id and (frame_filter is None or r.frame in frame_filter)
        and r.frame != prompt_frame
    ]
    result.masks = {m.frame: m for m in to_object_masks(wanted)}
    for cand in result.candidates:
        if cand.object_id == result.selected_id and (frame_filter is None or cand.frame in frame_filter):
            result.masks[cand.frame] = cand


def segment_video(
    video_path: str | Path,
    *,
    points: Sequence[Point] = (),
    text: str | None = None,
    prompt_frame: int = 0,
    direction: Direction = "both",
    frame_indices: Sequence[int] | None = None,
    size: tuple[int, int] | None = None,
    client: MetaClient | None = None,
) -> SegResult:
    """Track one object through a video with SAM 3.1.

    Args:
        video_path: MP4/MOV (H.264/H.265). Result frame N is the Nth decoded frame of this file.
        points: clicks on `prompt_frame`, normalized 0..1 (label 1 positive, 0 negative).
        text: noun phrase concept ("yellow bulldozer"). With points too, points pick the match.
        direction: tracking direction for object prompts (forward/backward cost fewer frames).
        frame_indices: keep masks only for these frames (the API still tracks and bills all).
        size: media (width, height) as displayed; default read from the first decoded frame.
            Rotated phone clips must pass the displayed size.
    Billing: per tracked frame, minimum 50 frames per request ($0.20 / 1000 frames).
    """
    own = client is None
    client = client or MetaClient()
    t0 = time.perf_counter()
    try:
        size = size or _video_size(video_path)
        t_up = time.perf_counter()
        file_id = client.upload_file(video_path)
        t_api = time.perf_counter()
        try:
            prompt = _prompt_text(points, text, size, prompt_frame)
            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ]
            meta = {"propagation_direction": direction} if not text else {}
            out, usage = _run(client, content, meta)
        finally:
            client.delete_file(file_id)
    finally:
        if own:
            client.close()
    t_dec = time.perf_counter()
    result = SegResult(size=size, selected_id=None, usage=usage, raw_text=out)
    point_px = _to_pixels(points[:1], size)[0][:2] if points else None
    _select(result, parse_output(out), prompt_frame, point_px, set(frame_indices) if frame_indices else None)
    t_end = time.perf_counter()
    result.seconds = t_end - t0
    result.timings = {"upload": t_api - t_up, "api": t_dec - t_api, "decode": t_end - t_dec}
    return result


def segment_images(
    images: Sequence[str | Path | np.ndarray],
    *,
    points: Sequence[Point] = (),
    text: str | None = None,
    client: MetaClient | None = None,
) -> list[SegResult]:
    """Segment each image (file path or RGB uint8 array) with the same prompt; one result each.

    Billed $2.50 per 1000 images. Images are sent as JPEG (q95) at full resolution, so masks come
    back at full resolution. Several images run concurrently (rate limit is 3000 RPM).
    """
    from concurrent.futures import ThreadPoolExecutor

    own = client is None
    client = client or MetaClient()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda img: _segment_one_image(client, img, points, text), images))
    finally:
        if own:
            client.close()


def _segment_one_image(
    client: MetaClient, image: str | Path | np.ndarray, points: Sequence[Point], text: str | None
) -> SegResult:
    t0 = time.perf_counter()
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # size must be the displayed size
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=95)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    size = pil.size
    content = [
        {"type": "input_text", "text": _prompt_text(points, text, size, 0)},
        {"type": "input_image", "image_url": url},
    ]
    out, usage = _run(client, content, {})
    result = SegResult(size=size, selected_id=None, usage=usage, raw_text=out)
    point_px = _to_pixels(points[:1], size)[0][:2] if points else None
    _select(result, parse_output(out), 0, point_px, None)
    result.seconds = time.perf_counter() - t0
    return result


def _video_size(path: str | Path) -> tuple[int, int]:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        return frame.width, frame.height
