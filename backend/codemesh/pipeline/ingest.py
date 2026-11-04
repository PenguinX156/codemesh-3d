"""Ingest stage: video or photo set -> K sharp, distinct, full-resolution frames + frames.json.

Video: one decode pass (threaded). Each frame is scored on a downscaled gray copy (variance of
Laplacian); the timeline is split into K windows and the sharpest frame of each window is kept,
unless it nearly duplicates the previous kept frame (32x32 thumbnail difference). Only kept
frames are converted to RGB at full resolution, rotated upright and saved.

Photos: same selection over the photos in file-name order, after applying EXIF orientation.

Output (jobs/<id>/frames/): 0000.jpg ... (full res, JPEG q95 4:4:4), thumbs/0000.jpg (512 px),
frames.json (paths inside it are relative to the frames folder).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image, ImageOps

from codemesh.jobs.cache import cached_metrics, invalidate, stage_key, write_manifest
from codemesh.jobs.context import StageContext

try:  # optional HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

PRESET_FRAMES = {"draft": 24, "standard": 40, "high": 64}
VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"} | ({".heic", ".heif"} if HEIF_SUPPORTED else set()))

INGEST_VERSION = 1  # bump when selection/output format changes, to invalidate caches
SCORE_LONG_SIDE = 640  # sharpness is measured at this size (fast, and robust to sensor noise)
THUMB_LONG_SIDE = 512
DUP_SIZE = 32
DUP_THRESHOLD = 2.0  # mean abs gray difference (0..255) of 32x32 thumbs below which frames are duplicates
# JPEG q95 4:4:4 instead of PNG: sources are already lossy video/JPEG, so the extra loss is
# negligible while files are ~8x smaller and much faster to write and read.
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
THUMB_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 85]
MAX_PENDING_WRITES = 6  # bounds memory held by full-res frames waiting to be encoded

Progress = Callable[[float], None]


def fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """(w, h) scaled so the longer side is at most `long_side`, keeping aspect."""
    scale = min(1.0, long_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dup_thumb(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (DUP_SIZE, DUP_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def rotate_upright(rgb: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Apply a display-matrix rotation (PyAV `frame.rotation`, degrees counter-clockwise)."""
    k = round(rotation_deg / 90) % 4
    return np.ascontiguousarray(np.rot90(rgb, k)) if k else rgb


@dataclass
class _Candidate:
    index: int
    score: float
    thumb: np.ndarray
    payload: Any


class WindowSelector:
    """Pick the sharpest item per window of the timeline, skipping near-duplicates.

    Feed items in order with `feed`; picks are passed to `emit` as soon as a window closes,
    so the caller only ever holds one candidate per window in memory.
    """

    def __init__(self, total: int, k: int, emit: Callable[[_Candidate], None]) -> None:
        self.total = max(total, 1)
        self.k = max(1, min(k, self.total))
        self.emit = emit
        self.dropped = 0
        self._window = -1
        self._best: _Candidate | None = None
        self._last_thumb: np.ndarray | None = None

    def feed(self, index: int, score: float, thumb: np.ndarray, payload: Any) -> None:
        window = min(index * self.k // self.total, self.k - 1)
        if window != self._window:
            self._flush()
            self._window = window
        if self._best is None or score > self._best.score:
            self._best = _Candidate(index, score, thumb, payload)

    def finish(self) -> None:
        self._flush()

    def _flush(self) -> None:
        best, self._best = self._best, None
        if best is None:
            return
        if self._last_thumb is not None and np.abs(best.thumb - self._last_thumb).mean() < DUP_THRESHOLD:
            self.dropped += 1
            return
        self._last_thumb = best.thumb
        self.emit(best)


class _FrameWriter:
    """Encodes kept frames on worker threads (cv2 releases the GIL) while decoding continues."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ingest-write")
        self._pending: list[Future] = []

    def add(self, rgb: np.ndarray, meta: dict[str, Any]) -> None:
        n = len(self.entries)
        entry = {"index": n, "file": f"{n:04d}.jpg", "thumb": f"thumbs/{n:04d}.jpg",
                 "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), **meta}
        self.entries.append(entry)
        for f in [f for f in self._pending if f.done()]:
            f.result()  # surface write errors early
        self._pending = [f for f in self._pending if not f.done()]
        while len(self._pending) >= MAX_PENDING_WRITES:
            self._pending.pop(0).result()
        self._pending.append(self._pool.submit(self._write, rgb, entry))

    def _write(self, rgb: np.ndarray, entry: dict[str, Any]) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _encode_to(self.out_dir / entry["file"], bgr, JPEG_PARAMS)
        tw, th = fit_size(bgr.shape[1], bgr.shape[0], THUMB_LONG_SIDE)
        _encode_to(self.out_dir / entry["thumb"], cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA), THUMB_PARAMS)

    def close(self) -> None:
        """Wait for all writes; re-raises the first write error."""
        try:
            for f in self._pending:
                f.result()
        finally:
            self._pool.shutdown(wait=True)


def _encode_to(path: Path, bgr: np.ndarray, params: list[int]) -> None:
    ok, buf = cv2.imencode(".jpg", bgr, params)  # imencode + write_bytes: cv2.imwrite fails on non-ASCII paths
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for {path.name}")
    path.write_bytes(buf.tobytes())


def _count_frames(path: Path) -> int:
    """Frame count from the header, else by demuxing packets (no decoding)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for packet in container.demux(stream) if packet.size)


def ingest_video(path: Path, out_dir: Path, k: int, progress: Progress | None = None) -> dict[str, Any]:
    """Select up to `k` frames from a video into `out_dir`. Returns the frames.json content."""
    total = _count_frames(path)
    writer = _FrameWriter(out_dir)
    info: dict[str, Any] = {"rotation": 0}

    def emit(c: _Candidate) -> None:
        frame = c.payload
        rgb = rotate_upright(frame.to_ndarray(format="rgb24"), info["rotation"])
        writer.add(rgb, {"source_index": c.index, "time": None if frame.time is None else round(frame.time, 4),
                         "sharpness": round(c.score, 2)})

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            selector = WindowSelector(total, k, emit)
            decoded = 0
            for decoded, frame in enumerate(container.decode(stream), start=1):
                if decoded == 1:
                    info["rotation"] = int(frame.rotation or 0)
                    sw, sh = fit_size(frame.width, frame.height, SCORE_LONG_SIDE)
                gray = frame.reformat(width=sw, height=sh, format="gray", interpolation="AREA").to_ndarray()
                selector.feed(decoded - 1, sharpness(gray), dup_thumb(gray), frame)
                if progress:
                    progress(min(decoded / max(total, 1), 0.99))
            selector.finish()
            fps = float(stream.average_rate) if stream.average_rate else None
    finally:
        writer.close()
    if not writer.entries:
        raise ValueError("the video contains no decodable frames")
    return {
        "version": INGEST_VERSION, "kind": "video", "sources": [path.name], "source_frames": decoded,
        "fps": fps, "rotation": info["rotation"], "dropped_duplicates": selector.dropped,
        "frames": writer.entries,
    }


def load_image_rgb(path: Path) -> np.ndarray:
    """Decode an image upright (EXIF orientation applied) as RGB uint8."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def ingest_images(paths: list[Path], out_dir: Path, k: int, progress: Progress | None = None) -> dict[str, Any]:
    """Select up to `k` photos (in name order) into `out_dir`. Returns the frames.json content."""
    paths = sorted(paths, key=lambda p: p.name)
    writer = _FrameWriter(out_dir)

    def emit(c: _Candidate) -> None:
        writer.add(c.payload, {"source_index": c.index, "source": paths[c.index].name, "time": None,
                               "sharpness": round(c.score, 2)})

    try:
        selector = WindowSelector(len(paths), k, emit)
        for i, path in enumerate(paths):
            rgb = load_image_rgb(path)
            sw, sh = fit_size(rgb.shape[1], rgb.shape[0], SCORE_LONG_SIDE)
            gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
            selector.feed(i, sharpness(gray), dup_thumb(gray), rgb)
            if progress:
                progress(min((i + 1) / len(paths), 0.99))
        selector.finish()
    finally:
        writer.close()
    return {
        "version": INGEST_VERSION, "kind": "images", "sources": [p.name for p in paths],
        "source_frames": len(paths), "fps": None, "rotation": 0, "dropped_duplicates": selector.dropped,
        "frames": writer.entries,
    }


def run(ctx: StageContext) -> dict[str, Any]:
    """Pipeline stage entry point. Settings: `preset` (draft|standard|high, default standard)."""
    preset = ctx.settings.get("preset", "standard")
    if preset not in PRESET_FRAMES:
        raise ValueError(f"unknown preset {preset!r}")
    k = PRESET_FRAMES[preset]
    files = sorted(p for p in ctx.paths.input.iterdir() if p.is_file())
    if not files:
        raise ValueError("no input files")
    out = ctx.paths.frames
    key = stage_key("ingest", files, {"k": k, "version": INGEST_VERSION})
    if (hit := cached_metrics(out, "ingest", key)) is not None:
        ctx.set_artifact("frames_json", "frames/frames.json")
        return {**hit, "cached": True}

    invalidate(out, "ingest")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if files[0].suffix.lower() in VIDEO_EXTS:
        frames = ingest_video(files[0], out, k, ctx.progress)
    else:
        frames = ingest_images(files, out, k, ctx.progress)
    (out / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")

    first = frames["frames"][0]
    metrics = {"frames": len(frames["frames"]), "source_frames": frames["source_frames"],
               "dropped_duplicates": frames["dropped_duplicates"], "width": first["width"],
               "height": first["height"]}
    outputs = [out / "frames.json"] + [out / e[n] for e in frames["frames"] for n in ("file", "thumb")]
    write_manifest(out, "ingest", key, outputs, metrics)
    ctx.set_artifact("frames_json", "frames/frames.json")
    return metrics
