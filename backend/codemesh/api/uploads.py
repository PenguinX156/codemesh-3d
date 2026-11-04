"""Upload validation (trust boundary): file counts, size limit, safe names, content decoding.

The content check decodes the data (PyAV for videos, Pillow for images); extensions only
decide which decoder is tried, never whether a file is accepted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import BinaryIO, Protocol

import av
from PIL import Image

from codemesh.jobs.models import JobInput
from codemesh.pipeline.ingest import IMAGE_EXTS, VIDEO_EXTS

# shortcut: read here because config.py is owned by another lane; move it there later.
DEFAULT_MAX_UPLOAD_BYTES = int(os.getenv("CODEMESH_MAX_UPLOAD_BYTES", str(4 * 1024**3)))
MIN_IMAGES, MAX_IMAGES = 2, 300
IMAGE_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP", "HEIF"})  # MPO = multi-picture JPEG from phones
_CHUNK = 1024 * 1024


class UploadError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class UploadedFile(Protocol):
    filename: str | None
    file: BinaryIO


def safe_filename(name: str | None, index: int) -> str:
    """'<index>_<name>.<ext>' with only [A-Za-z0-9._-]; directory parts are dropped."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:80] or "file"
    ext = ext.lower() if re.fullmatch(r"\.[a-z0-9]{1,5}", ext.lower()) else ""
    return f"{index:03d}_{stem}{ext}"


def _ext(name: str | None) -> str:
    return os.path.splitext(name or "")[1].lower()


def _verify_video(path: Path) -> None:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise UploadError(415, f"{path.name}: no video stream")
            decoded = 0
            for _ in container.decode(video=0):
                decoded += 1
                if decoded == 2:  # a single still image would also "decode" as one frame
                    return
    except UploadError:
        raise
    except Exception as exc:  # any decoder failure means the content is not an acceptable video
        raise UploadError(415, f"{path.name}: not a decodable video ({type(exc).__name__})") from exc
    raise UploadError(415, f"{path.name}: video has fewer than 2 frames")


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as im:
            if im.format not in IMAGE_FORMATS:
                raise UploadError(415, f"{path.name}: unsupported image format {im.format}")
            im.draft("RGB", (1024, 1024))  # JPEG: decode at reduced scale, still checks the whole stream
            im.load()
    except UploadError:
        raise
    except Exception as exc:  # includes Pillow's DecompressionBombError
        raise UploadError(415, f"{path.name}: not a decodable image ({type(exc).__name__})") from exc


def save_uploads(files: list[UploadedFile], dest: Path, max_bytes: int) -> JobInput:
    """Validate and copy uploads into `dest`. Raises UploadError; the caller cleans up `dest`."""
    if not files:
        raise UploadError(400, "no files uploaded (multipart field 'files[]')")
    exts = [_ext(f.filename) for f in files]
    if any(e in VIDEO_EXTS for e in exts):
        if len(files) != 1:
            raise UploadError(400, "upload exactly one video, or 2-300 images")
        kind, verify = "video", _verify_video
    elif all(e in IMAGE_EXTS for e in exts):
        if not MIN_IMAGES <= len(files) <= MAX_IMAGES:
            raise UploadError(400, f"upload {MIN_IMAGES}-{MAX_IMAGES} images (got {len(files)})")
        kind, verify = "images", _verify_image
    else:
        allowed = ", ".join(sorted(VIDEO_EXTS | IMAGE_EXTS))
        raise UploadError(415, f"unsupported file type; allowed: {allowed}")

    dest.mkdir(parents=True, exist_ok=True)
    total, names = 0, []
    for i, f in enumerate(files):
        name = safe_filename(f.filename, i)
        with open(dest / name, "wb") as out:
            while chunk := f.file.read(_CHUNK):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadError(413, f"upload exceeds the {max_bytes} byte limit")
                out.write(chunk)
        names.append(name)
    for name in names:
        verify(dest / name)
    return JobInput(kind=kind, files=names, bytes=total)
