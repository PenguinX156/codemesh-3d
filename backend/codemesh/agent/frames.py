"""Muse reviews every ingested frame from numbered contact sheets (photo + SAM mask outline).

The automatic focal-consistency check is only a hint in the prompt. Roles: `geometry` (carve with
it), `reference` (keep for Muse to look at: unique viewpoints or usable texture), `ignore`. A frame
leaves geometry carving only when Muse rates shape_use=bad. Writes agent/frames_review.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from codemesh.meta import muse

TILES, TILE = 12, 384  # 4 x 3 tiles per sheet
RATING = {"type": "string", "enum": ["good", "ok", "bad"]}
SCHEMA = {
    "type": "object",
    "properties": {"frames": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "shape_use": RATING, "texture_use": RATING,
                       "unique_viewpoint": {"type": "boolean"}, "reasons": {"type": "string"}},
        "required": ["index", "shape_use", "texture_use", "unique_viewpoint", "reasons"]}}},
    "required": ["frames"],
}
PROMPT = """You are reviewing every frame of a turntable/walk-around capture of one object before 3D
reconstruction. Each tile shows frame #N with the SAM segmentation mask outlined in green. Camera
notes from the automatic pose estimate (a HINT only, it is often wrong for unusual views):
{notes}
For EVERY frame number listed, judge:
- shape_use: is the green mask a correct, complete silhouette of the whole object? (bad = mask wrong,
  partial, includes background/hands, object cut off, heavy blur)
- texture_use: are the colours/details sharp and well lit?
- unique_viewpoint: true if this frame sees the object from a direction few other frames cover
  (e.g. top-down, underside, close-up of a detail). Never call a frame bad only because its camera is
  marked suspicious; judge the image.
- reasons: a few words (blur, mask partial, occluded, top view, ...).
Frames: {indices}"""


def _frames(job_dir: Path) -> list[tuple[int, Path, Path]]:
    out = []
    for photo in sorted((job_dir / "frames").glob("*.jpg")) + sorted((job_dir / "frames").glob("*.png")):
        if photo.stem.isdigit():
            out.append((int(photo.stem), photo, job_dir / "masks" / f"{photo.stem}.png"))
    return sorted(out)


def camera_notes(job_dir: Path) -> dict[int, str]:
    """Per frame: estimated focal and its deviation from the median, or 'no camera' if geometry dropped it."""
    path = job_dir / "geometry" / "cameras.json"
    cams = json.loads(path.read_text())["cameras"] if path.exists() else []
    fx = {c["index"]: c["K"][0][0] for c in cams}
    med = float(np.median(list(fx.values()))) if fx else 0.0
    notes = {}
    for index, _, _ in _frames(job_dir):
        if index not in fx:
            notes[index] = "no camera (dropped by the pose consistency filter)"
        else:
            dev = abs(fx[index] / med - 1)
            notes[index] = f"focal {fx[index]:.0f}px ({dev:.0%} off median)" + (" SUSPICIOUS" if dev > .1 else "")
    return notes


def tile(photo: Path, mask: Path, label: str) -> np.ndarray:
    img = cv2.imread(str(photo))
    s = TILE / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) if mask.exists() else None
    if m is not None:
        m = cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    canvas = np.zeros((TILE, TILE, 3), np.uint8)
    y, x = (TILE - img.shape[0]) // 2, (TILE - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.rectangle(canvas, (0, 0), (90, 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (4, 26), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2)
    return canvas[:, :, ::-1]  # RGB


def contact_sheets(job_dir: Path) -> list[tuple[list[int], np.ndarray]]:
    frames = _frames(job_dir)
    sheets = []
    for k in range(0, len(frames), TILES):
        chunk = frames[k:k + TILES]
        tiles = [tile(p, m, f"#{i}") for i, p, m in chunk]
        tiles += [np.zeros_like(tiles[0])] * (TILES - len(tiles))
        rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(TILES // 4)]
        sheets.append(([i for i, _, _ in chunk], np.vstack(rows)))
    return sheets


def role(v: dict) -> str:
    if v["shape_use"] != "bad":
        return "geometry"
    return "reference" if v["unique_viewpoint"] or v["texture_use"] != "bad" else "ignore"


def review_frames(job_dir: Path, chat=muse.chat) -> dict:
    """One structured Muse call per contact sheet; returns and writes agent/frames_review.json."""
    job_dir = Path(job_dir)
    notes = camera_notes(job_dir)
    verdicts, cost, sheets = {}, 0.0, contact_sheets(job_dir)
    for indices, sheet in sheets:
        text = PROMPT.format(notes="\n".join(f"#{i}: {notes[i]}" for i in indices), indices=indices)
        reply = chat([{"role": "user", "content": text}], images=[sheet], response_schema=SCHEMA,
                     max_side=1536, max_tokens=12000)
        cost += reply.usage.cost_usd
        for v in (reply.parsed or {}).get("frames", []):
            if v["index"] in indices:
                verdicts[v["index"]] = {**v, "camera_hint": notes[v["index"]], "role": role(v)}
    for i in notes:  # a frame Muse skipped stays usable: never silently drop evidence
        verdicts.setdefault(i, {"index": i, "shape_use": "ok", "texture_use": "ok", "unique_viewpoint": False,
                                "reasons": "not rated", "camera_hint": notes[i], "role": "geometry"})
    frames = [verdicts[i] for i in sorted(verdicts)]
    review = {"frames": frames, "cost_usd": cost, "sheets": len(sheets),
              "geometry": [f["index"] for f in frames if f["role"] == "geometry"],
              "reference": [f["index"] for f in frames if f["role"] == "reference" or f["unique_viewpoint"]],
              "excluded_from_carving": [f["index"] for f in frames if f["shape_use"] == "bad"]}
    out = job_dir / "agent"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames_review.json").write_text(json.dumps(review, indent=1))
    return review
