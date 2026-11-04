"""Live smoke test of the Meta API clients (SAM 3.1 + Muse Spark 1.3 Contributor).

Uses only the non-personal VGGT kitchen example images (a LEGO bulldozer on a table).
Spend is tiny: one 50-frame-minimum video request per prompt (~$0.01), two image-sized Muse calls.

Run from the repo root:
    uv run --project backend python bench/phase0/meta_smoke.py [--skip-sam] [--skip-muse]
Outputs go to bench/phase0/out/sam/ (overlay PNGs + the clip) and bench/phase0/out/muse.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from codemesh.meta import muse, sam  # noqa: E402

IMAGES = sorted((REPO / "third_party/vggt/examples/kitchen/images").glob("*.png"))
OUT = REPO / "bench/phase0/out"
LEGO_CLICK = sam.Point(0.55, 0.44)  # on the bulldozer body in frame 0 (normalized)
FPS = 10
HOLD = 2  # frames per photo; 25 photos x 2 = 50 frames = the API's per-request billing minimum


def make_clip(path: Path) -> int:
    """Write the kitchen photos as a short faststart H.264 MP4; returns the frame count."""
    first = cv2.imread(str(IMAGES[0]))
    h, w = first.shape[:2]
    w, h = w - w % 2, h - h % 2  # yuv420p needs even sizes
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "18"}
        for img_path in IMAGES:
            rgb = cv2.cvtColor(cv2.imread(str(img_path))[:h, :w], cv2.COLOR_BGR2RGB)
            for _ in range(HOLD):
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                out.mux(stream.encode(frame))
                count += 1
        out.mux(stream.encode())
    return count


def overlay(frame_rgb: np.ndarray, m: sam.ObjectMask, color=(255, 0, 80)) -> np.ndarray:
    img = frame_rgb.copy()
    img[m.mask] = (0.5 * img[m.mask] + 0.5 * np.array(color)).astype(np.uint8)
    x1, y1, x2, y2 = m.box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def decoded_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    with av.open(str(path)) as c:
        return {i: f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0)) if i in wanted}


def report(tag: str, res: sam.SegResult, clip: Path, out_dir: Path) -> None:
    n = len(res.masks)
    area = [m.mask.mean() for m in res.masks.values()]
    scores = [m.score for m in res.masks.values() if m.score is not None]
    print(
        f"[{tag}] {res.seconds:.1f}s total, selected id={res.selected_id}, candidates on prompt frame="
        f"{len(res.candidates)}, frames with mask={n}, mean coverage={np.mean(area) * 100:.1f}% of frame"
        if n
        else f"[{tag}] {res.seconds:.1f}s, NO MATCH",
        f"timings={ {k: round(v, 1) for k, v in res.timings.items()} } usage={res.usage}",
    )
    if scores:
        print(f"[{tag}] confidence min/mean={min(scores):.2f}/{np.mean(scores):.2f}")
    pick = sorted(res.masks)[:: max(1, n // 4)][:5] if n else []
    frames = decoded_frames(clip, set(pick))
    for i in pick:
        cv2.imwrite(str(out_dir / f"{tag}_f{i:03d}.png"), cv2.cvtColor(overlay(frames[i], res.masks[i]), cv2.COLOR_RGB2BGR))


def run_sam() -> None:
    out_dir = OUT / "sam"
    clip = out_dir / "kitchen.mp4"
    n = make_clip(clip)
    print(f"clip: {n} frames, {clip.stat().st_size / 1e3:.0f} kB")
    with sam.MetaClient() as client:
        point = sam.segment_video(clip, points=[LEGO_CLICK], prompt_frame=0, client=client)
        report("video_point", point, clip, out_dir)
        text = sam.segment_video(clip, text="yellow toy bulldozer", points=[LEGO_CLICK], client=client)
        report("video_text", text, clip, out_dir)
        imgs = sam.segment_images(IMAGES[:3], text="yellow toy bulldozer", client=client)
        for i, r in enumerate(imgs):
            print(f"[images] #{i}: {r.seconds:.1f}s candidates={len(r.candidates)} usage={r.usage}")
    (OUT / "sam" / "video_point_raw.txt").write_text(point.raw_text, encoding="utf8")


SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "surface_type": {"type": "string", "enum": ["hard_surface", "organic", "mixed"]},
        "symmetry": {"type": "string", "enum": ["none", "mirror", "radial", "approximate_mirror"]},
        "thin_parts": {"type": "array", "items": {"type": "string"}},
        "suggested_triangle_budget": {"type": "integer"},
    },
    "required": ["category", "surface_type", "symmetry", "thin_parts", "suggested_triangle_budget"],
}


def run_muse() -> None:
    reply = muse.chat(
        [{"role": "user", "content": "These are two views of one object. Describe it for a 3D reconstruction pipeline."}],
        images=[IMAGES[0], IMAGES[12]],
        response_schema=SCHEMA,
    )
    u = reply.usage
    print(f"[muse] {reply.seconds:.1f}s finish={reply.finish_reason} parsed={json.dumps(reply.parsed)}")
    print(f"[muse] tokens prompt={u.prompt_tokens} completion={u.completion_tokens} (reasoning={u.reasoning_tokens}) cost=${u.cost_usd:.5f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "muse.json").write_text(json.dumps({"parsed": reply.parsed, "usage": vars(u), "seconds": reply.seconds}, indent=2))


if __name__ == "__main__":  # guard required: sam.py decodes masks in a process pool (Windows spawn)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-muse", action="store_true")
    args = ap.parse_args()
    if not args.skip_sam:
        run_sam()
    if not args.skip_muse:
        run_muse()
