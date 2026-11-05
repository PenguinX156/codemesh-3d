"""Live: Muse reviews mug frames, writes build() code, sees the render, revises once.

uv run --project backend python bench/muse_tools/run_mug.py
"""

import functools
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from codemesh.agent import codegen, frames, meshapi
from codemesh.meta import muse

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[2]
SRC, JOB, OUT = ROOT / "jobs/cases/mug_f", ROOT / "jobs/cases/mug_code", Path(__file__).parent
if not JOB.exists():
    shutil.copytree(SRC, JOB)

cached = JOB / "agent" / "frames_review.json"
review = json.loads(cached.read_text()) if cached.exists() else frames.review_frames(JOB)
print("REVIEW", json.dumps({k: review[k] for k in ("geometry", "reference", "excluded_from_carving", "cost_usd")}))
for f in review["frames"]:
    if 25 <= f["index"] <= 39:
        print(f" #{f['index']}: shape={f['shape_use']} tex={f['texture_use']} unique={f['unique_viewpoint']} "
              f"role={f['role']} - {f['reasons']}")
spent = review["cost_usd"]

ev = meshapi.Evidence(JOB)
refs = sorted(set(review["reference"][:4]) | {f["index"] for f in review["frames"] if f["role"] == "geometry"}
              .intersection(range(0, 25, 6)))[:6]
images = [frames.tile(JOB / f"frames/{i:04d}.jpg", JOB / f"masks/{i:04d}.png", f"#{i}") for i in refs]
API_DOC = """api.box(w,h,d) centred; api.cylinder(r,h) base y=0 along +Y; api.capsule(r,h); api.torus(major,minor) hole along Y;
api.revolve(profile=[(r,y),...] bottom->top, segments=64, wall=None, floor=None) lathe about +Y (wall -> hollow vessel, floor thickness);
api.sweep(path=[(x,y,z),...], radius=float or (rx,ry), closed=False) tube; api.extrude(polygon2d=[(x,z)...], depth) along +Y;
api.translate(m,x,y,z); api.rotate(m,degrees,axis=(0,1,0),about=(0,0,0)); api.scale(m,sx,sy,sz); api.mirror(m,axis);
api.array(m,count,offset,degrees,axis)->list; api.combine(list)->mesh; api.pi, api.sin, api.cos, api.linspace(a,b,n).
evidence.bbox {min,max,size}; evidence.height/base_y/top_y; evidence.axis [x,z] of vertical axis;
evidence.radius_profile(n)->[(y,r)]; evidence.residual(body, tol)->hull points outside body; evidence.opening().
Sandbox: no imports, no names starting with '_', only builtins range len min max abs float int list dict tuple zip enumerate sum round."""
SCHEMA = {"type": "object", "properties": {"object": {"type": "string"}, "plan": {"type": "string"},
                                           "code": {"type": "string"}}, "required": ["object", "plan", "code"]}
evidence = ev.summary()
body = meshapi.cylinder(min(ev.bbox["size"][0], ev.bbox["size"][2]) / 2 * .8, ev.height)
evidence["residual_outside_central_cylinder_sample"] = ev.residual(
    meshapi.translate(body, ev.axis[0], ev.base_y, ev.axis[1]))[:40]
prompt = f"""Model this object as a clean low-poly but faithful 3D mesh by writing Python code.
Images: reference frames #{refs} (green = SAM mask). The frame review said: {json.dumps(review['frames'], separators=(',', ':'))[:3500]}
Evidence from the visual hull (asset frame, +Y up, metres if metric): {json.dumps(evidence)}
Note: the visual hull fills concavities (e.g. a cup interior) - use the photos for those.
API:
{API_DOC}
Write `def build(api, evidence):` returning api.combine([...]) of closed bodies positioned in the evidence frame
(centre on evidence.axis, base at evidence.base_y). Use evidence values, not hard-coded guesses, where possible."""
msgs = [{"role": "user", "content": prompt}]
result = None
for attempt in range(2):
    reply = muse.chat(msgs, images=images, response_schema=SCHEMA, max_side=768, max_tokens=16000)
    spent += reply.usage.cost_usd
    code = reply.parsed["code"]
    print(f"--- attempt {attempt} ({reply.usage.cost_usd:.4f} USD): {reply.parsed['object']}: {reply.parsed['plan']}\n{code}")
    result = codegen.build_with_code(JOB, code, name=f"code_{attempt}")
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "metrics"}),
          json.dumps({k: result.get("metrics", {}).get(k) for k in ("iou_mean", "iou_min")}))
    if attempt or spent > .1:
        break
    feedback = (f"Result: {json.dumps({k: v for k, v in result.items() if k not in ('mesh_path', 'preview_path')})}. "
                "The attached image shows photos with the SAM mask (green) and your mesh silhouette (red). "
                "Revise the code to match the photos better (fix errors first). Return the full code.")
    images = [cv2.imread(result["preview_path"])[:, :, ::-1].copy()] if "preview_path" in result else []
    msgs = msgs + [{"role": "assistant", "content": reply.text}, {"role": "user", "content": feedback}]

best = max((p for p in JOB.glob("agent/code_*.json")), default=None)
if "preview_path" in result:
    shutil.copy(result["preview_path"], OUT / "mug_code_preview.png")
(OUT / "mug_code_result.json").write_text(json.dumps({"result": result, "code": code, "spent_usd": spent}, indent=1))
print("SPENT", round(spent, 4))
