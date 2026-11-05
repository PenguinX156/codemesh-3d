"""One-off: kitchen MP4 for upload tests + a textured sample.glb for the viewer."""
import glob, av, numpy as np, trimesh
from PIL import Image
files = sorted(glob.glob("third_party/vggt/examples/kitchen/images/*.png"))[:8]
first = Image.open(files[0]).convert("RGB")
w, h = (first.width // 2) * 2, (first.height // 2) * 2
c = av.open("bench/ui/kitchen.mp4", "w")
s = c.add_stream("libx264", rate=2); s.width, s.height, s.pix_fmt = w, h, "yuv420p"
for f in files:
    img = Image.open(f).convert("RGB").resize((w, h))
    for _ in range(4):
        for p in s.encode(av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")): c.mux(p)
for p in s.encode(): c.mux(p)
c.close()
m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=(np.abs(m.vertices) / 0.5 * 255).astype(np.uint8))
m.export("frontend/public/showcase/sample.glb")
print(w, h, len(m.faces))
