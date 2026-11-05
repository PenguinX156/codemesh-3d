# 3-minute demo script

Record the screen (Windows: Win+Alt+R with Xbox Game Bar, or OBS). Upload to YouTube as Unlisted, paste the link in the form.
Have ready: the app at http://localhost:5180 (or the Modal URL), a finished mug job, and the kettle video.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:20 | Title / app home | "Making a 3D asset of a real object still means photogrammetry rigs or hours of cleanup. CodeMesh 3D turns an ordinary phone video into a game-ready model." |
| 0:20–0:45 | Drop the kettle or mug video, tick the disclosure, Upload | "I filmed this on an iPhone in 40 seconds, just walking around it. The app picks the sharpest frames at full 4K." |
| 0:45–1:10 | Select screen: type "black coffee mug", Find object | "I don't draw masks. One sentence, and SAM 3.1 tracks the object through every frame in a single video request." Show the mask preview. |
| 1:10–1:40 | Build 3D model; show stage progress | "MapAnything, Meta's open-weights model, recovers every camera and depth map at once, on a Modal GPU in seconds, or on my AMD laptop GPU." |
| 1:40–2:20 | Result: 3D viewer (rotate, wireframe, texture/normals), the Muse agent report, render-vs-photo images, quality grade | "Muse Spark is the modeling agent. It looked at the photos and the render comparisons, decided this is a hard-surface mug, and chose how to model it with tools. The pipeline measures every change against the real footage and keeps the best. The quality gate checks it is watertight, one piece, and how closely it matches." Mention the 0.28 → 0.82 jump. |
| 2:20–2:45 | Download GLB, drop it into Blender (or the three.js viewer) | "Out comes a GLB with UVs, a 4K texture and a baked normal map, ready for Blender, Unity or Unreal." |
| 2:45–3:00 | GitHub + Modal URL on screen | "Open source, runs locally or on Modal. Next: clean primitive modeling by Muse for every object type, and a phone capture app." |

Tips: keep the cursor slow, zoom the browser to 110%, and cut waiting time (stage progress) in the editor, or say "sped up" on screen.
