# Capture guide (test set for tonight)

Put files in `data/captures/` with the names below (they are gitignored). Add `data/captures/notes.txt` with one line per file: name, object, real height in cm.
No people, faces, documents or screens with personal info (images go to Meta's Contributor tier).

## Shots, in priority order (stop wherever time runs out)

| # | File | Subject | Tests |
|---|---|---|---|
| 1 | `01_mug_walkaround.mp4` | Patterned/printed mug **with handle** | Hard-surface shape, a hole (handle), texture |
| 2 | `02_shoe_walkaround.mp4` | Sneaker or shoe | Mixed curved + structured shape, fine texture detail |
| 3 | `03_plant_walkaround.mp4` | Potted plant (or a stuffed toy / bread loaf / large fruit if no plant) | Irregular/organic shape, thin parts |
| 4 | `04_mug_turntable.mp4` | Same mug on a plate or lazy Susan you rotate by hand; phone stays still | Turntable mode |
| 5 | `05_shoe_photos/` (30–40 JPGs) | Same shoe, photos instead of video | Photo-set upload |
| 3b | `07_mug_quick.mp4` | Same mug, ONE casual loop in 10–20 s at ~30° | Quick single-pass capture (must still work; top/inside will be estimated) |
| 6 | `06_hard_case.mp4` (optional) | Shiny metal or glossy object (kettle, can) | Known hard case: reflections |

Skip outdoor scenes tonight: at night the frames are dark and noisy, which hurts both segmentation and geometry.

## How to film

- **Light:** indoors, all room lights on, no flash, no window or lamp behind the object. Avoid one hard light that throws strong shadows.
- **Camera:** main 1× lens (not ultra-wide), 4K 30 fps (1080p is fine for the Modal tests: smaller upload). Stabilization on. **Turn HDR / Dolby Vision video off** (iPhone: Settings → Camera → Record Video → HDR Video off). No portrait/cinematic/action modes. Tap and hold on the object to lock focus and exposure.
- **Setup:** object alone on a table with a **textured surface under it** (newspaper, patterned cloth, magazine) and normal room clutter behind it — texture around the object helps camera tracking. Leave room to walk all the way around. Objects about 10–60 cm tall.
- **Path (walk-around):** 30–45 s. Keep the object centered and filling about half the frame, at a steady 0.5–1 m distance, no zoom. Walk slowly: one full loop looking down ~30°, then a second loop lower (~10°), then 3 s from above. No sudden pans.
- **Turntable:** phone fixed (prop it up), object in the middle of the plate, rotate slowly through 1–2 full turns in ~30 s, background static.
- **Photos:** 30–40 photos around the object at two heights, each overlapping the previous by most of the frame.
- **Measure** one real dimension (usually height) and write it in `notes.txt` — used for the real-world scale feature.

## Transfer

AirDrop / USB / Google Drive download into `C:\Users\Filip\Dev\Projects\Meshify Capture\data\captures\`.
