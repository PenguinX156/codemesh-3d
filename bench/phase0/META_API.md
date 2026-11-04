# Meta Model API: SAM 3.1 and Muse Spark 1.3 Contributor

Code: `backend/codemesh/meta/` (`client.py`, `sam.py`, `muse.py`, `_maskdec.py`). Live check: `uv run --project backend python bench/phase0/meta_smoke.py`. Docs: https://dev.meta.ai/docs (overview, `sam/segmenting`, `sam/reading-segmentation`, `sam/client-libraries`, `image-understanding`, `structured-output`, `reasoning`, `pricing-rate-limits`).

## Basics

- Base `https://api.meta.ai/v1`, `Authorization: Bearer <META_API_KEY>`. Same key for both models. Keys look like `LLM_<id>_<secret>`. OpenAI-SDK compatible, but we use plain `httpx` (one dependency less, full control over retries/SSE).
- Rate limits are per team: `muse-spark-1.3-contributor` 150 RPM / 3M TPM, `sam-3.1` 3000 RPM. Retry 429/5xx with exponential backoff + jitter, honour `Retry-After` (done in `MetaClient`). A 400 is permanent for that request: never retry unchanged.
- **Contributor mode**: prompts, files and outputs may be used by Meta for training. The UI needs a disclosure; send only non-personal content.

## SAM 3.1 (`POST /v1/responses`, model `sam-3.1`)

Request (streamed, SSE `data:` events; `stream: true`):

```json
{"model": "sam-3.1", "stream": true,
 "metadata": {"mask_encoding": "one_bit", "include_confidence": "true", "propagation_direction": "both"},
 "input": [{"type": "message", "role": "user", "content": [
   {"type": "input_text", "text": "<0f>1<|point;x=430;y=230;w=778;h=520|>"},
   {"type": "input_video", "file_id": "file-..."}   // or {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
 ]}]}
```

- **Prompt** (`input_text`) is either a concept noun phrase (`"yellow toy bulldozer"`) or object prompts (`<Nf>id<|box;x1;y1;x2;y2;w;h|>`, `<|point;x;y;w;h|>`, `-<|point..|>` for negative), never both. **Coordinates are media pixels, not normalized**; `w`/`h` must match the media aspect ratio (else 400); x2/y2 are inclusive; points must be inside the frame. Up to 16 objects. Build them with `meta_sam_parser.build_object_prompt` (we do).
- **Video**: upload with `POST /v1/files` (multipart, `purpose=user_data`) then pass `file_id` (inline data URLs also work for images). MP4/MOV with H.264/H.265/MPEG-4; VP9/WebM rejected; up to 15,000 frames. Non-faststart MP4s are remuxed server-side (lossless). Frame N in the answer is the Nth decoded frame. Rotated videos: send the displayed (post-rotation) size.
- `propagation_direction` (`both` default | `forward` | `backward`) only matters for object prompts. `backward` needs a prompt on a frame > 0. Fewer tracked frames = cheaper.
- **Response**: SSE events `response.created`, `response.output_item.added`, `response.output_text.delta` (one per frame, text = one line), `response.output_text.done`, `response.completed` (carries `usage`). Zero matches = completes with empty text, not an error. Usage reports `video_frames_processed` / `images_processed` (tokens are 0).
- **Output line**: `<Nf>id<|box;x1;y1;x2;y2;w;h;c=..|><|mask;x=0;y=0;c=..;data=H,W,<enc>payload|>,id<|box..|><|mask..|>`.
  - Frames are explicit and not necessarily ascending (with `both`, frames before the prompt frame come last, descending). Frames with no object emit no line. Ids are stable per object but sparse, never use line position.
  - The mask raster is **exactly the inclusive box** (W = x2-x1+1, H = y2-y1+1; verified on real data), so a full-resolution mask = raster pasted at (x1, y1) into an empty frame.
  - Encoding marker `~` lossless (default), `!` one_bit. Payload is base85 with a custom range coder; the marker is part of the payload string. Always request `one_bit` (measured: 574 B vs 45,793 B for the same 409x298 mask, and decode 0.37 s vs 0.51 s).
  - `c` = confidence 0..1 (needs `include_confidence: "true"`; metadata values are strings).
- **Decoding** needs the official `meta-sam-parser` (PyPI, pure Python, SAM License, see `deps`). Pure-Python decode costs about 0.2-0.45 s per 100k-pixel mask, so `sam.to_object_masks` uses a process pool above 5 masks (50 masks: about 4 s when the machine is idle, 12 s when other jobs were running, versus ~22 s serial). Workers import only `_maskdec.py` so they start fast.

### Semantics we implemented (`sam.py`)

- `segment_video(path, points=[Point(x,y,label)], text=..., prompt_frame=..., direction=..., frame_indices=...)` and `segment_images(images, points=, text=)` take **normalized** 0..1 points and convert to pixels (clamped inside the frame).
- Points only -> object prompt (id 1). Text only -> concept prompt. Both -> concept prompt, and the clicked point picks the match (`choose_object`: contains the click, highest confidence wins, then larger area; no match under the click -> highest confidence).
- Result `SegResult`: `masks[frame]` = full-resolution `bool` array + inclusive box + confidence for the selected object, `candidates` (all matches on the prompt frame), `usage`, `timings`.

### Measured (real calls, 25 kitchen photos x2 = 50-frame 778x520 clip, LEGO bulldozer)

| Call | Result | Latency |
|---|---|---|
| image, 1 point, one_bit | 1 object, box 409x298, conf 0.93 | 1.9 s |
| image, text concept (3 images, 4 threads) | 1 match each | 1.8-2.9 s each |
| video, 1 point, 50 frames | mask on all 50 frames, mean 12.4% of the frame, conf min/mean 0.91/0.97 | 26 s total = upload 2.9 + API 11.1 + decode 12.2 (decode ~4 s when idle) |
| video, text + point (concept, point picks the match), 50 frames | 50/50 frames, 12.2%, conf 0.93 | 22.6 s (upload 3.5, API 7.3, decode 11.7) |

Visual check of overlays (`bench/phase0/out/sam/*.png`): masks are tight around the bulldozer, tracks and bucket included, across very different viewpoints (side, front, 3/4). Cost: video $0.20 / 1000 frames with a 50-frame minimum, images $2.50 / 1000 (about $0.01 per clip, $0.0025 per photo).

## Muse Spark 1.3 Contributor (`POST /v1/chat/completions`)

We use Chat Completions (stateless, standard tool loop); the Responses API is the one Meta recommends for multi-step agents and works too (`reasoning.effort`, `text.format`).

- `model: "muse-spark-1.3-contributor"`, `reasoning_effort: "high"` (top-level on Chat Completions; `"none"` unsupported). Reasoning tokens count towards `max_tokens`, so keep it large (we default to 8000).
- Images: content part `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` (Chat Completions form; Responses uses a plain string). Inline limit 50 MB per image. We downscale to 1280 px and JPEG q90 to keep prompt tokens low (2 images ~ 1000 prompt tokens).
- Structured output: `response_format: {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`; decoding is constrained (always valid JSON). No recursive schemas.
- Tools: standard `tools: [{"type": "function", "function": {...}}]`, `tool_calls` on the assistant message, answer with `role: "tool"` + `tool_call_id`; return the full assistant message first. Only `tool_choice: "auto"` is supported.
- Usage: `usage.prompt_tokens`, `completion_tokens`, `completion_tokens_details.reasoning_tokens`, `prompt_tokens_details.cached_tokens`. Contributor prices per 1M tokens: input $0.10, cached $0.002, output $0.20 (`muse.Usage.cost_usd`).

Measured (2 kitchen photos, JSON-schema description request: category, hard_surface/organic, symmetry, thin parts, triangle budget):

| Run | Latency | prompt / completion (reasoning) tokens | Cost |
|---|---|---|---|
| 1 | 17.9 s | 1002 / 1288 (1211) | $0.00036 |
| 2 | 15.6 s | 1002 / 1383 (1284) | $0.00038 |

Both runs answered correctly: `lego technic tracked loader bulldozer`, `hard_surface`, mirror / approximate mirror symmetry, thin parts (track belts, bucket teeth, hydraulic rods, steering wheels, roll-cage struts), budget 80-100k triangles. High reasoning dominates latency (~90% of completion tokens are reasoning); use `reasoning="low"` for cheap classification calls.

## Gotchas

1. SAM coordinates are pixels with matching `w`/`h`; our API contract uses normalized 0..1, so convert at the edge (done in `sam.py`).
2. The clip's frame count is billed with a 50-frame minimum: tracking a 24-frame set costs the same as 50. Tiny clips are cheap, but do not call per-frame video requests.
3. Mask decode is the pipeline bottleneck on our side, not the API. Possible future work: a native (Rust/C) decoder, or only decode frames that survive frame selection (`frame_indices`).
4. Results vary slightly between identical runs (docs say so); do not hard-assert pixel counts on live output.
5. `multiprocessing` on Windows spawns: scripts calling `segment_video` with many masks need an `if __name__ == "__main__":` guard (uvicorn workers are fine).
6. SAM output text deltas end with `\n` and arrive as separate events; join them before parsing (blank lines are ignored by `parse_output`).
7. Total spend for developing and testing this lane: about $0.08 (6 video requests of 50 frames, ~9 images, 2 Muse calls).
