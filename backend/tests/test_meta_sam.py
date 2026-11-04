"""SAM client: parser + mask decoding on a captured real response, selection logic, mocked request."""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from codemesh.meta import sam
from codemesh.meta.client import MetaClient

FIXTURE = (Path(__file__).parent / "fixtures" / "sam_video_onebit.txt").read_text()  # real sam-3.1 output
W, H = 778, 520  # frame size in the fixture (kitchen clip, LEGO bulldozer)


def test_parse_real_response():
    recs = sam.parse_output(FIXTURE)
    assert [(r.frame, r.object_id) for r in recs] == [(0, 1), (1, 1), (24, 1)]  # explicit frame markers
    r = recs[0]
    assert r.box == (273, 63, 681, 360) and r.frame_size == (W, H)
    assert r.score == pytest.approx(0.906, abs=1e-3)
    assert r.mask.encoding == "one_bit" and (r.mask.width, r.mask.height) == (409, 298)  # inclusive box size


def test_decode_real_masks_to_full_resolution():
    recs = sam.parse_output(FIXTURE)
    masks = sam.to_object_masks(recs)
    for m in masks:
        assert m.mask.shape == (H, W) and m.mask.dtype == bool
        x1, y1, x2, y2 = m.box
        outside = m.mask.copy()
        outside[y1 : y2 + 1, x1 : x2 + 1] = False
        assert not outside.any()  # nothing outside the box
    assert [int(m.mask.sum()) for m in masks] == [63022, 62583, 39086]  # golden values from the live decode
    assert masks[0].contains(430, 230)  # the bulldozer body (the click used in the smoke test)
    assert not masks[0].contains(5, 5)


def test_parse_handles_sparse_ids_skipped_frames_and_missing_confidence():
    def box(x: int) -> str:
        return f"<|box;x1={x};y1=0;x2={x + 3};y2=1;w=10;h=10|>"

    mask = "<|mask;x=0;y=0;data=2,4,!AAAA|>"
    text = f"<0f>0{box(0)}{mask},2{box(5)}{mask}\n\n<3f>2{box(5)}{mask}\n"  # frames 1-2 absent, id 1 absent
    recs = sam.parse_output(text)
    assert [(r.frame, r.object_id, r.score) for r in recs] == [(0, 0, None), (0, 2, None), (3, 2, None)]


def test_parse_empty_output_is_no_match():
    assert sam.parse_output("") == []


def obj(oid, score, region, frame=0):
    m = np.zeros((10, 10), bool)
    y1, y2, x1, x2 = region
    m[y1:y2, x1:x2] = True
    return sam.ObjectMask(oid, frame, m, (x1, y1, x2 - 1, y2 - 1), score)


def test_choose_prefers_object_under_click_over_higher_score():
    a, b = obj(0, 0.99, (0, 4, 0, 4)), obj(1, 0.60, (5, 9, 5, 9))
    assert sam.choose_object([a, b], (6, 6)).object_id == 1
    assert sam.choose_object([a, b], (9, 0)).object_id == 0  # click on neither -> highest score
    assert sam.choose_object([a, b]).object_id == 0
    assert sam.choose_object([], (1, 1)) is None


def test_choose_among_overlapping_uses_score_then_area():
    big, small = obj(0, None, (0, 8, 0, 8)), obj(1, None, (2, 4, 2, 4))
    assert sam.choose_object([small, big], (3, 3)).object_id == 0  # no scores: larger mask
    assert sam.choose_object([obj(0, 0.5, (0, 8, 0, 8)), obj(1, 0.9, (2, 4, 2, 4))], (3, 3)).object_id == 1


def test_prompt_text_points_vs_text():
    p = sam._prompt_text([sam.Point(0.5, 0.25), sam.Point(1.0, 1.0, 0)], None, (200, 100), 7)
    assert p == "<7f>1<|point;x=100;y=25;w=200;h=100|>-<|point;x=199;y=99;w=200;h=100|>"  # clamped, negative marked
    assert sam._prompt_text([sam.Point(0.5, 0.5)], " red bus ", (200, 100), 0) == "red bus"  # text wins
    with pytest.raises(ValueError):
        sam._prompt_text([], None, (200, 100), 0)


def test_segment_images_end_to_end_with_mocked_api():
    line = FIXTURE.splitlines()[0]
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        events = [
            {"type": "response.output_text.delta", "delta": line + "\n"},
            {"type": "response.completed", "response": {"usage": {"images_processed": 1}}},
        ]
        return httpx.Response(200, text="".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n")

    client = MetaClient(api_key="k", base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    img = np.zeros((H, W, 3), np.uint8)
    (res,) = sam.segment_images([img], points=[sam.Point(430 / W, 230 / H)], client=client)

    body = seen["body"]
    assert body["model"] == "sam-3.1" and body["metadata"]["mask_encoding"] == "one_bit"
    parts = body["input"][0]["content"]
    assert parts[0]["text"] == f"<0f>1<|point;x=430;y=230;w={W};h={H}|>"
    assert parts[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert res.selected_id == 1 and res.frames_with_mask == [0] and res.usage == {"images_processed": 1}
    assert res.masks[0].mask.sum() == 63022
