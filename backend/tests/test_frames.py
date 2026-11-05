import json
from types import SimpleNamespace

import cv2
import numpy as np

from codemesh.agent import frames


def test_review_frames_fake_muse(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(14):
        cv2.imwrite(str(tmp_path / "frames" / f"{i:04d}.jpg"), np.full((64, 48, 3), 120, np.uint8))
        cv2.imwrite(str(tmp_path / "masks" / f"{i:04d}.png"), np.pad(np.full((20, 20), 255, np.uint8), 14))
    calls = []

    def fake_chat(messages, images, response_schema, **kw):
        calls.append(images[0].shape)
        idx = [int(s) for s in messages[0]["content"].rsplit("Frames: [", 1)[1].rstrip("]").split(", ")]
        rate = {0: ("bad", "bad", True), 1: ("bad", "bad", False), 2: ("bad", "good", False)}
        out = [{"index": i, "shape_use": rate.get(i, ("good",) * 2 + (False,))[0],
                "texture_use": rate.get(i, ("good",) * 2 + (False,))[1],
                "unique_viewpoint": rate.get(i, (0, 0, False))[2], "reasons": "x"} for i in idx if i != 13]
        return SimpleNamespace(parsed={"frames": out}, usage=SimpleNamespace(cost_usd=.001))

    review = frames.review_frames(tmp_path, chat=fake_chat)
    assert calls == [(1152, 1536, 3)] * 2
    roles = {f["index"]: f["role"] for f in review["frames"]}
    assert roles[0] == "reference" and roles[1] == "ignore" and roles[2] == "reference"
    assert roles[13] == "geometry"  # unrated frames are kept
    assert review["excluded_from_carving"] == [0, 1, 2]
    assert json.loads((tmp_path / "agent" / "frames_review.json").read_text())["sheets"] == 2
