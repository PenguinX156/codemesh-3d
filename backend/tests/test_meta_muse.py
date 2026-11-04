"""Muse wrapper: request shape, JSON/tool-call parsing, usage and cost (mocked API)."""

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from codemesh.meta import muse
from codemesh.meta.client import MetaClient

USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "completion_tokens_details": {"reasoning_tokens": 400},
    "prompt_tokens_details": {"cached_tokens": 200},
}


def run(reply_message, **kw):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": reply_message, "finish_reason": "stop"}], "usage": USAGE})

    client = MetaClient(api_key="k", base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    return muse.chat([{"role": "user", "content": "hi"}], client=client, **kw), seen["body"]


def test_schema_images_and_reasoning_in_request_and_parsed_output():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    img = np.zeros((2000, 1000, 3), np.uint8)
    reply, body = run({"content": '{"a": 3}'}, images=[img], response_schema=schema)
    assert reply.parsed == {"a": 3} and reply.text == '{"a": 3}'
    assert body["reasoning_effort"] == "high" and body["model"] == "muse-spark-1.3-contributor"
    assert body["response_format"]["json_schema"]["schema"] == schema
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "hi"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_tool_calls_and_usage_cost():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
    reply, body = run({"content": None, "tool_calls": [call]}, tools=[{"type": "function", "function": {"name": "f"}}])
    assert reply.text == "" and reply.tool_calls == [muse.ToolCall("c1", "f", {"x": 1})]
    assert "tools" in body and "response_format" not in body
    u = reply.usage
    assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.cached_tokens) == (1000, 500, 400, 200)
    assert u.cost_usd == pytest.approx((800 * 0.10 + 200 * 0.002 + 500 * 0.20) / 1e6)
    assert (u + u).prompt_tokens == 2000


def test_image_part_downscales_but_never_upscales():
    def size(arr):
        url = muse.image_part(arr, max_side=500)["image_url"]["url"]
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size

    assert size(np.zeros((1000, 2000, 3), np.uint8)) == (2000 // 4, 1000 // 4)
    assert size(np.zeros((100, 200, 3), np.uint8)) == (200, 100)
