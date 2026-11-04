"""Muse Spark 1.3 Contributor via Chat Completions (`POST /v1/chat/completions`).

`chat()` is a thin wrapper: messages (+ optional images, tools, JSON schema, reasoning effort) in,
`MuseReply` (text, parsed JSON, tool calls, token usage, cost) out. We use Chat Completions rather
than Responses because the app keeps its own tool loop and needs no server-side state.

Contributor mode: prompts and outputs may be used by Meta to train models. Never send personal
content (the UI shows a disclosure). Images go inline as base64 JPEG data URLs (docs: up to 50 MB
each); they are downscaled to `max_side` first to keep token usage small.
"""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from codemesh import config
from codemesh.meta.client import MetaClient

# USD per 1M tokens, Contributor mode (https://dev.meta.ai/docs/pricing-rate-limits).
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT = 0.10, 0.002, 0.20

ImageInput = str | Path | np.ndarray


@dataclass(frozen=True)
class Usage:
    """Token usage of one call; `completion_tokens` already includes `reasoning_tokens`."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        fresh = self.prompt_tokens - self.cached_tokens
        return (fresh * PRICE_INPUT + self.cached_tokens * PRICE_CACHED + self.completion_tokens * PRICE_OUTPUT) / 1e6

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(a + b for a, b in zip(self._t(), other._t())))

    def _t(self) -> tuple[int, int, int, int]:
        return (self.prompt_tokens, self.completion_tokens, self.reasoning_tokens, self.cached_tokens)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class MuseReply:
    text: str  # visible answer ("" when the model only called tools)
    parsed: Any | None  # json.loads(text) when response_schema was given
    tool_calls: list[ToolCall]
    usage: Usage
    seconds: float
    finish_reason: str
    message: dict[str, Any] = field(repr=False, default_factory=dict)  # raw assistant message; append to
    # `messages` (with tool results) to continue a tool loop


def image_part(image: ImageInput, max_side: int = 1280) -> dict[str, Any]:
    """Chat Completions image content part from a path or RGB uint8 array (JPEG q90, <= max_side)."""
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    pil.thumbnail((max_side, max_side))  # only ever shrinks
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=90)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def chat(
    messages: Sequence[dict[str, Any]],
    *,
    images: Sequence[ImageInput] | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    response_schema: dict[str, Any] | None = None,
    reasoning: str = config.MUSE_REASONING,
    max_tokens: int | None = 8000,
    max_side: int = 1280,
    client: MetaClient | None = None,
) -> MuseReply:
    """Call Muse Spark.

    Args:
        messages: OpenAI-style `{"role", "content"}` dicts (content may be a string or part list).
        images: attached to the last user message, in order.
        tools: function tools in Chat Completions form `{"type": "function", "function": {...}}`.
            Meta only supports tool_choice "auto".
        response_schema: JSON Schema; decoding is constrained to it and `reply.parsed` is set.
        reasoning: reasoning effort (low | medium | high | xhigh). "none" is unsupported.
        max_tokens: cap on reasoning + visible tokens combined; keep it generous at high effort.
    """
    msgs = [dict(m) for m in messages]
    if images:
        last = next((m for m in reversed(msgs) if m["role"] == "user"), None)
        if last is None:
            raise ValueError("images need a user message to attach to")
        content = last["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        last["content"] = parts + [image_part(i, max_side) for i in images]

    body: dict[str, Any] = {"model": config.MUSE_MODEL, "messages": msgs, "reasoning_effort": reasoning}
    if max_tokens:
        body["max_tokens"] = max_tokens
    if tools:
        body["tools"] = list(tools)
    if response_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": response_schema},
        }

    own = client is None
    client = client or MetaClient()
    t0 = time.perf_counter()
    try:
        data = client.post_json("/chat/completions", body)
    finally:
        if own:
            client.close()
    seconds = time.perf_counter() - t0

    choice = data["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    u = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=u.get("prompt_tokens", 0),
        completion_tokens=u.get("completion_tokens", 0),
        reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    )
    calls = [
        ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
        for tc in message.get("tool_calls") or []
    ]
    parsed = json.loads(text) if response_schema and text else None
    return MuseReply(text, parsed, calls, usage, seconds, choice.get("finish_reason", ""), message)
