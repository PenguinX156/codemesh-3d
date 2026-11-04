"""MetaClient: retry/backoff, error text, SSE, key hygiene (no network; httpx.MockTransport)."""

import httpx
import pytest

from codemesh.meta.client import MetaAPIError, MetaClient

KEY = "LLM_123_secretsecretsecret"


def client(handler, **kw) -> MetaClient:
    return MetaClient(api_key=KEY, base_url="https://api.test/v1", backoff=0, transport=httpx.MockTransport(handler), **kw)


def test_retries_429_and_5xx_then_succeeds():
    codes = iter([429, 503, 200])
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers["authorization"])
        code = next(codes)
        return httpx.Response(code, json={"ok": True} if code == 200 else {"error": {"message": "slow down"}})

    assert client(handler).post_json("/x", {}) == {"ok": True}
    assert len(seen) == 3 and set(seen) == {f"Bearer {KEY}"}


def test_400_is_not_retried_and_error_has_api_message_but_no_key():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "invalid prompt: object 1 carries more than one box"}})

    c = client(handler)
    with pytest.raises(MetaAPIError) as e:
        c.post_json("/responses", {})
    assert e.value.status == 400 and "more than one box" in str(e.value)
    assert len(calls) == 1
    assert KEY not in str(e.value) and KEY not in repr(c)


def test_gives_up_after_max_retries():
    c = client(lambda req: httpx.Response(503, json={}), max_retries=2)
    with pytest.raises(MetaAPIError) as e:
        c.post_json("/x", {})
    assert e.value.status == 503


def test_transport_error_is_retried():
    n = {"i": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        if n["i"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": 1})

    assert client(handler).post_json("/x", {}) == {"ok": 1}


def test_stream_events_parses_sse_and_sets_stream_flag():
    sse = 'event: a\ndata: {"type": "t1", "n": 1}\n\ndata: {"type": "t2"}\n\ndata: [DONE]\n\ndata: {"type": "after"}\n'
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(req.content)
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    events = list(client(handler).stream_events("/responses", {"model": "m"}))
    assert [e["type"] for e in events] == ["t1", "t2"]
    assert b'"stream":true' in bodies[0].replace(b" ", b"")


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr("codemesh.config.META_API_KEY", "")
    with pytest.raises(MetaAPIError):
        MetaClient()
