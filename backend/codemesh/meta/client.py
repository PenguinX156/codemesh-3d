"""Shared HTTP client for the Meta Model API (https://api.meta.ai/v1).

One small class used by `sam.py` and `muse.py`: Bearer auth, timeouts, retries with exponential
backoff on 429/5xx/transport errors, SSE streaming and file upload. The API key is only ever put in
the Authorization header; it is never logged, never in exception text and never in `repr`.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from codemesh import config

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Video segmentation can run for minutes before the first byte; connect stays short.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=300.0, pool=10.0)


class MetaAPIError(RuntimeError):
    """Non-retryable (or retries exhausted) API failure. `status` is None for transport errors."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class MetaClient:
    """Thin retrying wrapper around `httpx.Client` for api.meta.ai."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        max_retries: int = 4,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else config.META_API_KEY
        if not key:
            raise MetaAPIError("META_API_KEY is not set (put it in the repo-root .env)")
        self._http = httpx.Client(
            base_url=(base_url or config.META_API_BASE).rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            transport=transport,
        )
        self.max_retries = max_retries
        self.backoff = backoff

    def __repr__(self) -> str:
        return f"MetaClient(base_url={str(self._http.base_url)!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MetaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core ---------------------------------------------------------------------------------

    def _send(self, request: httpx.Request, stream: bool = False) -> httpx.Response:
        """Send with retry on 429/5xx/transport errors; raise MetaAPIError on final failure."""
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                resp = self._http.send(request, stream=stream)
            except httpx.TransportError as exc:  # timeouts, connection resets, DNS
                if last:
                    raise MetaAPIError(f"transport error: {type(exc).__name__}: {exc}") from exc
                self._sleep(attempt, None, f"transport {type(exc).__name__}")
                continue
            if resp.status_code < 400:
                return resp
            if resp.status_code in RETRY_STATUS and not last:
                retry_after = resp.headers.get("retry-after")
                resp.read()
                resp.close()
                self._sleep(attempt, retry_after, f"HTTP {resp.status_code}")
                continue
            resp.read()
            resp.close()
            raise MetaAPIError(_error_message(resp), resp.status_code)
        raise AssertionError("unreachable")

    def _sleep(self, attempt: int, retry_after: str | None, why: str) -> None:
        delay = self.backoff * (2**attempt) * random.uniform(0.75, 1.25)
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            delay = max(delay, float(retry_after))
        log.warning("meta api retry %d (%s) in %.1fs", attempt + 1, why, delay)
        time.sleep(delay)

    # -- public helpers -------------------------------------------------------------------------

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON, return the parsed JSON body."""
        resp = self._send(self._http.build_request("POST", path, json=payload))
        return resp.json()

    def stream_events(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """POST with stream=true and yield each SSE `data:` JSON event until [DONE]/close.

        Retries only apply to opening the stream; a failure mid-stream raises MetaAPIError.
        """
        request = self._http.build_request(
            "POST",
            path,
            json={**payload, "stream": True},
            headers={"Accept": "text/event-stream"},
        )
        resp = self._send(request, stream=True)
        try:
            for raw in resp.iter_lines():
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    return
                if data:
                    yield json.loads(data)
        except httpx.TransportError as exc:
            raise MetaAPIError(f"stream interrupted: {type(exc).__name__}: {exc}") from exc
        finally:
            resp.close()

    def upload_file(self, path: str | Path, purpose: str = "user_data") -> str:
        """Upload a file via POST /files and return its id (`file-...`)."""
        path = Path(path)
        # bytes (not a file handle) so a retried request can re-send the body.
        request = self._http.build_request(
            "POST", "/files", data={"purpose": purpose}, files={"file": (path.name, path.read_bytes())}
        )
        return self._send(request).json()["id"]

    def delete_file(self, file_id: str) -> None:
        """Best-effort delete of an uploaded file."""
        try:
            self._send(self._http.build_request("DELETE", f"/files/{file_id}"))
        except MetaAPIError as exc:
            log.warning("could not delete %s: %s", file_id, exc)


def _error_message(resp: httpx.Response) -> str:
    """`HTTP <status>: <api error message>` without echoing request headers or the key."""
    try:
        err = resp.json().get("error", {})
        detail = err.get("message") or str(err) if isinstance(err, dict) else str(err)
    except (ValueError, AttributeError):
        detail = resp.text[:300]
    return f"HTTP {resp.status_code}: {detail}"
