from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from speech_to_speech.pipeline.cancellable_http import CancellableAsyncSSEStream, StreamCancelled


class _BlockingResponse:
    status_code = 200
    headers: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def aiter_lines(self):
        await asyncio.Event().wait()
        yield "unreachable"


class _BlockingClient:
    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, *_args, **_kwargs):
        return _BlockingResponse()


def test_close_cancels_async_transport_waiting_for_stream_data(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _BlockingClient)
    stream = CancellableAsyncSSEStream(
        "POST",
        "http://model.invalid/v1/chat/completions",
        json_body={"stream": True},
    )
    observed = SimpleNamespace(error=None)
    started = threading.Event()

    def consume() -> None:
        try:
            stream.wait_for_headers()
            started.set()
            list(stream.iter_lines())
        except BaseException as exc:
            observed.error = exc

    thread = threading.Thread(target=consume)
    thread.start()
    assert started.wait(1.0)

    stream.close()
    thread.join(1.0)

    assert not thread.is_alive()
    assert stream.closed
    assert isinstance(observed.error, StreamCancelled)


def test_close_before_worker_start_is_honored():
    stream = CancellableAsyncSSEStream(
        "POST",
        "http://model.invalid/v1/chat/completions",
        json_body={"stream": True},
    )
    stream.close()

    with pytest.raises(StreamCancelled):
        stream.wait_for_headers()
