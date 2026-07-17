from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class StreamCancelled(RuntimeError):
    """Raised in the synchronous consumer when its async transport is cancelled."""


_END = object()


class CancellableAsyncSSEStream:
    """Expose an async HTTP stream to synchronous pipeline handlers.

    Closing a synchronous ``httpx.Client`` from another thread does not reliably
    abort a request that is waiting for response headers on Windows. Cancelling
    the owning async task does close the socket, which lets llama.cpp release its
    inference slot immediately.
    """

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: httpx.Timeout | float = 30.0,
    ) -> None:
        self.method = method
        self.url = url
        self.headers = dict(headers or {})
        self.json_body = dict(json_body or {})
        self.timeout = timeout
        self.status_code: int | None = None
        self.response_headers: dict[str, str] = {}

        self._items: queue.Queue[str | object] = queue.Queue()
        self._headers_ready = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._cancel_requested = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._thread_main,
                name="cancellable-http-stream",
                daemon=True,
            )
            self._thread.start()

    def wait_for_headers(self) -> None:
        self.start()
        while not self._headers_ready.wait(0.05):
            if self._done.is_set():
                break
        self.raise_for_status()

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error
        if self.status_code is None:
            if self._cancel_requested:
                raise StreamCancelled("HTTP stream cancelled before response headers")
            raise RuntimeError("HTTP stream ended before response headers")
        if self.status_code >= 400:
            request = httpx.Request(self.method, self.url, headers=self.headers)
            response = httpx.Response(
                self.status_code,
                headers=self.response_headers,
                request=request,
            )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} from model endpoint",
                request=request,
                response=response,
            )

    def iter_lines(self) -> Iterator[str]:
        self.wait_for_headers()
        while True:
            item = self._items.get()
            if item is _END:
                break
            yield str(item)
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        with self._lock:
            self._cancel_requested = True
            loop = self._loop
            task = self._task
        if loop is not None and task is not None and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    @property
    def closed(self) -> bool:
        return self._done.is_set()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._run())
        with self._lock:
            self._loop = loop
            self._task = task
            cancel_requested = self._cancel_requested
        if cancel_requested:
            task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            self._error = StreamCancelled("HTTP stream cancelled")
        except BaseException as exc:
            self._error = exc
        finally:
            self._headers_ready.set()
            self._items.put(_END)
            self._done.set()
            with self._lock:
                self._task = None
                self._loop = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _run(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                self.method,
                self.url,
                headers=self.headers,
                json=self.json_body,
            ) as response:
                self.status_code = response.status_code
                self.response_headers = dict(response.headers)
                self._headers_ready.set()
                if response.status_code >= 400:
                    await response.aread()
                    return
                async for line in response.aiter_lines():
                    self._items.put(line)


class CancellableAsyncByteStream:
    """Cancellable binary counterpart to :class:`CancellableAsyncSSEStream`.

    The FasterQwen3TTS API returns raw PCM, not SSE.  It still needs the same
    task-owned async transport: closing a synchronous ``httpx`` response from a
    different handler thread can leave a Windows socket waiting for headers and
    retain the single realtime pipeline worker.
    """

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: httpx.Timeout | float = 30.0,
    ) -> None:
        self.method = method
        self.url = url
        self.headers = dict(headers or {})
        self.json_body = dict(json_body or {})
        self.timeout = timeout
        self.status_code: int | None = None
        self.response_headers: dict[str, str] = {}
        self._items: queue.Queue[bytes | object] = queue.Queue()
        self._headers_ready = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._cancel_requested = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._thread_main, name="cancellable-http-bytes", daemon=True)
            self._thread.start()

    def wait_for_headers(self) -> None:
        self.start()
        while not self._headers_ready.wait(0.05):
            if self._done.is_set():
                break
        self.raise_for_status()

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error
        if self.status_code is None:
            if self._cancel_requested:
                raise StreamCancelled("HTTP byte stream cancelled before response headers")
            raise RuntimeError("HTTP byte stream ended before response headers")
        if self.status_code >= 400:
            request = httpx.Request(self.method, self.url, headers=self.headers)
            response = httpx.Response(self.status_code, headers=self.response_headers, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} from TTS endpoint", request=request, response=response
            )

    def iter_bytes(self) -> Iterator[bytes]:
        self.wait_for_headers()
        while True:
            item = self._items.get()
            if item is _END:
                break
            yield bytes(item)
        if self._error is not None and not self._cancel_requested:
            raise self._error

    def close(self) -> None:
        with self._lock:
            self._cancel_requested = True
            loop = self._loop
            task = self._task
        if loop is not None and task is not None and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    @property
    def closed(self) -> bool:
        return self._done.is_set()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._run())
        with self._lock:
            self._loop = loop
            self._task = task
            if self._cancel_requested:
                task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            self._error = StreamCancelled("HTTP byte stream cancelled")
        except BaseException as exc:
            self._error = exc
        finally:
            self._headers_ready.set()
            self._items.put(_END)
            self._done.set()
            with self._lock:
                self._task = None
                self._loop = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _run(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                self.method,
                self.url,
                headers=self.headers,
                json=self.json_body,
            ) as response:
                self.status_code = response.status_code
                self.response_headers = dict(response.headers)
                self._headers_ready.set()
                if response.status_code >= 400:
                    await response.aread()
                    return
                async for chunk in response.aiter_bytes():
                    if chunk:
                        self._items.put(chunk)


class ChatCompletionSSEStream:
    """Parse OpenAI-compatible SSE into ChatCompletionChunk objects."""

    def __init__(self, transport: CancellableAsyncSSEStream, chunk_type: type[Any]) -> None:
        self.transport = transport
        self.chunk_type = chunk_type

    def wait_for_headers(self) -> None:
        self.transport.wait_for_headers()

    def close(self) -> None:
        self.transport.close()

    def __iter__(self) -> Iterator[Any]:
        for line in self.transport.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON model stream line")
                continue
            yield self.chunk_type.model_validate(payload)
