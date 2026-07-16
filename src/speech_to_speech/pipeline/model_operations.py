from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOperationToken:
    operation_id: int
    kind: str
    session_id: str | None
    turn_id: str | None
    turn_revision: int | None
    cancel_generation: int | None


@dataclass(frozen=True)
class ModelCancellationResult:
    operation: ModelOperationToken | None
    released: bool
    detached: bool
    elapsed_ms: float
    reason: str


@dataclass
class _ActiveOperation:
    token: ModelOperationToken
    cancel: Callable[[], None] | None = None
    cancel_requested: bool = False
    cancel_started: bool = False


class ModelOperationCoordinator:
    """Serializes model HTTP operations owned by one realtime pipeline.

    Direct-audio Gemma and post-tool chat generation run on different handler
    threads. This coordinator gives the selected endpoint one logical owner and
    lets the websocket router cancel that owner's transport without guessing
    which handler is currently blocked.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_id = 1
        self._active: _ActiveOperation | None = None

    def acquire(
        self,
        *,
        kind: str,
        session_id: str | None,
        turn_id: str | None,
        turn_revision: int | None,
        cancel_generation: int | None,
        drop_if_busy: bool = False,
        stale: Callable[[], bool] | None = None,
    ) -> ModelOperationToken | None:
        with self._condition:
            while self._active is not None:
                if drop_if_busy or (stale is not None and stale()):
                    return None
                self._condition.wait(0.05)
            if stale is not None and stale():
                return None
            token = ModelOperationToken(
                operation_id=self._next_id,
                kind=kind,
                session_id=session_id,
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=cancel_generation,
            )
            self._next_id += 1
            self._active = _ActiveOperation(token=token)
            return token

    def bind_cancel(self, token: ModelOperationToken, cancel: Callable[[], None]) -> bool:
        invoke = False
        with self._condition:
            active = self._active
            if active is None or active.token.operation_id != token.operation_id:
                return False
            active.cancel = cancel
            if active.cancel_requested and not active.cancel_started:
                active.cancel_started = True
                invoke = True
        if invoke:
            self._invoke_cancel(token, cancel)
        return True

    def is_current(self, token: ModelOperationToken) -> bool:
        with self._condition:
            return self._active is not None and self._active.token.operation_id == token.operation_id

    def release(self, token: ModelOperationToken) -> None:
        with self._condition:
            active = self._active
            if active is not None and active.token.operation_id == token.operation_id:
                self._active = None
                self._condition.notify_all()

    def cancel_and_wait(self, reason: str, timeout_s: float = 2.0) -> ModelCancellationResult:
        started = time.monotonic()
        callback: Callable[[], None] | None = None
        token: ModelOperationToken | None = None
        with self._condition:
            active = self._active
            if active is None:
                return ModelCancellationResult(None, True, False, 0.0, reason)
            token = active.token
            active.cancel_requested = True
            if active.cancel is not None and not active.cancel_started:
                active.cancel_started = True
                callback = active.cancel

        if callback is not None:
            self._invoke_cancel(token, callback)

        deadline = started + max(0.0, timeout_s)
        detached = False
        with self._condition:
            while self._active is not None and self._active.token.operation_id == token.operation_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._active = None
                    self._condition.notify_all()
                    detached = True
                    break
                self._condition.wait(min(0.05, remaining))
            released = self._active is None or self._active.token.operation_id != token.operation_id

        elapsed_ms = (time.monotonic() - started) * 1000
        if detached:
            logger.warning(
                "Detached stale model operation after cancellation timeout "
                "(operation=%s kind=%s reason=%s elapsed_ms=%.1f)",
                token.operation_id,
                token.kind,
                reason,
                elapsed_ms,
            )
        return ModelCancellationResult(token, released, detached, elapsed_ms, reason)

    def reset(self) -> None:
        with self._condition:
            self._active = None
            self._condition.notify_all()

    @staticmethod
    def _invoke_cancel(token: ModelOperationToken, callback: Callable[[], None]) -> None:
        def run() -> None:
            try:
                callback()
            except Exception:
                logger.exception(
                    "Model transport cancellation failed (operation=%s kind=%s)",
                    token.operation_id,
                    token.kind,
                )

        threading.Thread(
            target=run,
            name=f"model-cancel-{token.operation_id}",
            daemon=True,
        ).start()
