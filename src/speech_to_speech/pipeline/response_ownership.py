"""Conversation-local ownership for provisional realtime responses.

The OpenAI Realtime response flags are intentionally presentation-oriented: a
direct-audio request may be working before it has produced a response object.
This tracker is the cancellation authority for that gap.  It is deliberately
small and independent of transports so callers can mark an old owner stale
*before* attempting a potentially slow HTTP cancellation.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Any, Iterator, Literal

ResponseOwnerState = Literal[
    "idle",
    "capturing",
    "pending",
    "producing",
    "priming",
    "audible",
    "completed",
    "cancelled",
]

_OUTPUT_ADMISSIBLE_STATES = frozenset({"pending", "producing", "priming", "audible"})


def response_epoch_admission(
    *,
    runtime_config: Any | None,
    response_epoch: int | None,
    cancel_scope: Any | None = None,
) -> bool | None:
    """Return authoritative response-epoch admission when it is available.

    Realtime sessions install ``_response_epoch_admits_output`` on their local
    runtime configuration.  An explicit response epoch governed by that
    callback owns downstream output even when a newer speculative input has
    already been accepted behind an audible, non-interrupted answer.  The
    older ``_response_epoch_is_current`` callback remains a compatibility
    boundary for embedders that have not separated output admission from late
    playback acknowledgement.  Legacy and non-Realtime callers return
    ``None`` so their turn tracker remains the compatibility authority.
    """

    if response_epoch is None:
        return None
    local_pipeline = getattr(runtime_config, "local_pipeline", None)
    predicate = local_pipeline.get("_response_epoch_admits_output") if isinstance(local_pipeline, dict) else None
    if callable(predicate):
        return bool(predicate(response_epoch))
    predicate = local_pipeline.get("_response_epoch_is_current") if isinstance(local_pipeline, dict) else None
    if callable(predicate):
        return bool(predicate(response_epoch))

    # Some embedders attach the epoch authority directly to their shared
    # cancellation scope.  Keep that migration boundary supported without
    # requiring every message to carry a RuntimeConfig instance.
    stale_predicate = getattr(cancel_scope, "is_response_epoch_stale", None)
    if callable(stale_predicate):
        return not bool(stale_predicate(response_epoch))
    return None


@contextmanager
def response_epoch_history_transaction(
    *,
    runtime_config: Any | None,
    response_epoch: int | None,
) -> Iterator[bool]:
    """Atomically admit one short history transaction for a response epoch.

    Direct-Gemma workers run independently from the VAD thread. A plain
    ``admits_output`` check followed by ``Chat.add_item`` leaves a small window
    where VAD can invalidate and roll back a provisional owner before that
    worker inserts stale user, assistant, or function-call history. Realtime
    installs a tracker-owned guard that holds only for the in-memory history
    mutation. It never spans model or network work.
    """

    local_pipeline = getattr(runtime_config, "local_pipeline", None)
    guard = local_pipeline.get("_response_epoch_history_transaction") if isinstance(local_pipeline, dict) else None
    if callable(guard):
        with guard(response_epoch) as admitted:
            yield bool(admitted)
        return
    # Legacy callers have no shared VAD ownership lock. Retain their existing
    # predicate rather than synchronizing unrelated backends.
    with nullcontext(response_epoch_admission(runtime_config=runtime_config, response_epoch=response_epoch)) as admitted:
        yield admitted is not False


def wait_for_response_epoch_admission(
    *,
    runtime_config: Any | None,
    response_epoch: int | None,
) -> bool:
    """Wait for a retained successor response to become authoritative.

    When interruption is disabled, accepted speech can be queued behind an
    already-audible response.  Realtime sessions install a condition-backed
    waiter which promotes that queued epoch after the preceding response's
    terminal event.  Non-Realtime callers have no queue and continue
    immediately.  Cancellation, supersession, or connection close returns
    ``False`` so the detached request performs no model work.
    """

    if response_epoch is None:
        return True
    local_pipeline = getattr(runtime_config, "local_pipeline", None)
    waiter = local_pipeline.get("_wait_response_epoch_current") if isinstance(local_pipeline, dict) else None
    if not callable(waiter):
        return True
    return bool(waiter(response_epoch))


def response_output_allowed(
    *,
    runtime_config: Any | None,
    response_epoch: int | None,
    turn_id: str | None,
    turn_revision: int | None,
    speculative_turns: Any | None,
    cancel_scope: Any | None = None,
) -> bool:
    """Resolve output ownership before falling back to speculative turns.

    The ordering is deliberate: response epochs are the conversation-wide
    lifecycle authority; speculative turn identity exists for pre-ownership
    and backwards-compatible traffic only.
    """

    admission = response_epoch_admission(
        runtime_config=runtime_config,
        response_epoch=response_epoch,
        cancel_scope=cancel_scope,
    )
    if admission is not None:
        return admission
    if speculative_turns is None:
        return True
    return bool(speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision))


@dataclass(frozen=True)
class ResponseOwner:
    input_epoch: int
    response_epoch: int
    state: ResponseOwnerState
    turn_id: str | None = None
    turn_revision: int | None = None
    response_id: str | None = None
    playback_started: bool = False
    reason: str | None = None

    @property
    def audible(self) -> bool:
        # ``response.done`` can arrive before the browser's worklet renders its
        # first scheduled sample.  Transport completion is not proof of sound.
        return self.playback_started


@dataclass(frozen=True)
class Supersession:
    previous: ResponseOwner | None
    input_epoch: int
    reason: str
    invalidated: bool = True
    superseded_queued: tuple[ResponseOwner, ...] = ()

    @property
    def requires_transport_cancel(self) -> bool:
        return (
            self.invalidated
            and self.previous is not None
            and self.previous.state not in {"completed", "cancelled", "idle"}
        )

    @property
    def requires_output_suppression(self) -> bool:
        return self.invalidated and self.previous is not None and not self.previous.audible

    @property
    def pre_audible(self) -> bool:
        return self.invalidated and self.previous is not None and not self.previous.audible


class ResponseOwnershipTracker:
    """Monotonic input/response epochs for one conversation.

    ``input_started`` invalidates the old response synchronously.  The next
    accepted speech stop then claims a new pending response owner.  This split
    means a resumed user utterance cancels unheard work immediately while still
    preserving the existing VAD accumulation policy.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._closed = False
        self._input_epoch = 0
        self._response_epoch = 0
        self._active: ResponseOwner | None = None
        self._queued: list[int] = []
        self._owners_by_turn: dict[tuple[str | None, int | None], int] = {}
        self._owners: dict[int, ResponseOwner] = {}

    @property
    def input_epoch(self) -> int:
        with self._lock:
            return self._input_epoch

    @property
    def response_epoch(self) -> int:
        with self._lock:
            return self._response_epoch

    def active(self) -> ResponseOwner | None:
        with self._lock:
            return self._active

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize a short ownership-adjacent in-memory mutation.

        Rollback and an admission-checked Chat mutation must not interleave
        with ``input_started``. Do not use this around model work, HTTP, or
        queue waits.
        """

        with self._lock:
            yield

    @contextmanager
    def history_transaction(self, response_epoch: int | None) -> Iterator[bool]:
        """Hold ownership while admitting one response history mutation."""

        with self.transaction():
            admitted = (
                response_epoch is None
                or (
                    self._active is not None
                    and self._active.response_epoch == response_epoch
                    and self._active.state in _OUTPUT_ADMISSIBLE_STATES
                )
            )
            yield admitted

    def input_started(self, *, reason: str, invalidate_active: bool = True) -> Supersession:
        """Advance input identity, optionally retaining an audible answer.

        Capture must still receive a new monotonic input epoch when barge-in is
        disabled.  In that case the already-audible owner remains authoritative
        until its real terminal event completes it.
        """
        with self._lock:
            self._input_epoch += 1
            previous = self._active
            superseded_queued: list[ResponseOwner] = []
            for response_epoch in self._queued:
                queued = self._owners.get(response_epoch)
                if queued is None or queued.state in {"completed", "cancelled"}:
                    continue
                superseded_queued.append(self._replace(queued, state="cancelled", reason=reason))
            self._queued.clear()
            if previous is not None and invalidate_active:
                if previous.state not in {"completed", "cancelled"} or not previous.audible:
                    self._replace(previous, state="cancelled", reason=reason)
                # Even an already-heard completed response cannot remain the
                # current owner after newer input starts: late text, PCM, and
                # terminal events for it are stale.  Preserve the completed
                # record for diagnostics/history, but detach it from routing.
                self._active = None
            self._condition.notify_all()
            return Supersession(
                previous=previous,
                input_epoch=self._input_epoch,
                reason=reason,
                invalidated=invalidate_active,
                superseded_queued=tuple(superseded_queued),
            )

    def claim_pending(
        self,
        *,
        turn_id: str | None,
        turn_revision: int | None,
        activate: bool = True,
    ) -> ResponseOwner:
        """Claim the next response before model work begins."""
        with self._lock:
            self._response_epoch += 1
            owner = ResponseOwner(
                input_epoch=self._input_epoch,
                response_epoch=self._response_epoch,
                state="pending",
                turn_id=turn_id,
                turn_revision=turn_revision,
            )
            self._owners[owner.response_epoch] = owner
            self._owners_by_turn[(turn_id, turn_revision)] = owner.response_epoch
            if activate:
                self._active = owner
            else:
                self._queued.append(owner.response_epoch)
            self._condition.notify_all()
            return owner

    def promote_next(self) -> ResponseOwner | None:
        """Make the oldest retained-turn successor authoritative."""
        with self._lock:
            while self._queued:
                response_epoch = self._queued.pop(0)
                owner = self._owners.get(response_epoch)
                if owner is None or owner.state in {"completed", "cancelled"}:
                    continue
                self._active = owner
                self._condition.notify_all()
                return owner
            self._condition.notify_all()
            return None

    def is_queued(self, response_epoch: int | None) -> bool:
        if response_epoch is None:
            return False
        with self._lock:
            return response_epoch in self._queued

    def has_queued(self) -> bool:
        """Return whether a non-terminal successor is waiting for promotion."""

        with self._lock:
            return any(
                (owner := self._owners.get(response_epoch)) is not None
                and owner.state not in {"completed", "cancelled"}
                for response_epoch in self._queued
            )

    def wait_until_current(self, response_epoch: int | None) -> bool:
        """Wait for a retained-turn successor to become the active owner.

        An interrupt-disabled audible answer keeps ownership until its terminal
        event.  Accepted successor audio is already in the pipeline by then, so
        its model worker waits here instead of treating the queued epoch as
        stale and silently dropping the turn.  Supersession or connection close
        wakes the worker and returns ``False``.
        """
        if response_epoch is None:
            return True
        with self._condition:
            while True:
                if self._closed:
                    return False
                owner = self._owners.get(response_epoch)
                if owner is None or owner.state in {"completed", "cancelled"}:
                    return False
                if (
                    self._active is not None
                    and self._active.response_epoch == response_epoch
                    and self._active.state in _OUTPUT_ADMISSIBLE_STATES
                ):
                    return True
                if response_epoch not in self._queued:
                    return False
                self._condition.wait()

    def close(self) -> None:
        """Wake queued model workers when the owning conversation ends."""
        with self._condition:
            self._closed = True
            for response_epoch in self._queued:
                owner = self._owners.get(response_epoch)
                if owner is not None and owner.state not in {"completed", "cancelled"}:
                    self._replace(owner, state="cancelled", reason="connection_closed")
            self._queued.clear()
            self._condition.notify_all()

    def owner_for_turn(self, turn_id: str | None, turn_revision: int | None) -> ResponseOwner | None:
        with self._lock:
            epoch = self._owners_by_turn.get((turn_id, turn_revision))
            return self._owners.get(epoch) if epoch is not None else None

    def is_current(self, response_epoch: int | None) -> bool:
        if response_epoch is None:
            return True
        with self._lock:
            return self._active is not None and self._active.response_epoch == response_epoch and self._active.state != "cancelled"

    def admits_output(self, response_epoch: int | None) -> bool:
        """Whether model/TTS output may still mutate the active response.

        A completed owner intentionally remains current long enough for a late
        browser first-render acknowledgement, but it is terminal for text,
        PCM, metrics, and further completion events.
        """
        if response_epoch is None:
            return True
        with self._lock:
            return (
                self._active is not None
                and self._active.response_epoch == response_epoch
                and self._active.state in _OUTPUT_ADMISSIBLE_STATES
            )

    def matches_active_identity(
        self,
        *,
        input_epoch: int | None,
        response_epoch: int | None,
        response_id: str | None = None,
        require_output_admission: bool = True,
    ) -> bool:
        """Compare one immutable pipeline identity with the active owner.

        Callers commonly use this while holding :meth:`transaction`, making
        the comparison and the adjacent presentation/history mutation one
        operation. A missing response epoch denotes a legacy, unowned caller;
        owned realtime work must match both epochs and, once known, the exact
        response id.
        """

        if response_epoch is None:
            return True
        with self._lock:
            owner = self._active
            if owner is None or owner.response_epoch != response_epoch:
                return False
            if input_epoch is None or owner.input_epoch != input_epoch:
                return False
            if response_id is not None and owner.response_id != response_id:
                return False
            if require_output_admission and owner.state not in _OUTPUT_ADMISSIBLE_STATES:
                return False
            return True

    def owner_for_response_epoch(self, response_epoch: int | None) -> ResponseOwner | None:
        if response_epoch is None:
            return None
        with self._lock:
            return self._owners.get(response_epoch)

    def transition(
        self,
        response_epoch: int | None,
        state: ResponseOwnerState,
        *,
        response_id: str | None = None,
        playback_started: bool | None = None,
        reason: str | None = None,
    ) -> ResponseOwner | None:
        if response_epoch is None:
            return None
        with self._lock:
            current = self._owners.get(response_epoch)
            if current is None or current.state == "cancelled":
                return current
            updated = self._replace(
                current,
                state=state,
                response_id=current.response_id if response_id is None else response_id,
                playback_started=current.playback_started if playback_started is None else playback_started,
                reason=reason,
            )
            if self._active is not None and self._active.response_epoch == response_epoch:
                self._active = updated
            self._condition.notify_all()
            return updated

    def playback_started(self, *, response_id: str, response_epoch: int) -> ResponseOwner | None:
        with self._lock:
            current = self._owners.get(response_epoch)
            if current is None or current.response_id != response_id or not self.is_current(response_epoch):
                return None
            if current.playback_started:
                return current
            # Keep the terminal lifecycle state when the browser renders after
            # ``response.done``.  The independent heard bit, not that state,
            # decides whether a newer input is a pre-audible supersession.
            state: ResponseOwnerState = "completed" if current.state == "completed" else "audible"
            return self.transition(response_epoch, state, response_id=response_id, playback_started=True)

    def complete(self, response_epoch: int | None, *, reason: str | None = None) -> ResponseOwner | None:
        return self.transition(response_epoch, "completed", reason=reason)

    def _replace(self, owner: ResponseOwner, **changes: object) -> ResponseOwner:
        values = {
            "input_epoch": owner.input_epoch,
            "response_epoch": owner.response_epoch,
            "state": owner.state,
            "turn_id": owner.turn_id,
            "turn_revision": owner.turn_revision,
            "response_id": owner.response_id,
            "playback_started": owner.playback_started,
            "reason": owner.reason,
        }
        values.update(changes)
        updated = ResponseOwner(**values)  # type: ignore[arg-type]
        self._owners[owner.response_epoch] = updated
        return updated
