from __future__ import annotations

import threading
import time

from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator


def _acquire(coordinator: ModelOperationCoordinator, kind: str = "direct_audio"):
    return coordinator.acquire(
        kind=kind,
        session_id="session-1",
        turn_id="turn-1",
        turn_revision=1,
        cancel_generation=0,
    )


def test_optional_preview_is_dropped_while_model_is_busy() -> None:
    coordinator = ModelOperationCoordinator()
    active = _acquire(coordinator)

    preview = coordinator.acquire(
        kind="transcription_preview",
        session_id="session-1",
        turn_id="turn-1",
        turn_revision=1,
        cancel_generation=0,
        drop_if_busy=True,
    )

    assert active is not None
    assert preview is None
    coordinator.release(active)


def test_model_operations_are_serialized_until_owner_releases() -> None:
    coordinator = ModelOperationCoordinator()
    first = _acquire(coordinator)
    acquired = threading.Event()
    second_tokens = []

    def acquire_second() -> None:
        second_tokens.append(_acquire(coordinator, "post_tool"))
        acquired.set()

    worker = threading.Thread(target=acquire_second)
    worker.start()
    assert not acquired.wait(0.05)

    assert first is not None
    coordinator.release(first)
    assert acquired.wait(0.5)
    worker.join(0.5)
    assert second_tokens[0] is not None
    coordinator.release(second_tokens[0])


def test_cancellation_closes_transport_and_releases_waiter() -> None:
    coordinator = ModelOperationCoordinator()
    token = _acquire(coordinator)
    cancelled = threading.Event()

    assert token is not None

    def close_transport() -> None:
        cancelled.set()
        coordinator.release(token)

    assert coordinator.bind_cancel(token, close_transport)
    result = coordinator.cancel_and_wait("barge_in", timeout_s=0.5)

    assert cancelled.wait(0.5)
    assert result.released
    assert not result.detached
    assert result.operation == token
    replacement = _acquire(coordinator, "post_tool")
    assert replacement is not None
    coordinator.release(replacement)


def test_late_cancel_binding_is_invoked() -> None:
    coordinator = ModelOperationCoordinator()
    token = _acquire(coordinator)
    cancellation_started = threading.Event()
    result_holder = []

    assert token is not None

    def cancel() -> None:
        result_holder.append(coordinator.cancel_and_wait("revision", timeout_s=0.5))

    worker = threading.Thread(target=cancel)
    worker.start()
    time.sleep(0.05)

    def close_transport() -> None:
        cancellation_started.set()
        coordinator.release(token)

    assert coordinator.bind_cancel(token, close_transport)
    assert cancellation_started.wait(0.5)
    worker.join(0.5)
    assert result_holder[0].released
    assert not result_holder[0].detached


def test_stuck_generation_detaches_without_poisoning_new_owner() -> None:
    coordinator = ModelOperationCoordinator()
    stale = _acquire(coordinator)
    cancellation_started = threading.Event()

    assert stale is not None
    assert coordinator.bind_cancel(stale, cancellation_started.set)
    result = coordinator.cancel_and_wait("response_cancel", timeout_s=0.02)

    assert cancellation_started.wait(0.5)
    assert result.released
    assert result.detached

    current = _acquire(coordinator, "post_tool")
    assert current is not None
    coordinator.release(stale)
    assert coordinator.is_current(current)
    coordinator.release(current)


def test_stale_work_never_acquires_model_owner() -> None:
    coordinator = ModelOperationCoordinator()
    assert coordinator.acquire(
        kind="direct_audio",
        session_id="session-1",
        turn_id="turn-1",
        turn_revision=1,
        cancel_generation=2,
        stale=lambda: True,
    ) is None
