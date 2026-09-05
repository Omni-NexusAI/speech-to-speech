"""Runtime pool diagnostics must not advertise legacy turn trimming for direct Gemma."""

from speech_to_speech.s2s_pipeline import _idle_pool_context_descriptor


def test_direct_gemma_idle_descriptor_reports_default_token_compaction_not_turn_cap():
    descriptor = _idle_pool_context_descriptor(
        stt="gemma-audio",
        chat_size=30,
        context_window=16384,
        compact_history=False,
    )

    assert descriptor["history_tokens"] is None
    assert descriptor["max_tokens"] == 16384
    assert descriptor["history_compaction"] == {
        "policy": {
            "enabled": True,
            "trigger_ratio": 0.70,
            "target_ratio": 0.50,
            "recent_turns": 6,
        },
        "status": {"status": "idle", "last_failure": None},
    }
    assert "limit" not in descriptor
    assert "turn_limit" not in descriptor
    assert "compact_history" not in descriptor


def test_non_direct_descriptor_retains_legacy_context_shape_and_policy():
    descriptor = _idle_pool_context_descriptor(
        stt="parakeet-tdt",
        chat_size=30,
        context_window=8192,
        compact_history=True,
    )

    assert descriptor == {
        "limit": 30,
        "turn_limit": 30,
        "history_tokens": 0,
        "max_tokens": 8192,
        "compact_history": True,
        "policy": "compact",
    }
