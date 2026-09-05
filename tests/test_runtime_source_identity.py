from speech_to_speech.api.openai_realtime.source_identity import runtime_source_identity


def test_runtime_source_identity_accepts_content_free_launcher_values(monkeypatch) -> None:
    revision = "a" * 40
    fingerprint = "b" * 64
    monkeypatch.setenv("S2S_RUNTIME_REVISION", revision)
    monkeypatch.setenv("S2S_RUNTIME_DIRTY", "1")
    monkeypatch.setenv("S2S_RUNTIME_SOURCE_FINGERPRINT", fingerprint)
    monkeypatch.setenv(
        "S2S_UI_ASSET_GENERATION",
        "main=31-live-voice-default;ws=18-adaptive-safe-start;chat=3-tool-privacy;playback=16-adaptive-safe-start",
    )

    assert runtime_source_identity() == {
        "source_revision": revision,
        "source_dirty": True,
        "source_fingerprint": fingerprint,
        "ui_asset_generation": (
            "main=31-live-voice-default;ws=18-adaptive-safe-start;"
            "chat=3-tool-privacy;playback=16-adaptive-safe-start"
        ),
    }


def test_runtime_source_identity_fails_closed_on_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("S2S_RUNTIME_REVISION", "not a revision")
    monkeypatch.setenv("S2S_RUNTIME_DIRTY", "maybe")
    monkeypatch.setenv("S2S_RUNTIME_SOURCE_FINGERPRINT", "short")
    monkeypatch.setenv("S2S_UI_ASSET_GENERATION", "contains\na newline")

    assert runtime_source_identity() == {
        "source_revision": "unknown",
        "source_dirty": None,
        "source_fingerprint": "unknown",
        "ui_asset_generation": "unknown",
    }


def test_runtime_source_identity_rejects_printable_but_noncanonical_asset_generation(monkeypatch) -> None:
    monkeypatch.setenv("S2S_RUNTIME_REVISION", "a" * 40)
    monkeypatch.setenv("S2S_RUNTIME_DIRTY", "0")
    monkeypatch.setenv("S2S_RUNTIME_SOURCE_FINGERPRINT", "b" * 64)
    monkeypatch.setenv("S2S_UI_ASSET_GENERATION", "main=31;ws=18;chat=3;playback=16;path=C:/private")

    assert runtime_source_identity()["ui_asset_generation"] == "unknown"
