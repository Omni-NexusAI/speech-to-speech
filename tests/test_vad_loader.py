from pathlib import Path

from speech_to_speech.VAD import vad_handler


def test_silero_loader_prefers_valid_cached_checkout(tmp_path: Path, monkeypatch) -> None:
    invalid = tmp_path / "snakers4_silero-vad_newer"
    invalid.mkdir()
    cached = tmp_path / "snakers4_silero-vad_master"
    cached.mkdir()
    (cached / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(vad_handler.torch.hub, "get_dir", lambda: str(tmp_path))

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "model", "helpers"

    monkeypatch.setattr(vad_handler.torch.hub, "load", fake_load)

    assert vad_handler._load_silero_vad() == ("model", "helpers")
    assert calls == [
        (
            (str(cached), "silero_vad"),
            {"source": "local", "trust_repo": True},
        )
    ]


def test_silero_loader_tries_next_valid_cache_before_remote(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "snakers4_silero-vad_release"
    second = tmp_path / "snakers4_silero-vad_master"
    for cached in (first, second):
        cached.mkdir()
        (cached / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(vad_handler.torch.hub, "get_dir", lambda: str(tmp_path))

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == str(first):
            raise RuntimeError("invalid local checkout")
        return "model", "helpers"

    monkeypatch.setattr(vad_handler.torch.hub, "load", fake_load)

    assert vad_handler._load_silero_vad() == ("model", "helpers")
    assert [call[0][0] for call in calls] == [str(first), str(second)]
    assert all(call[1]["source"] == "local" for call in calls)


def test_silero_loader_uses_remote_when_no_valid_cache(tmp_path: Path, monkeypatch) -> None:
    incomplete = tmp_path / "snakers4_silero-vad_master"
    incomplete.mkdir()
    calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(vad_handler.torch.hub, "get_dir", lambda: str(tmp_path))

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "model", "helpers"

    monkeypatch.setattr(vad_handler.torch.hub, "load", fake_load)

    assert vad_handler._load_silero_vad() == ("model", "helpers")
    assert calls == [
        (
            ("snakers4/silero-vad", "silero_vad"),
            {"trust_repo": True, "skip_validation": True},
        )
    ]
