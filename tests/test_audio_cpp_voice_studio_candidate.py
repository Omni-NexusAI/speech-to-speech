import asyncio
import importlib.util
import io
import json
import sys
import types
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "integrations" / "audio-cpp"


def load_supervisor(monkeypatch, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "models": [
            {"id": "qwen3-tts-0.6b-base-bf16"},
            {"id": "qwen3-tts-1.7b-base-bf16"},
        ]
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("AUDIO_CPP_CONFIG_TEMPLATE", str(config_path))
    monkeypatch.setenv("VOICE_LIBRARY_DIR", str(tmp_path / "voices"))
    spec = importlib.util.spec_from_file_location("audio_cpp_candidate_supervisor", CANDIDATE / "supervisor.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_request_tuning_survives_named_profile_deletion(monkeypatch, tmp_path):
    """Later phrases must not reread a profile that changed after the response began."""
    supervisor = load_supervisor(monkeypatch, tmp_path)
    document = supervisor._default_tuning_profiles()
    frozen = dict(document["profiles"]["balanced"])
    frozen.update(
        id="response-frozen",
        name="Response frozen",
        revision=9,
        model="qwen3-tts-1.7b-base-bf16",
        max_reference_seconds=20,
        temperature=0.7,
        top_k=42,
        top_p=0.9,
        repetition_penalty=1.05,
        seed=17,
    )
    document["profiles"]["response-frozen"] = frozen
    supervisor._write_tuning_profiles(document)

    # Model a profile edit/deletion after phrase one.  The request-scoped
    # snapshot contains only bounded, already-resolved values and must remain
    # usable for phrase two without rereading the named profile document.
    document = supervisor._read_tuning_profiles()
    document["profiles"].pop("response-frozen")
    supervisor._write_tuning_profiles(document)

    snapshot = supervisor._resolve_request_tuning(
        {
            "tuning": {
                "provider": "qwen3tts-audiocpp",
                "profile_id": "response-frozen",
                "profile_revision": 9,
                "effective": {
                    key: frozen[key]
                    for key in supervisor.TUNING_VALUE_FIELDS
                    if key in frozen
                },
                "overrides": {"seed": 17},
            }
        }
    )

    assert snapshot["id"] == "response-frozen"
    assert snapshot["revision"] == 9
    assert snapshot["effective"]["temperature"] == 0.7
    assert snapshot["effective"]["max_reference_seconds"] == 20
    assert snapshot["engine_fields"] == {
        "temperature": 0.7,
        "top_k": 42,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "seed": 17,
    }


def test_full_voice_studio_exposes_truthful_candidate_controls():
    source = (CANDIDATE / "gradio_voice_studio.py").read_text(encoding="utf-8")
    required = [
        "orange_theme",
        'primary_hue="orange"',
        'theme=orange_theme()',
        "#f97316",
        "Settings & candidate model controls",
        "Load selected model",
        "Unload model",
        "Save profile changes",
        "Non-streaming (Full Quality)",
        "Buffered phrase PCM (rollback fallback)",
        "value=DEFAULT_PLAYBACK_MODE",
        "System default input",
        "Save LLM settings",
        "Remember API key",
        "sv-checkbox",
        "LLM connection",
        "Diagnostics &amp; tuning",
        "Output transport:",
        "model-native PCM16 at 24 kHz",
        "sv-help",
        "GPU admission guard (candidate only)",
        "Apply GPU guard settings",
        "Output container / codec",
        "Quality path",
        "initial_profile_ids",
        "allow_custom_value=True",
        "profile_dropdown_update",
        "Seed both Playground",
        "Pre-roll ms",
        "Start live mic",
        "supports native incremental PCM when the native runtime is enabled",
        "response_format: 'wav'",
        "stream: false",
        "state.activeTtsRequest.abort()",
        '"proxyBaseUrl": "/api/audio-cpp"',
        "queueProgressiveSpeech",
        "Progressive PCM diagnostics",
        "safe word/clause boundary",
        "No audio.cpp model is resident",
        "savePersistentStudioSettings",
        "loadPersistentStudioSettings",
    ]
    for marker in required:
        assert marker in source
    assert "http://127.0.0.1:7852/ingest/" not in source
    assert "config.proxyBaseUrl + '/llamacpp-audio-turn/stream'" in source


def test_streaming_widget_is_reconciled_on_initial_page_load():
    source = (CANDIDATE / "gradio_voice_studio.py").read_text(encoding="utf-8")
    assert "demo.load(fn=on_library_refresh" in source
    assert "fn=on_update_streaming_widget" in source
    assert "outputs=[s_streaming_widget]" in source


def test_profile_metadata_write_is_atomic_and_editable(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(CANDIDATE))
    monkeypatch.setitem(sys.modules, "gradio", types.SimpleNamespace(Blocks=object))
    import gradio_voice_studio as studio

    profile = studio.VoiceProfile(
        profile_id="profile-a",
        name="Original",
        task_type="Base",
        created_at=studio.now_iso(),
        ref_text="reference",
        ref_audio_filename="ref.wav",
    )
    studio.save_profile(tmp_path, profile)
    (studio.profile_dir(tmp_path, "profile-a") / "ref.wav").write_bytes(b"reference-audio")
    profile.name = "Renamed"
    profile.language = "English"
    studio.save_profile(tmp_path, profile)

    saved = studio.load_profile(tmp_path, "profile-a")
    assert saved.name == "Renamed"
    assert saved.language == "English"
    assert not (studio.profile_dir(tmp_path, "profile-a") / ".meta.json.tmp").exists()


def test_backend_status_reports_only_one_resident_model(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded")
    supervisor.engine = types.SimpleNamespace(poll=lambda: None)

    async def ready():
        return True

    monkeypatch.setattr(supervisor, "_engine_ready", ready)
    status = asyncio.run(supervisor.backend_models())
    assert status["loaded_models"] == ["qwen3-tts-1.7b-base-bf16"]
    assert status["singleResident"] is True
    assert status["runtime"]["native_incremental_pcm"] is False
    assert status["runtime"]["tts_mode"] == "offline-buffered"
    assert status["engineEpoch"] == 0
    assert len(status["supervisorInstanceId"]) == 32
    health = asyncio.run(supervisor.health())
    assert health["engineEpoch"] == 0
    assert health["supervisorInstanceId"] == supervisor.SUPERVISOR_INSTANCE_ID


def test_expected_engine_epoch_blocks_same_model_reload_and_never_reaches_engine(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    model = "qwen3-tts-1.7b-base-bf16"
    supervisor.state.update(activeModel=model, state="loaded", engineEpoch=8)
    supervisor.engine = types.SimpleNamespace(poll=lambda: None)

    async def ready():
        return True

    monkeypatch.setattr(supervisor, "_engine_ready", ready)
    monkeypatch.setattr(
        supervisor,
        "gpu_guard",
        lambda *_args, **_kwargs: {"ok": True, "reason": None},
    )
    stale = {
        "model": model,
        "expected_engine_epoch": 7,
        "expected_supervisor_instance_id": supervisor.SUPERVISOR_INSTANCE_ID,
    }
    with pytest.raises(HTTPException) as error:
        asyncio.run(supervisor._assert_synthesis_ready(stale))
    assert error.value.status_code == 409
    assert "expected_engine_epoch" not in stale
    assert "expected_supervisor_instance_id" not in stale
    assert "_engine_epoch" not in stale

    admitted = {
        "model": model,
        "expected_engine_epoch": "8",
        "expected_supervisor_instance_id": supervisor.SUPERVISOR_INSTANCE_ID,
    }
    assert asyncio.run(supervisor._assert_synthesis_ready(admitted)) == model
    assert admitted["_engine_epoch"] == 8
    assert "expected_engine_epoch" not in admitted
    assert "expected_supervisor_instance_id" not in admitted
    supervisor._strip_private_engine_fields(admitted)
    assert all(not key.startswith("_") for key in admitted)


def test_same_model_and_epoch_from_a_restarted_supervisor_is_rejected(monkeypatch, tmp_path):
    """A process restart must not reuse a matching numeric epoch as identity."""

    first = load_supervisor(monkeypatch, tmp_path / "first")
    second = load_supervisor(monkeypatch, tmp_path / "second")
    model = "qwen3-tts-1.7b-base-bf16"
    assert first.SUPERVISOR_INSTANCE_ID != second.SUPERVISOR_INSTANCE_ID
    second.state.update(activeModel=model, state="loaded", engineEpoch=1)
    second.engine = types.SimpleNamespace(poll=lambda: None)

    async def ready():
        return True

    monkeypatch.setattr(second, "_engine_ready", ready)
    monkeypatch.setattr(second, "gpu_guard", lambda *_args, **_kwargs: {"ok": True, "reason": None})
    stale_from_first = {
        "model": model,
        "expected_engine_epoch": 1,
        "expected_supervisor_instance_id": first.SUPERVISOR_INSTANCE_ID,
    }
    with pytest.raises(HTTPException) as error:
        asyncio.run(second._assert_synthesis_ready(stale_from_first))

    assert error.value.status_code == 409
    assert "expected_supervisor_instance_id" not in stale_from_first
    assert "_engine_epoch" not in stale_from_first


def test_gpu_busy_switch_preserves_current_resident_model(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded")
    supervisor.engine = types.SimpleNamespace(poll=lambda: None)
    stopped = []
    monkeypatch.setattr(
        supervisor,
        "gpu_guard",
        lambda model, **kwargs: {
            "ok": False,
            "reason": "GPU busy/insufficient VRAM; model load was not attempted.",
            "freeMiB": 1000,
            "requiredMiB": 5500,
            "utilizationPercent": 95,
        },
    )
    monkeypatch.setattr(supervisor, "_stop_engine", lambda **kwargs: stopped.append(kwargs))

    with pytest.raises(HTTPException) as error:
        asyncio.run(supervisor.switch_model("qwen3-tts-0.6b-base-bf16"))

    assert error.value.status_code == 409
    assert error.value.detail["state"] == "blocked"
    assert supervisor.state["activeModel"] == "qwen3-tts-1.7b-base-bf16"
    assert supervisor.state["state"] == "loaded"
    assert stopped == []


def test_unload_releases_engine_and_clears_resident_set(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded")
    stopped = []
    monkeypatch.setattr(supervisor, "_stop_engine", lambda **kwargs: stopped.append(kwargs))

    result = asyncio.run(supervisor.backend_unload())

    assert stopped == [{"action": "explicit-unload"}]
    assert result["state"] == "unloaded"
    assert result["loaded_models"] == []


def test_saved_clone_id_resolves_to_private_profile_audio(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    profile = tmp_path / "voices" / "profiles" / "abc123"
    profile.mkdir(parents=True)
    (profile / "ref.wav").write_bytes(b"RIFF-test")
    (profile / "meta.json").write_text(
        json.dumps(
            {
                "ref_audio_filename": "ref.wav",
                "ref_text": "reference transcript",
                "x_vector_only_mode": False,
            }
        ),
        encoding="utf-8",
    )
    payload = {"voice": "clone:abc123", "input": "hello"}

    supervisor._apply_clone_profile(payload)

    assert payload["task_type"] == "Base"
    assert payload["ref_text"] == "reference transcript"
    assert payload["ref_audio"]


def test_candidate_profile_import_exposes_stable_clone_id(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    result = supervisor._write_candidate_profile(
        "abc12345",
        {"name": "Candidate clone", "ref_text": "reference", "ref_audio": "UklGRg=="},
    )

    assert result["voice"] == "clone:abc12345"
    assert supervisor._voice_profiles()[0]["id"] == "abc12345"


def test_gradio_random_seed_sentinel_is_omitted_for_audio_cpp(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    random_payload = {"seed": -1, "input": "hello"}
    fixed_payload = {"seed": 42.0, "input": "hello"}

    supervisor._normalize_gradio_seed(random_payload)
    supervisor._normalize_gradio_seed(fixed_payload)

    assert "seed" not in random_payload
    assert fixed_payload["seed"] == 42


def test_same_loaded_model_is_a_lifecycle_noop(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded")
    supervisor.engine = types.SimpleNamespace(poll=lambda: None)

    async def ready():
        return True

    monkeypatch.setattr(supervisor, "_engine_ready", ready)
    result = asyncio.run(supervisor.switch_model("qwen3-tts-1.7b-base-bf16"))

    assert result["state"] == "loaded"
    assert result["events"][-1]["action"] == "load-noop"


def test_wav_and_pcm_format_rendering_preserve_master_metadata(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    with io.BytesIO() as stream:
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b"\0\0" * 240)
        master = stream.getvalue()

    wav_bytes, wav_type, wav_ext, metadata = supervisor._render_master_wav(master, "wav")
    pcm_bytes, pcm_type, pcm_ext, _ = supervisor._render_master_wav(master, "pcm")

    assert wav_bytes == master
    assert (wav_type, wav_ext) == ("audio/wav", "wav")
    assert (pcm_type, pcm_ext, len(pcm_bytes)) == ("audio/pcm", "pcm", 480)
    assert metadata["sampleRate"] == 24000
    assert metadata["container"] == "WAV"
    assert metadata["quality"] == "native lossless master"


def test_gpu_guard_disabled_is_an_explicit_bypass(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    supervisor.gpu_guard_settings.update(mode="disabled")
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *args, **kwargs: "128, 99\n")

    result = supervisor.gpu_guard("qwen3-tts-1.7b-base-bf16", operation="synthesis")

    assert result["ok"] is True
    assert result["bypassed"] is True
    assert result["guardMode"] == "disabled"


def test_gpu_guard_settings_persist_atomically(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    result = supervisor._update_gpu_guard(
        {
            "mode": "custom",
            "load_min_free_mib": 4096,
            "synthesis_min_free_mib": 512,
            "load_max_utilization_percent": 90,
            "synthesis_max_utilization_percent": 98,
        }
    )

    saved = json.loads((tmp_path / "voices" / "candidate_gpu_guard.json").read_text(encoding="utf-8"))
    assert result["gpu_guard"]["mode"] == "custom"
    assert saved["load_min_free_mib"] == 4096
    assert saved["synthesis_min_free_mib"] == 512


def test_streaming_playground_uses_llamacpp_input_audio_and_host_gateway(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "_transcode_llm_audio_to_wav", lambda audio, source_format: b"RIFFconverted")

    endpoint = supervisor._container_reachable_llm_endpoint("http://127.0.0.1:8818/v1/chat/completions")
    part = supervisor._llamacpp_audio_part("data:audio/wav;base64,UklGRg==")
    history = supervisor._llamacpp_history(
        [{"role": "user", "audio_data_url": "data:audio/wav;base64,UklGRg=="}, {"role": "assistant", "content": "Hello"}]
    )

    assert endpoint == "http://host.docker.internal:8818/v1/chat/completions"
    assert part == {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
    assert history == [{"role": "user", "content": "[Earlier spoken user turn]"}, {"role": "assistant", "content": "Hello"}]


def test_streaming_playground_transcodes_browser_audio_to_requested_format(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    calls = []

    def transcode(audio, source_format, target_format):
        calls.append((audio, source_format, target_format))
        return b"converted-audio"

    monkeypatch.setattr(supervisor, "_transcode_llm_audio", transcode)
    part = supervisor._llamacpp_audio_part(
        "data:audio/webm;codecs=opus;base64,d2VibS1yZWNvcmRpbmc=", requested_format="wav"
    )

    assert calls == [(b"webm-recording", "webm", "wav")]
    assert part == {"type": "input_audio", "input_audio": {"data": "Y29udmVydGVkLWF1ZGlv", "format": "wav"}}

    part = supervisor._llamacpp_audio_part("data:audio/wav;base64,UklGRg==", requested_format="mp3")
    assert calls[-1] == (b"RIFF", "wav", "mp3")
    assert part["input_audio"]["format"] == "mp3"


def test_streaming_playground_normalizes_a_bare_llamacpp_server_url(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)

    assert supervisor._container_reachable_llm_endpoint("http://127.0.0.1:8818") == (
        "http://host.docker.internal:8818/v1/chat/completions"
    )
    assert supervisor._container_reachable_llm_endpoint("http://192.168.1.10:8818/v1") == (
        "http://192.168.1.10:8818/v1/chat/completions"
    )


def test_streaming_playground_settings_persist_without_api_key(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    saved = supervisor._write_voice_studio_settings(
        {
            "endpoint": "http://127.0.0.1:8818/v1/chat/completions",
            "model": "gemma",
            "system_prompt": "Answer briefly.",
            "mic_id": "default-device",
            "vad": {"silence-delay": 900},
            "phrase": {"phrase-min": 24},
            "api_key": "must-not-persist",
        }
    )

    assert saved["endpoint"].endswith("/chat/completions")
    assert saved["vad"]["silence-delay"] == "900"
    assert "api_key" not in saved
    assert "must-not-persist" not in (tmp_path / "voices" / "voice_studio_settings.json").read_text(encoding="utf-8")
