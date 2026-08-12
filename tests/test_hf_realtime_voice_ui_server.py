import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


def _load_ui_server_module():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    sys.path.insert(0, str(ui_dir))
    try:
        spec = importlib.util.spec_from_file_location("hf_realtime_voice_server", ui_dir / "server.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(ui_dir))
        except ValueError:
            pass


def _mock_serper(monkeypatch, server, payloads, *, status_code=200):
    calls = []
    queued = list(payloads)

    class Response:
        def __init__(self, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, headers, json):
            calls.append({"url": url, "headers": headers, "json": dict(json)})
            return Response(queued.pop(0))

    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    return calls


def _response_json(response):
    return json.loads(response.body.decode("utf-8"))


def test_ui_config_runtime_identity_is_bounded_and_process_immutable(monkeypatch):
    revision = "a" * 40
    fingerprint = "b" * 64
    assets = "main=34-opaque-echo-route;ws=21-opaque-echo-route;chat=5-opaque-echo-route;playback=16-adaptive-safe-start"
    monkeypatch.setenv("S2S_RUNTIME_REVISION", revision)
    monkeypatch.setenv("S2S_RUNTIME_DIRTY", "0")
    monkeypatch.setenv("S2S_RUNTIME_SOURCE_FINGERPRINT", fingerprint)
    monkeypatch.setenv("S2S_UI_ASSET_GENERATION", assets)
    server = _load_ui_server_module()

    expected = {
        "source_revision": revision,
        "source_dirty": False,
        "source_fingerprint": fingerprint,
        "ui_asset_generation": assets,
    }
    assert server.config()["runtime"] == expected
    monkeypatch.setenv("S2S_RUNTIME_REVISION", "c" * 40)
    assert server.config()["runtime"] == expected


@pytest.mark.parametrize(
    ("name", "value", "field"),
    [
        ("S2S_RUNTIME_REVISION", "not-a-revision", "source_revision"),
        ("S2S_RUNTIME_DIRTY", "maybe", "source_dirty"),
        ("S2S_RUNTIME_SOURCE_FINGERPRINT", "short", "source_fingerprint"),
        ("S2S_UI_ASSET_GENERATION", "main=" + ("x" * 300) + ";ws=1;chat=1;playback=1", "ui_asset_generation"),
    ],
)
def test_ui_config_runtime_identity_fails_closed(name, value, field, monkeypatch):
    monkeypatch.setenv("S2S_RUNTIME_REVISION", "a" * 40)
    monkeypatch.setenv("S2S_RUNTIME_DIRTY", "0")
    monkeypatch.setenv("S2S_RUNTIME_SOURCE_FINGERPRINT", "b" * 64)
    monkeypatch.setenv("S2S_UI_ASSET_GENERATION", "main=34;ws=21;chat=5;playback=16")
    monkeypatch.setenv(name, value)
    server = _load_ui_server_module()
    expected = None if field == "source_dirty" else "unknown"
    assert server.config()["runtime"][field] == expected


def test_search_defaults_to_unfiltered_web_and_returns_truthful_versioned_results(monkeypatch):
    server = _load_ui_server_module()
    calls = _mock_serper(
        monkeypatch,
        server,
        [
            {
                "answerBox": {"answer": "A bounded direct answer"},
                "organic": [
                    {
                        "title": "Reference",
                        "snippet": "A factual snippet.",
                        "link": "https://example.test/reference",
                        "position": 4,
                    }
                ],
            }
        ],
    )

    response = asyncio.run(server.search(server.SearchRequest(query="reference topic", key="test-key")))
    payload = _response_json(response)

    assert calls == [
        {
            "url": "https://google.serper.dev/search",
            "headers": {"X-API-KEY": "test-key", "Content-Type": "application/json"},
            "json": {"q": "reference topic", "num": server.MAX_RESULTS},
        }
    ]
    assert payload["type"] == "web_search_result"
    assert payload["schema_version"] == 1
    assert payload["provider"] == "serper"
    assert payload["requested_mode"] == "auto"
    assert payload["effective_mode"] == "web"
    assert payload["freshness"] == "none"
    assert payload["recency_filter_applied"] is False
    assert payload["fallback_applied"] is False
    assert payload["answer"] == "A bounded direct answer"
    assert payload["results"] == [
        {
            "title": "Reference",
            "snippet": "A factual snippet.",
            "url": "https://example.test/reference",
            "date": None,
            "source": None,
            "position": 4,
        }
    ]
    assert payload["retrieved_at_utc"].endswith("Z")
    assert "today" not in json.dumps(payload).lower()
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"


@pytest.mark.parametrize(
    ("mode", "freshness", "endpoint", "qdr", "result_key"),
    [
        ("auto", "day", "news", "qdr:d", "news"),
        ("auto", "week", "news", "qdr:w", "news"),
        ("web", "month", "search", "qdr:m", "organic"),
        ("news", "year", "news", "qdr:y", "news"),
        ("news", "none", "news", None, "news"),
    ],
)
def test_search_maps_mode_and_freshness_to_serper_endpoint_and_qdr(
    monkeypatch, mode, freshness, endpoint, qdr, result_key
):
    server = _load_ui_server_module()
    calls = _mock_serper(
        monkeypatch,
        server,
        [{result_key: [{"title": "Dated", "link": "https://example.test", "date": "2 hours ago", "source": "Wire"}]}],
    )

    response = asyncio.run(
        server.search(server.SearchRequest(query="dated topic", mode=mode, freshness=freshness, key="test-key"))
    )
    payload = _response_json(response)

    assert calls[0]["url"] == f"https://google.serper.dev/{endpoint}"
    assert calls[0]["json"] == {
        "q": "dated topic",
        "num": server.MAX_RESULTS,
        **({"tbs": qdr} if qdr else {}),
    }
    assert payload["requested_mode"] == mode
    assert payload["effective_mode"] == ("news" if endpoint == "news" else "web")
    assert payload["freshness"] == freshness
    assert payload["recency_filter_applied"] is (qdr is not None)
    assert payload["answer"] is None
    assert payload["results"][0]["date"] == "2 hours ago"
    assert payload["results"][0]["source"] == "Wire"


def test_empty_effective_news_uses_one_same_filter_web_fallback_without_answer_panel(monkeypatch):
    server = _load_ui_server_module()
    calls = _mock_serper(
        monkeypatch,
        server,
        [
            {"news": [], "answerBox": {"answer": "must not leak"}},
            {
                "organic": [{"title": "Fallback", "link": "https://example.test/fallback"}],
                "answerBox": {"answer": "also must not leak"},
            },
        ],
    )

    response = asyncio.run(
        server.search(server.SearchRequest(query="recent topic", mode="auto", freshness="week", key="test-key"))
    )
    payload = _response_json(response)

    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == ["news", "search"]
    assert [call["json"]["tbs"] for call in calls] == ["qdr:w", "qdr:w"]
    assert payload["requested_mode"] == "auto"
    assert payload["effective_mode"] == "web"
    assert payload["fallback_applied"] is True
    assert payload["recency_filter_applied"] is True
    assert payload["answer"] is None
    assert payload["results"][0]["title"] == "Fallback"


def test_empty_explicit_news_stays_news_without_web_fallback(monkeypatch):
    server = _load_ui_server_module()
    calls = _mock_serper(monkeypatch, server, [{"news": []}])

    response = asyncio.run(
        server.search(server.SearchRequest(query="recent topic", mode="news", freshness="week", key="test-key"))
    )
    payload = _response_json(response)

    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == ["news"]
    assert calls[0]["json"]["tbs"] == "qdr:w"
    assert payload["requested_mode"] == "news"
    assert payload["effective_mode"] == "news"
    assert payload["fallback_applied"] is False
    assert payload["results"] == []
    assert payload["answer"] is None


def test_search_rejects_invalid_or_oversized_inputs_before_provider_call(monkeypatch):
    server = _load_ui_server_module()
    with pytest.raises(ValidationError):
        server.SearchRequest(query="topic", mode="images", key="test-key")
    with pytest.raises(ValidationError):
        server.SearchRequest(query="topic", freshness="hour", key="test-key")
    with pytest.raises(ValidationError):
        server.SearchRequest(query="topic", extra_field="private", key="test-key")
    with pytest.raises(HTTPException, match="Empty query"):
        asyncio.run(server.search(server.SearchRequest(query="   ", key="test-key")))
    with pytest.raises(HTTPException, match="too long"):
        asyncio.run(server.search(server.SearchRequest(query="q" * (server.MAX_QUERY_CHARS + 1), key="test-key")))


def test_search_bounds_provider_strings_and_provider_failure_is_content_free(monkeypatch, caplog):
    server = _load_ui_server_module()
    calls = _mock_serper(
        monkeypatch,
        server,
        [{"organic": [{"title": "t" * 500, "snippet": "s" * 2000, "link": "u" * 3000, "date": "d" * 200, "source": "x" * 400}]}],
    )
    payload = _response_json(
        asyncio.run(server.search(server.SearchRequest(query="bounded", key="test-key")))
    )
    result = payload["results"][0]
    assert len(result["title"]) == server.MAX_RESULT_TITLE_CHARS
    assert len(result["snippet"]) == server.MAX_RESULT_SNIPPET_CHARS
    assert len(result["url"]) == server.MAX_RESULT_URL_CHARS
    assert len(result["date"]) == server.MAX_RESULT_DATE_CHARS
    assert len(result["source"]) == server.MAX_RESULT_SOURCE_CHARS
    assert len(calls) == 1

    secret_query = "private query sentinel"
    secret_body = "private provider body sentinel"
    _mock_serper(monkeypatch, server, [{"message": secret_body, "query": secret_query}], status_code=429)
    with caplog.at_level("WARNING"), pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.search(server.SearchRequest(query=secret_query, key="test-key")))
    public_failure = str(exc_info.value.detail) + caplog.text
    assert secret_query not in public_failure
    assert secret_body not in public_failure


def test_voice_library_uses_portable_default_and_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICE_LIBRARY_DIR", raising=False)
    server = _load_ui_server_module()
    expected = Path.home() / ".speech-to-speech" / "qwen3-tts-voices"

    assert server.DEFAULT_VOICE_LIBRARY_DIR == expected
    assert server.DEFAULT_VOICE_LIBRARY_DIR.relative_to(Path.home()) == Path(
        ".speech-to-speech/qwen3-tts-voices"
    )

    override = tmp_path / "shared-voices"
    monkeypatch.setenv("VOICE_LIBRARY_DIR", str(override))
    overridden_server = _load_ui_server_module()
    assert overridden_server.DEFAULT_VOICE_LIBRARY_DIR == override


def test_load_base_clone_profiles_filters_to_base_profiles(tmp_path):
    server = _load_ui_server_module()
    profiles = tmp_path / "profiles"
    base = profiles / "alpha-base-0001"
    custom = profiles / "not-base"
    base.mkdir(parents=True)
    custom.mkdir()
    (base / "meta.json").write_text(
        json.dumps(
            {
                "profile_id": "alpha-base-0001",
                "name": "Alpha Voice",
                "task_type": "Base",
                "created_at": "2026-05-28T05:29:38Z",
                "ref_text": "Systems are now fully operational.",
                "language": "Auto",
            }
        ),
        encoding="utf-8",
    )
    (custom / "meta.json").write_text(
        json.dumps({"profile_id": "custom", "name": "Vivian", "task_type": "CustomVoice"}),
        encoding="utf-8",
    )

    voices = server._load_base_clone_profiles(tmp_path)

    assert voices == [
        {
            "id": "alpha-base-0001",
            "voice": "clone:alpha-base-0001",
            "name": "Alpha Voice",
            "task_type": "Base",
            "created_at": "2026-05-28T05:29:38Z",
            "ref_text": "Systems are now fully operational.",
            "language": "Auto",
        }
    ]


def test_audio_cpp_inventory_matches_candidate_live_backend_without_cross_filtering(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    _base_profile(tmp_path, "alpha-base-0001")
    _base_profile(tmp_path, "47a4e1ef5258")
    monkeypatch.setattr(server, "DEFAULT_VOICE_LIBRARY_DIR", tmp_path)
    payload = {
        "defaultVoice": "clone:alpha-base-0001",
        "selectedVoice": "clone:776a491528c9",
        "voices": [
            {"id": "alpha-base-0001", "voice": "clone:alpha-base-0001", "name": "Faster copy", "task_type": "Base"},
            {"id": "22fb07ef3a80", "voice": "clone:22fb07ef3a80", "name": "Candidate A", "task_type": "Base"},
            {"id": "47a4e1ef5258", "voice": "clone:47a4e1ef5258", "name": "Faster copy 2", "task_type": "Base"},
            {"id": "776a491528c9", "voice": "clone:776a491528c9", "name": "Candidate B", "task_type": "Base"},
        ],
    }

    scoped = server._scope_audio_cpp_profile_response(payload)

    assert [profile["voice"] for profile in scoped["voices"]] == [
        "clone:alpha-base-0001",
        "clone:22fb07ef3a80",
        "clone:47a4e1ef5258",
        "clone:776a491528c9",
    ]
    assert scoped["defaultVoice"] == "clone:alpha-base-0001"
    assert scoped["selectedVoice"] == "clone:776a491528c9"


def test_live_backend_voice_inventory_disables_http_caching(monkeypatch):
    server = _load_ui_server_module()

    async def inventory(backend):
        return {"backend": backend, "reachable": False, "voices": []}

    monkeypatch.setattr(server, "_backend_voice_inventory", inventory)
    response = server.Response()
    result = asyncio.run(server.backend_voices("qwen3tts-audiocpp", response))

    assert result["backend"] == "qwen3tts-audiocpp"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"


def _base_profile(library: Path, profile_id: str = "alpha-base-0001", name: str = "Alpha Voice") -> None:
    profile_dir = library / "profiles" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "ref_audio.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (profile_dir / "meta.json").write_text(
        json.dumps({
            "profile_id": profile_id,
            "name": name,
            "task_type": "Base",
            "language": "Auto",
            "ref_text": "Original reference.",
            "ref_audio_filename": "ref_audio.wav",
        }),
        encoding="utf-8",
    )


def test_profile_response_prefers_valid_selection_then_first_live_base_or_none(tmp_path):
    server = _load_ui_server_module()
    _base_profile(tmp_path, "beta-base-0002", "Beta Voice")
    _base_profile(tmp_path, "alpha-base-0001", "Alpha Voice")
    selected_path = tmp_path / "selected_profile.json"

    selected_path.write_text(json.dumps({"profile_id": "beta-base-0002"}), encoding="utf-8")
    selected = server._profile_response(tmp_path)
    assert selected["defaultVoice"] == "clone:alpha-base-0001"
    assert selected["selectedVoice"] == "clone:beta-base-0002"

    selected_path.write_text(json.dumps({"profile_id": "stale-base-0099"}), encoding="utf-8")
    fallback = server._profile_response(tmp_path)
    assert fallback["defaultVoice"] == "clone:alpha-base-0001"
    assert fallback["selectedVoice"] == "clone:alpha-base-0001"

    empty = server._profile_response(tmp_path / "empty")
    assert empty["defaultVoice"] is None
    assert empty["selectedVoice"] is None
    assert empty["voices"] == []


def test_selected_profile_can_be_deleted_without_a_privileged_identity(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    _base_profile(tmp_path)
    monkeypatch.setattr(server, "DEFAULT_VOICE_LIBRARY_DIR", tmp_path)
    server.select_qwen3_profile(server.ProfileSelectRequest(profile_id="alpha-base-0001"))

    deleted = server.delete_qwen3_profile("alpha-base-0001")

    assert deleted["voices"] == []
    assert deleted["selectedVoice"] is None


def test_clone_profile_management_persists_to_configured_library(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    _base_profile(tmp_path)
    monkeypatch.setattr(server, "DEFAULT_VOICE_LIBRARY_DIR", tmp_path)

    created = server.create_qwen3_profile(
        server.ProfileCreateRequest(name="Copied", ref_text="Copied reference.", source_profile_id="alpha-base-0001")
    )
    copied = next(profile for profile in created["voices"] if profile["name"] == "Copied")
    assert (tmp_path / "profiles" / copied["id"] / "ref_audio.wav").is_file()

    edited = server.edit_qwen3_profile(copied["id"], server.ProfileEditRequest(name="Renamed", language="English"))
    assert any(profile["name"] == "Renamed" and profile["language"] == "English" for profile in edited["voices"])

    selected = server.select_qwen3_profile(server.ProfileSelectRequest(profile_id=copied["id"]))
    assert selected["selectedVoice"] == copied["voice"]
    assert json.loads((tmp_path / "selected_profile.json").read_text(encoding="utf-8"))["profile_id"] == copied["id"]

    deleted = server.delete_qwen3_profile(copied["id"])
    assert copied["id"] not in {profile["id"] for profile in deleted["voices"]}


def test_profile_import_requires_wav_and_preserves_base_metadata(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "DEFAULT_VOICE_LIBRARY_DIR", tmp_path)
    encoded = __import__("base64").b64encode(b"RIFF\x00\x00\x00\x00WAVEpayload").decode("ascii")

    result = server.import_qwen3_profile(
        server.ProfileImportRequest(name="Imported", ref_text="Transcript.", language="English", audio_base64=encoded)
    )

    profile = next(profile for profile in result["voices"] if profile["name"] == "Imported")
    metadata = json.loads((tmp_path / "profiles" / profile["id"] / "meta.json").read_text(encoding="utf-8"))
    assert metadata["task_type"] == "Base"
    assert metadata["ref_audio_filename"] == "ref_audio.wav"


def test_faster_model_inventory_does_not_fake_lifecycle_controls(monkeypatch):
    server = _load_ui_server_module()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"model_loaded": True, "capabilities": {"clone_only": True}}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, _url):
            return Response()

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: Client())
    inventory = asyncio.run(server._faster_model_inventory())
    assert inventory["activeModel"] == "1.7B-Base"
    assert inventory["models"][0]["loaded"] is True
    assert inventory["controls"] == {
        "inventory": True, "load": False, "switch": False, "unload": False,
        "reason": "The stable FasterQwen3TTS API is clone-only and exposes no model lifecycle endpoints.",
    }


def test_audio_cpp_candidate_needs_explicit_capability_validation(monkeypatch):
    server = _load_ui_server_module()

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return Response(
                {"data": [{"id": "qwen3-tts-0.6b-base-bf16"}, {"id": "qwen3-tts-1.7b-base-bf16"}]}
                if url.endswith("/models") else {
                    "status": "ok",
                    "backend": {
                        "model_id": "qwen3-tts-1.7b-base-bf16",
                        "runtime": {"state": "loaded", "progressive_phrase_pcm": True, "native_incremental_pcm": False},
                    },
                }
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: Client())
    result = asyncio.run(server._probe_audio_cpp_candidate(run_speech_probe=False))

    assert result["reachable"] is True
    assert result["currentModel"] == "qwen3-tts-1.7b-base-bf16"
    assert result["ready"] is False
    assert result["streaming"] is False
    assert result["progressivePcm"] is True
    assert result["nativeIncrementalPcm"] is False
    assert result["deliveryMode"] == "progressive-buffered-pcm"
    assert result["bufferedFallback"] is True
    assert "Validate the loaded audio.cpp model" in result["error"]


def test_audio_cpp_validation_labels_verified_native_pcm(monkeypatch):
    server = _load_ui_server_module()
    seen_payloads = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class SpeechResponse:
        headers = {
            "content-type": "audio/pcm",
            "x-tts-streaming-mode": "native-incremental-pcm",
        }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"\x00\x00" * 32
            yield b"\x01\x00" * 32

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/models"):
                return Response({"data": [{"id": "qwen3-tts-1.7b-base-bf16"}]})
            if url.endswith("/voices"):
                return Response({"data": [{"id": "candidate-profile"}]})
            return Response({
                "status": "ok",
                "backend": {
                    "model_id": "qwen3-tts-1.7b-base-bf16",
                    "runtime": {
                        "state": "loaded",
                        "progressive_phrase_pcm": True,
                        "native_incremental_pcm": True,
                        "sample_rate": 24000,
                    },
                },
            })

        def stream(self, _method, _url, *, json):
            seen_payloads.append(json)
            return SpeechResponse()

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **_kwargs: Client())

    result = asyncio.run(
        server._probe_audio_cpp_candidate(run_speech_probe=True, profile_id="candidate-profile")
    )

    assert seen_payloads == [{
        "model": "qwen3-tts-1.7b-base-bf16",
        "input": "Capability check.",
        "voice": "clone:candidate-profile",
        "response_format": "pcm",
        "stream": True,
    }]
    assert result["ready"] is True
    assert result["nativeStreaming"] is True
    assert result["mode"] == "native-incremental-pcm"
    assert result["bufferedFallback"] is False
    assert result["speechProbe"]["chunks"] == 2
    assert result["speechProbe"]["bytes"] == 128
    assert result["speechProbe"]["streamingModeHeader"] == "native-incremental-pcm"
    assert result["speechProbe"]["nativeHeaderValid"] is True
    assert result["speechProbe"]["incrementalEvidence"] is True


@pytest.mark.parametrize(
    ("streaming_header", "speech_chunks", "failure_text"),
    [
        (None, [b"first", b"second"], "exact X-TTS-Streaming-Mode"),
        ("progressive-buffered-pcm", [b"first", b"second"], "exact X-TTS-Streaming-Mode"),
        ("native-incremental-pcm", [b"complete-buffered-body"], "fewer than two nonempty PCM chunks"),
    ],
)
def test_audio_cpp_validation_does_not_relabel_buffered_body_as_native(
    monkeypatch, streaming_header, speech_chunks, failure_text
):
    server = _load_ui_server_module()

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class SpeechResponse:
        def __init__(self):
            self.headers = {"content-type": "audio/pcm"}
            if streaming_header is not None:
                self.headers["x-tts-streaming-mode"] = streaming_header

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            for chunk in speech_chunks:
                yield chunk

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/models"):
                return Response({"data": [{"id": "qwen3-tts-1.7b-base-bf16"}]})
            if url.endswith("/voices"):
                return Response({"data": [{"id": "candidate-profile"}]})
            return Response({
                "status": "ok",
                "backend": {
                    "model_id": "qwen3-tts-1.7b-base-bf16",
                    "runtime": {
                        "state": "loaded",
                        "progressive_phrase_pcm": True,
                        "native_incremental_pcm": True,
                    },
                },
            })

        def stream(self, _method, _url, *, json):
            return SpeechResponse()

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **_kwargs: Client())

    result = asyncio.run(
        server._probe_audio_cpp_candidate(run_speech_probe=True, profile_id="candidate-profile")
    )

    assert result["ready"] is True
    assert result["nativeStreaming"] is False
    assert result["streaming"] is False
    assert result["bufferedSpeech"] is True
    assert result["bufferedFallback"] is True
    assert result["deliveryMode"] == "buffered-fallback"
    assert failure_text in result["nativeProbeError"]
    assert result["speechProbe"]["nativeHeaderValid"] is (
        streaming_header == "native-incremental-pcm"
    )
    assert result["speechProbe"]["incrementalEvidence"] is (len(speech_chunks) >= 2)


def test_audio_cpp_saved_fallback_is_not_relabelled_native_without_a_new_probe(monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(
        server,
        "_read_public_ui_settings",
        lambda: {
            "ttsBackend": "qwen3tts-audiocpp",
            "voice": "clone:candidate-profile",
            "voiceByBackend": {"qwen3tts-audiocpp": "clone:candidate-profile"},
        },
    )
    monkeypatch.setattr(
        server,
        "_find_tts_validation",
        lambda *_args: {
            "model": "qwen3-tts-1.7b-base-bf16",
            "voice": "clone:candidate-profile",
            "speech": True,
            "deliveryMode": "buffered-fallback",
            "nativeStreaming": False,
        },
    )

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/health"):
                return Response({
                    "backend": {
                        "model_id": "qwen3-tts-1.7b-base-bf16",
                        "runtime": {
                            "state": "loaded",
                            "progressive_phrase_pcm": True,
                            "native_incremental_pcm": True,
                        },
                    }
                })
            if url.endswith("/models"):
                return Response({"data": [{"id": "qwen3-tts-1.7b-base-bf16"}]})
            return Response({"data": [{"id": "candidate-profile"}]})

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **_kwargs: Client())

    result = asyncio.run(server._probe_audio_cpp_candidate(run_speech_probe=False))

    assert result["ready"] is True
    assert result["nativeIncrementalPcm"] is True
    assert result["nativeStreaming"] is False
    assert result["deliveryMode"] == "buffered-fallback"
    assert result["bufferedFallback"] is True
    assert "revalidate to prove native chunks" in " ".join(result["limitations"])


def test_audio_cpp_same_origin_speech_proxy_relays_chunks_and_closes_on_disconnect(monkeypatch):
    server = _load_ui_server_module()

    class Upstream:
        status_code = 200
        headers = {
            "content-type": "audio/pcm",
            "x-tts-streaming-mode": "native-incremental-pcm",
        }
        is_error = False
        closed = False

        async def aiter_raw(self):
            yield b"first"
            yield b"second"

        async def aclose(self):
            self.closed = True

    upstream = Upstream()

    class Client:
        closed = False

        def build_request(self, method, url, *, json):
            assert method == "POST"
            assert url.endswith("/audio/speech")
            assert json["stream"] is True
            return object()

        async def send(self, _request, *, stream):
            assert stream is True
            return upstream

        async def aclose(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **_kwargs: client)

    class Request:
        checks = 0

        async def json(self):
            return {"stream": True, "response_format": "pcm"}

        async def is_disconnected(self):
            self.checks += 1
            return self.checks > 1

    response = asyncio.run(server.proxy_audio_cpp_speech(Request()))

    async def collect():
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())
    assert chunks == [b"first"]
    assert response.headers["x-tts-streaming-mode"] == "native-incremental-pcm"
    assert upstream.closed is True
    assert client.closed is True


def test_voice_studio_test_uses_root_health_and_public_clone_contract(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    _base_profile(tmp_path, "candidate-profile")
    monkeypatch.setattr(server, "DEFAULT_VOICE_LIBRARY_DIR", tmp_path)
    urls = []
    speech_payload = {}

    class Response:
        def __init__(self, payload=None, content=b"RIFFaudio", headers=None):
            self.payload = payload or {}
            self.content = content
            self.headers = headers or {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            urls.append(url)
            return Response({
                "backend": {
                    "model_id": "qwen3-tts-1.7b-base-bf16",
                    "runtime": {"state": "loaded"},
                }
            })

        async def post(self, url, json):
            urls.append(url)
            if url.endswith("/audio/speech"):
                speech_payload.update(json)
                return Response(content=b"RIFFaudio", headers={"content-type": "audio/wav"})
            return Response()

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **_kwargs: Client())
    request = server.VoiceStudioSynthesisRequest(
        model_id="qwen3-tts-1.7b-base-bf16",
        profile_id="candidate-profile",
        text="Test phrase.",
    )

    response = asyncio.run(server.voice_studio_test(request))

    assert "http://127.0.0.1:8890/health" in urls
    assert speech_payload == {
        "model": "qwen3-tts-1.7b-base-bf16",
        "input": "Test phrase.",
        "voice": "clone:candidate-profile",
        "response_format": "wav",
        "language": "Auto",
        "stream": False,
    }
    assert response.body == b"RIFFaudio"
    assert response.media_type == "audio/wav"


def test_load_base_clone_profiles_handles_missing_library(tmp_path):
    server = _load_ui_server_module()

    assert server._load_base_clone_profiles(tmp_path / "missing") == []


def test_tts_backend_status_reports_faster_and_voice_studio_model(monkeypatch):
    server = _load_ui_server_module()

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/health"):
                return Response(
                    {
                        "model_loaded": True,
                        "capabilities": {"native_pcm_streaming": True, "clone_only": True},
                    }
                )
            return Response(
                {"state": "loaded", "current": "0.6B-Base", "loaded_models": ["0.6B-Base"]}
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: Client())

    faster = asyncio.run(server._probe_tts_backend("faster", server.TTS_BACKENDS["faster"]))
    groxaxo = asyncio.run(server._probe_tts_backend("groxaxo", server.TTS_BACKENDS["groxaxo"]))

    assert faster["ready"] is True
    assert faster["currentModel"] == "1.7B-Base"
    assert groxaxo["ready"] is True
    assert groxaxo["currentModel"] == "0.6B-Base"


def test_local_ui_identity_keeps_upstream_credit_and_shows_local_provider_slots():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    html = (ui_dir / "index.html").read_text(encoding="utf-8")
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")

    assert "Built by" in html
    assert "Modified by" in html
    assert "Omni-NexusAI" in html
    assert "https://github.com/Omni-NexusAI/speech-to-speech" in html
    assert 'id="local-provider-gemma"' in html
    assert 'id="local-provider-tts"' in html
    assert 'set("local-provider-gemma"' in main_js
    assert 'set("local-provider-tts"' in main_js


def test_live_transcript_toggle_only_controls_the_floating_user_bubble():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    chat_js = (ui_dir / "ui" / "chat.js").read_text(encoding="utf-8")

    assert "showUserBubble: settings.liveTranscript" in main_js
    assert "if (options.showUserBubble && (d.partial" in chat_js
    assert "this._pendingUserHist || this._appendHistMsg" in chat_js
    assert "Speech could not be transcribed." not in main_js
    assert "discardPendingUserTurn" not in chat_js


def test_tool_output_is_acknowledged_before_one_post_tool_response():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert main_js.index("sessionClient.sendToolOutput") < main_js.index("sessionClient.requestToolResponse")
    assert main_js.index("sessionClient.requestToolResponse") < main_js.index("await outputAck")
    assert "requestToolResponse(opts = {})" in client_js
    assert 'type: "response.create"' in client_js
    assert "...(opts.tools ? { tools: opts.tools } : {})" in client_js
    assert "...(opts.toolChoice ? { tool_choice: opts.toolChoice } : {})" in client_js


def test_local_ui_teardown_isolates_closed_client_events():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert "if (client !== c) return;" in main_js
    assert "const closingClient = client;\n  client = null;" in main_js
    assert "runTool(c, name, callId, lifecycle, prepared)" in main_js
    assert 'if (this._closed && status !== "closed") return;' in client_js
    assert 'this._captureNode?.port.postMessage({ kind: "enable", value: false });' in client_js
    assert 'this._invalidatePlayback("stop")' in client_js
    assert 'kind: "clear",' in client_js
    assert 'generation: this._playbackGeneration' in client_js


def test_diagnostics_use_transcription_and_dynamic_context_tokens():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")

    assert '"transcription"' in main_js
    assert '"gemma_preview"' not in main_js
    assert "history_tokens" in main_js
    assert "contextWindow" in main_js


def test_remote_model_connection_test_redacts_key(monkeypatch):
    server = _load_ui_server_module()
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            calls.append(("get", url))
            if url.endswith("/models"):
                return Response({"data": [{"id": "remote-audio-model"}]})
            return Response({"default_generation_settings": {"n_ctx": 32768}})

    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    result = asyncio.run(
        server.test_model_endpoint(
            server.ModelTestRequest(
                provider="remote",
                base_url="http://10.0.0.60:8080",
                model="configured-model",
                api_key="secret",
            )
        )
    )

    assert result["model"] == "remote-audio-model"
    assert result["context_window"] == 32768
    assert result["api_key_set"] is True
    assert "api_key" not in result
    assert calls[0][1]["headers"] == {"Authorization": "Bearer secret"}
    assert all("127.0.0.1:8818" not in str(call) for call in calls)


def test_hf_ui_persistent_preferences_are_atomic_and_exclude_api_keys(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "UI_SETTINGS_PATH", tmp_path / "hf_realtime_ui_settings.json")
    many_routes = {
        f"route_{index:064x}": {"delayMs": index} for index in range(18)
    }

    result = server._write_public_ui_settings(
        {
            "ttsBackend": "qwen3tts-audiocpp",
            "ttsProfileByBackend": {
                "qwen3tts-audiocpp": "low-latency",
                "faster": "provider-default",
                "foreign": "must-not-persist",
            },
            "modelUrl": "http://127.0.0.1:8818",
            "fullBufferTts": False,
            "echoCalibrations": {
                "mic-a::output-b": {"delayMs": 99},
                f"route_{'b' * 64}": {"delayMs": math.nan},
                f"route_{'c' * 64}": {"delayMs": 10**10000},
                **many_routes,
                f"route_{'a' * 64}": {
                    "delayMs": 72,
                    "suppressionStrength": 0.8,
                    "leakageThreshold": 4,
                    "doubleTalkSensitivity": -1,
                    "echoTailMs": 0,
                    "unexpected": 99,
                },
            },
            "modelApiKey": "must-not-persist",
        }
    )

    saved = json.loads(server.UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert result["ttsBackend"] == "qwen3tts-audiocpp"
    assert saved["modelUrl"] == "http://127.0.0.1:8818"
    assert saved["ttsProfileByBackend"] == {
        "qwen3tts-audiocpp": "low-latency",
        "faster": "provider-default",
    }
    assert saved["echoCalibrations"][f"route_{'a' * 64}"] == {
        "delayMs": 72.0,
        "suppressionStrength": 0.8,
        "leakageThreshold": 1.0,
        "doubleTalkSensitivity": 0.0,
        "echoTailMs": 350.0,
    }
    assert "mic-a::output-b" not in saved["echoCalibrations"]
    assert f"route_{'b' * 64}" not in saved["echoCalibrations"]
    assert f"route_{'c' * 64}" not in saved["echoCalibrations"]
    assert len(saved["echoCalibrations"]) == 16
    assert f"route_{0:064x}" not in saved["echoCalibrations"]
    assert "modelApiKey" not in saved
    assert "must-not-persist" not in server.UI_SETTINGS_PATH.read_text(encoding="utf-8")
    assert not server.UI_SETTINGS_PATH.with_suffix(".tmp").exists()


def test_hf_ui_echo_route_order_survives_write_read_and_newest_eviction(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "UI_SETTINGS_PATH", tmp_path / "hf_realtime_ui_settings.json")
    initial = {
        f"route_{index:064x}": {"delayMs": index}
        for index in range(16, 32)
    }
    server._write_public_ui_settings({"echoCalibrations": initial})
    loaded = server._read_public_ui_settings()["echoCalibrations"]
    newest = f"route_{'0' * 63}a"
    loaded[newest] = {"delayMs": 77}
    server._write_public_ui_settings({"echoCalibrations": loaded})
    reloaded = server._read_public_ui_settings()["echoCalibrations"]

    assert list(reloaded)[-1] == newest
    assert f"route_{16:064x}" not in reloaded
    assert f"route_{31:064x}" in reloaded


def test_hf_ui_read_scrubs_current_and_legacy_raw_echo_routes(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    current = tmp_path / "current" / "hf_realtime_ui_settings.json"
    legacy = tmp_path / "legacy" / "hf_realtime_ui_settings.json"
    current.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    payload = {"echoCalibrations": {"mic-a::output-b": {"delayMs": 99}}}
    current.write_text(json.dumps(payload), encoding="utf-8")
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(server, "UI_SETTINGS_PATH", current)
    monkeypatch.setattr(server, "_LEGACY_UI_SETTINGS_PATH", legacy)
    monkeypatch.setattr(server, "_ui_settings_default", current)

    assert server._read_public_ui_settings()["echoCalibrations"] == {}
    assert "mic-a::output-b" not in current.read_text(encoding="utf-8")
    assert "mic-a::output-b" not in legacy.read_text(encoding="utf-8")


def test_hf_ui_normalizes_legacy_audio_cpp_provider_settings(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "UI_SETTINGS_PATH", tmp_path / "hf_realtime_ui_settings.json")

    saved = server._write_public_ui_settings(
        {
            "ttsBackend": "audio-cpp",
            "voiceByBackend": {"audio-cpp": "clone:legacy-profile"},
            "ttsProfileByBackend": {"audio-cpp": "balanced"},
        }
    )

    assert saved["ttsBackend"] == "qwen3tts-audiocpp"
    assert saved["voiceByBackend"] == {"qwen3tts-audiocpp": "clone:legacy-profile"}
    assert saved["ttsProfileByBackend"] == {"qwen3tts-audiocpp": "balanced"}
    assert '"audio-cpp"' not in server.UI_SETTINGS_PATH.read_text(encoding="utf-8")


def test_audio_cpp_validation_persists_model_and_profile_without_secrets(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "AUDIO_CPP_VALIDATION_PATH", tmp_path / "audio_cpp_validation.json")

    saved = server._write_audio_cpp_validation("qwen3-tts-0.6b-base-bf16", "abc12345")

    assert saved["model_id"] == "qwen3-tts-0.6b-base-bf16"
    assert server._read_audio_cpp_validation()["profile_id"] == "abc12345"
    assert not server.AUDIO_CPP_VALIDATION_PATH.with_suffix(".tmp").exists()


def test_tts_validation_keeps_each_model_clone_pair(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "TTS_VALIDATION_PATH", tmp_path / "tts_backend_validation.json")
    monkeypatch.setattr(server, "AUDIO_CPP_VALIDATION_PATH", tmp_path / "missing_audio_cpp_validation.json")

    first = server._write_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-1.7b-base-bf16", "clone:22fb07ef3a80"
    )
    second = server._write_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-1.7b-base-bf16", "clone:776a491528c9"
    )

    assert server._find_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-1.7b-base-bf16", "clone:22fb07ef3a80"
    ) == first
    assert server._find_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-1.7b-base-bf16", "clone:776a491528c9"
    ) == second
    assert server._find_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-0.6b-base-bf16", "clone:776a491528c9"
    ) == {}


def test_audio_cpp_validation_persists_verified_delivery_mode(tmp_path, monkeypatch):
    server = _load_ui_server_module()
    monkeypatch.setattr(server, "TTS_VALIDATION_PATH", tmp_path / "tts_backend_validation.json")

    saved = server._write_tts_validation(
        "qwen3tts-audiocpp",
        "qwen3-tts-1.7b-base-bf16",
        "clone:candidate-profile",
        delivery_mode="native-incremental-pcm",
        native_streaming=True,
    )

    assert saved["deliveryMode"] == "native-incremental-pcm"
    assert saved["nativeStreaming"] is True
    assert server._find_tts_validation(
        "qwen3tts-audiocpp", "qwen3-tts-1.7b-base-bf16", "clone:candidate-profile"
    ) == saved


def test_audio_cpp_clone_validation_uses_speech_sized_timeout(monkeypatch):
    server = _load_ui_server_module()
    seen = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok", "backend": {"model_id": "qwen3-tts-1.7b-base-bf16", "runtime": {}}}

    class Client:
        def __init__(self, **kwargs):
            seen.append(kwargs["timeout"].read)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/models"):
                return type("Models", (), {"raise_for_status": lambda self: None, "json": lambda self: {"data": [{"id": "qwen3-tts-1.7b-base-bf16"}]}})()
            return Response()

    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    asyncio.run(server._probe_audio_cpp_candidate(run_speech_probe=False))
    assert seen == [8.0]


def test_settings_show_model_provider_before_stacked_tts_and_voice():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    html = (ui_dir / "index.html").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert html.index('id="model-provider"') < html.index('id="tts-backend"') < html.index('id="voice"')
    assert 'id="model-api-key" type="password"' in html
    assert client_js.index("this.updateLocalPipeline") < client_js.index("this._sendSessionUpdate")
