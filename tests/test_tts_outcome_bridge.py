"""Late candidate failures are attributed without another synthesis request."""
import pytest

from speech_to_speech.TTS import qwen3_tts_handler as module


@pytest.mark.parametrize("status,reason,exception", [
    ("completed", "eos", None),
    ("limited", "max_generated_frames", module.TTSRunawayError),
    ("error", "wall-timeout", RuntimeError),
    ("cancelled", "client-cancelled", RuntimeError),
    ("producing", None, RuntimeError),
])
def test_outcome_checks_exact_frozen_provider_request(monkeypatch, status, reason, exception):
    handler = object.__new__(module.Qwen3TTSHandler)
    handler._openai_api_headers = lambda: {}
    requests = []
    key = "a" * 32

    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"requestId": key, "state": status, "reason": reason}

    def get(url, **kwargs):
        requests.append((url, kwargs))
        return Response()
    monkeypatch.setattr(module.httpx, "get", get)
    if exception:
        with pytest.raises(exception):
            handler._check_candidate_outcome("http://candidate/v1", {"x-tts-request-id": key})
    else:
        handler._check_candidate_outcome("http://candidate/v1", {"x-tts-request-id": key})
    assert requests == [(f"http://candidate/v1/audio/outcomes/{key}", {"headers": {}, "timeout": 1.0})]


def test_outcome_unknown_or_foreign_id_cannot_be_success(monkeypatch):
    handler = object.__new__(module.Qwen3TTSHandler)
    handler._openai_api_headers = lambda: {}
    with pytest.raises(RuntimeError, match="invalid"):
        handler._check_candidate_outcome("http://candidate/v1", {"x-tts-request-id": "../../other"})
    def unavailable(*args, **kwargs):
        raise module.httpx.ConnectError("offline")
    monkeypatch.setattr(module.httpx, "get", unavailable)
    with pytest.raises(RuntimeError, match="could not be verified"):
        handler._check_candidate_outcome("http://candidate/v1", {"x-tts-request-id": "a" * 32})
    # Lookup failure on an already failed transport preserves the real failure.
    handler._check_candidate_outcome("http://candidate/v1", {"x-tts-request-id": "a" * 32}, require_complete=False)


def test_candidate_missing_completion_identity_fails_without_foreign_probe(monkeypatch):
    handler = object.__new__(module.Qwen3TTSHandler)
    monkeypatch.setattr(module.httpx, "get", lambda *a, **k: pytest.fail("foreign probe"))
    with pytest.raises(RuntimeError, match="completion identity"):
        handler._check_candidate_outcome("http://candidate/v1", {})
    handler._check_candidate_outcome("http://candidate/v1", {}, require_complete=False)
