import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "latency_gpu_settle", Path(__file__).parents[1] / "scripts/measure_candidate_tts_paths.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
settled_gpu_snapshot = module.settled_gpu_snapshot


def sample(util=98, free=6000, temperature=70):
    return {"utilization_percent": util, "free_mib": free, "temperature_c": temperature}


def test_prior_request_utilization_can_age_out_without_relaxing_threshold():
    readings = iter([sample(), sample(40)])
    waits = []
    assert settled_gpu_snapshot(83, snapshot=lambda: next(readings), sleep=waits.append) == sample(40)
    assert waits == [1]


def test_persistent_utilization_stops_after_five_seconds():
    waits = []
    assert settled_gpu_snapshot(83, snapshot=sample, sleep=waits.append) == sample()
    assert waits == [1] * 5


def test_no_wait_or_retry_for_memory_temperature_or_missing_telemetry():
    for reading in [sample(free=1024), sample(temperature=90), {"unavailable": True}]:
        waits = []
        assert settled_gpu_snapshot(83, snapshot=lambda: reading, sleep=waits.append) == reading
        assert waits == []


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload


def _main_args():
    return SimpleNamespace(
        candidate="http://candidate",
        studio="http://studio",
        voice="clone:fixed",
        profile="balanced",
        text="fixed diagnostic",
        seed=321,
        execute=True,
        audio_dir=None,
        max_gpu_temperature=83,
        inter_request_cooldown_seconds=0,
    )


def _install_main_client(monkeypatch):
    health = {
        "activeModel": "qwen3-tts-1.7b-base-bf16", "engineEpoch": 1,
        "supervisorInstanceId": "candidate-1", "state": "loaded", "singleResident": True,
    }
    profile = {"revision": 1, "first_block_frames": 4}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url):
            if url.endswith("/health"):
                return _Response(health)
            if url.endswith("/tuning/profiles"):
                return _Response({"profiles": {"balanced": profile}})
            if "/voices/profiles/" in url:
                return _Response({"content_hash": "c" * 64, "content_revision": 1})
            raise AssertionError(f"unexpected read {url}")

    monkeypatch.setattr(module.httpx, "Client", Client)
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda _self: _main_args())


def test_first_measurement_dispatch_uses_strict_snapshot_without_settle(monkeypatch):
    _install_main_client(monkeypatch)
    settled = []
    measured = []
    monkeypatch.setattr(module, "gpu_snapshot", lambda: sample(98))
    monkeypatch.setattr(module, "settled_gpu_snapshot", lambda *_args: settled.append(True))
    monkeypatch.setattr(module, "measure", lambda *_args, **_kwargs: measured.append(True))

    module.main()

    assert settled == []
    assert measured == []


def test_report_discloses_cooldown_without_measuring(monkeypatch, capsys):
    args = _main_args()
    args.execute = False
    args.inter_request_cooldown_seconds = 3.5
    _install_main_client(monkeypatch)
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda _self: args)
    monkeypatch.setattr(module, "gpu_snapshot", lambda: sample(40))

    module.main()

    assert json.loads(capsys.readouterr().out)["inter_request_cooldown_seconds"] == 3.5


def test_settle_is_used_only_after_this_diagnostic_completes_a_measure(monkeypatch):
    _install_main_client(monkeypatch)
    calls = []

    class SettledAfterOwnRequest(Exception):
        pass

    monkeypatch.setattr(module, "gpu_snapshot", lambda: sample(40))
    monkeypatch.setattr(module, "measure", lambda *_args, **_kwargs: calls.append("measure") or {"completion_proven": True})

    def stop_on_settle(*_args):
        calls.append("settle")
        raise SettledAfterOwnRequest

    monkeypatch.setattr(module, "settled_gpu_snapshot", stop_on_settle)
    with pytest.raises(SettledAfterOwnRequest):
        module.main()
    assert calls == ["measure", "settle"]


def test_cooldown_follows_own_completed_measure_and_precedes_fresh_checks(monkeypatch):
    args = _main_args()
    args.inter_request_cooldown_seconds = 3.5
    _install_main_client(monkeypatch)
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda _self: args)
    calls = []

    class SettledAfterCooldown(Exception):
        pass

    monkeypatch.setattr(module, "gpu_snapshot", lambda: sample(40))

    def fake_measure(*_args, **_kwargs):
        calls.extend(["measure_start", "measure_end"])
        return {"completion_proven": True}

    monkeypatch.setattr(module, "measure", fake_measure)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: calls.append(("cooldown", seconds)))

    def stop_after_cooldown(*_args):
        calls.append("settle")
        raise SettledAfterCooldown

    monkeypatch.setattr(module, "settled_gpu_snapshot", stop_after_cooldown)
    with pytest.raises(SettledAfterCooldown):
        module.main()
    assert calls == ["measure_start", "measure_end", ("cooldown", 3.5), "settle"]


@pytest.mark.parametrize("value", ["-0.1", "60.1", "nan", "inf", "not-a-number"])
def test_inter_request_cooldown_rejects_nonfinite_or_out_of_bounds_values(value):
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.inter_request_cooldown_seconds(value)
