import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AEC3 = ROOT / "web" / "hf-realtime-voice" / "worklets" / "aec3"


def test_aec3_manifest_authenticates_pinned_runtime_asset():
    manifest = json.loads((AEC3 / "aec3.manifest.json").read_text(encoding="utf-8"))
    wasm = (AEC3 / manifest["wasm"]).read_bytes()

    assert manifest["available"] is True
    assert manifest["abiVersion"] == 1
    assert manifest["frameMs"] == 10
    assert manifest["output"] == "pcm_s16le/16000/mono"
    assert manifest["sourceRevision"] == "a024d6ef8351add55be5e8b1d6cc35f555787660"
    assert hashlib.sha256(wasm).hexdigest() == manifest["sha256"]


def test_aec3_loader_and_capture_keep_truthful_fallback_contract():
    loader = (AEC3 / "aec3-loader.js").read_text(encoding="utf-8")
    capture = (AEC3 / "aec3-capture.js").read_text(encoding="utf-8")
    fallback = (
        ROOT / "web" / "hf-realtime-voice" / "worklets" / "mic-capture.js"
    ).read_text(encoding="utf-8")

    assert 'crypto.subtle.digest("SHA-256", wasmBytes)' in loader
    assert "WebAssembly.Module.imports(module)" in loader
    assert 'processorName: "aec3-capture"' in loader
    assert 'registerProcessor("aec3-capture", Aec3CaptureProcessor)' in capture
    assert capture.index("this._session.process(reference, capture") < capture.index(
        "this._appendOutput(output"
    )
    assert 'this._effectiveMode = this._moduleReady ? "adaptive" : "native"' in capture
    assert "Fail closed by omitting the 10 ms frame" in capture
    assert 'requested === "strict" ? "strict" : "native"' in fallback
    assert "ECHO_NLMS_STEP" not in fallback
    assert "_echoPrediction" not in fallback


def test_aec3_build_is_pinned_and_reproducible():
    cargo = (AEC3 / "build" / "Cargo.toml").read_text(encoding="utf-8")
    lock = (AEC3 / "build" / "Cargo.lock").read_text(encoding="utf-8")
    build = (AEC3 / "build" / "build.ps1").read_text(encoding="utf-8")

    revision = "a024d6ef8351add55be5e8b1d6cc35f555787660"
    assert f'rev = "{revision}"' in cargo
    assert revision in lock
    assert "cargo build --locked --target $target --release" in build
    assert "System.Text.UTF8Encoding($false)" in build
