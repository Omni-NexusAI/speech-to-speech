# AEC3 AudioWorklet

This directory contains the browser-side WebRTC Audio Processing AEC3 path.
It uses the pure-Rust Sonora port of WebRTC M145, pinned to commit
`a024d6ef8351add55be5e8b1d6cc35f555787660`, compiled as an import-free
`wasm32-unknown-unknown` module. Sonora is BSD-3-Clause and reports parity
against the upstream WebRTC M145 audio-processing tests.

The worklet receives microphone capture on input 0 and the exact rendered
playback signal on input 1. It buffers both aligned inputs into 10 ms frames,
submits render before capture, supplies the calibrated output/device delay,
and converts the processed capture to 16 kHz PCM16 only after AEC3. Adaptive
never uploads a JS predictor residual. Strict fails closed by omitting uncertain
frames; it never sends replacement zero frames.

`aec3-loader.js` verifies the manifest ABI and SHA-256, rejects modules with
imports, compiles the module on the main thread, and registers
`aec3-capture.js`. If any gate fails, Adaptive must remain Native browser AEC.

Build with PowerShell from this folder:

```powershell
.\build\build.ps1
```

The build requires Rust 1.91 or newer and an explicitly installed
`wasm32-unknown-unknown` target. It writes `aec3.wasm` and atomically refreshes
`aec3.manifest.json`, then runs a deterministic import/ABI/timing/echo-reduction
smoke test. Build output under `build/target/` is disposable and is not a
runtime asset.

Limit: Sonora preserves near-end speech through AEC3's internal near-end
logic, but does not publicly expose that private detector boolean. The bridge's
`doubleTalk` field is therefore labeled `aec3-output-evidence`: it is
diagnostic evidence derived from the processed AEC3 output and residual-echo
statistics, and never controls Adaptive admission. Physical speaker-loopback,
multiple device pairs, reconnects, and browser AudioWorklet deadline tests are
still required before making Adaptive the default.

Primary references:

- https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing.h
- https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/echo_canceller3.h
- https://github.com/dignifiedquire/sonora
