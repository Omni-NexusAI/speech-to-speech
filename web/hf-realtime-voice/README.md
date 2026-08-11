---
title: HF Realtime Voice
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Local direct-audio voice chat over an OpenAI-compatible Realtime WebSocket
hf_oauth: true
---

# HF Realtime Voice

This browser client is the local testing surface for the repository's Realtime
speech pipeline. It captures microphone audio, waits for the server to accept a
complete conversation configuration, sends each VAD-accepted turn directly to
Gemma, and plays the selected Qwen3-TTS provider through one continuous PCM
queue. It also keeps the original hosted Hugging Face session-handshake mode for
Space deployments.

For the complete local stack and model configuration, see
[`../../docs/local-gemma-fasterqwen3tts.md`](../../docs/local-gemma-fasterqwen3tts.md).

## Current local pipeline

```text
microphone
  → browser capture, Native AEC or verified AEC3
  → server VAD admission
  → local or explicitly selected remote Gemma direct audio
  → optional browser tools and tool-result continuation
  → selected Faster, Groxaxo, or audio.cpp Qwen3-TTS provider
  → generation-safe AudioWorklet PCM queue
```

The local direct-audio path deliberately does not require an STT transcript to
answer or call tools. Every accepted user turn gets one semantic history anchor.
A validated transcript may upgrade it; if no transcript exists, a validated
same-response semantic memory may upgrade it; otherwise the session retains the
historical input audio. The UI's `[User audio]` label is display-only.

## Start locally

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\local_realtime.ps1 -Action start -Open
```

The managed launcher starts only this repository's backend and frontend:

- UI: <http://127.0.0.1:7862>
- Realtime WebSocket: `ws://127.0.0.1:8765/v1/realtime`

Gemma and every TTS service remain external dependencies. The launcher reports
their status but never starts, stops, loads, switches, or rebuilds them.

For a foreground frontend alone:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\start_hf_realtime_frontend.ps1
```

Or run this directory as a standalone Docker/FastAPI app:

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

Browsers require HTTPS or a loopback host for microphone and camera access.

## Startup contract

The client opens `/v1/realtime`, receives `session.created`, sends the
conversation-scoped pipeline configuration, and waits for the matching
`pipeline.config.updated` acknowledgement. Microphone PCM is not sent before
that acknowledgement. A rejected configuration or a missing acknowledgement
within 15 seconds closes the session visibly instead of leaving the UI in a
permanent Connecting state.

Standard voice, instructions, and tools use `session.update`. Local/remote model
routing, TTS provider, clone, and candidate tuning use
`pipeline.config.update`. Updates are transactional: queued speech keeps its
acknowledged settings, while a confirmed update applies to the next turn.

## Model routing

Model inference is conversation-scoped:

- **Local** uses the configured llama.cpp/Gemma audio endpoint.
- **Remote** uses one explicitly configured OpenAI-compatible endpoint and
  bearer. It replaces Local for that conversation and never silently falls
  back.

API keys stay only in this browser/device's `localStorage`. They are excluded
from UI-server persistence payloads and are never returned by the UI server.
The About and Diagnostics surfaces report the effective provider, model,
context window, and runtime identity without logging prompt or response text.

## TTS providers and voices

- **FasterQwen3TTS** on `8881` is the stable default and exposes clone-only
  native PCM streaming. It has no model lifecycle API, so load/switch controls
  remain unavailable.
- **Groxaxo** on `8882` is user-managed and selectable only while it advertises
  a compatible loaded Base model.
- **Qwen3-TTS audio.cpp** on `8890` is an isolated experimental candidate. It
  requires live health, resident-model, clone, profile-resolution, and speech
  validation. Native incremental PCM is used only when the response proves it;
  otherwise the UI labels and keeps buffered fallback.

Voice inventory is backend-scoped and authoritative for the selected provider.
Only live Base clone profiles are selectable. Faster's shared library is read
from `VOICE_LIBRARY_DIR`; if unset, both the pipeline and UI use
`~/.speech-to-speech/qwen3-tts-voices`. The expected layout is
`profiles/<profile-id>/meta.json` plus that profile's reference WAV.

Realtime and Voice Studio profile selections remain independent. Candidate
profile updates may be applied during a conversation after acknowledgement,
but they do not change Voice Studio's selection.

## Playback and latency

All providers deliver PCM16 to the browser's 16 kHz playback graph. Faster,
Groxaxo, and audio.cpp buffered fallback start on their existing ready queue.
Validated native audio.cpp uses adaptive safe-start:

- a cold or changed signature primes the acknowledged first plus steady decoder
  blocks;
- two clean full-prime responses establish warm evidence;
- warm startup uses the logical decoder-block p95 gap plus a fixed 64 ms margin,
  never less than the first block plus 160 ms;
- an underrun immediately restores the full cold ceiling and requires three
  clean recoveries.

The worklet preserves every input sample, flushes short final responses, keeps
one queue across phrase and tool continuations, and rejects stale generations
after Stop, barge-in, replacement, disconnect, or cancellation. Network
`response.done` releases protocol ownership; only worklet `started` and
`drained` events control the audible speaking state.

## Echo control

- **Native** is the safe default and uses the browser's built-in echo
  cancellation, noise suppression, and automatic gain control.
- **Adaptive** uses the bundled Sonora/WebRTC AEC3 worklet only after manifest,
  SHA-256, ABI, compile, and worklet loading succeed. Failure resolves
  truthfully to Native.
- **Strict** uses the reference-aware path more aggressively and withholds
  uncertain microphone upload through the echo tail rather than inserting
  silence.

The capture path receives the exact PCM scheduled for playback, never the static
voice-clone reference. Calibration is stored per microphone/output-device pair.
AEC3 and physical speaker-loopback behavior remain experimental across device
pairs and remote-audio routing.

## Tools

The browser executes enabled tools and returns each result through the normal
Realtime function-call protocol:

- **Web search** uses the same-origin `/api/search` proxy when `SERPER_API_KEY` is
  configured. The key never reaches browser JavaScript.
- **Camera snapshot** captures a fresh frame only when the model calls the tool.
  Every real call gets a distinct visible card, including unavailable captures.

Malformed or schema-invalid arguments are rejected before execution without
displaying their values. Function output acknowledgement precedes exactly one
post-tool `response.create`, and tool-result continuation stays in the same
semantic turn and playback policy.

## Hosted Space mode

When `LOAD_BALANCER_URL` is configured, `/api/session` can proxy the hosted
Hugging Face load-balancer handshake without exposing the upstream URL. The
returned signed `connect_url` is then used for the Realtime WebSocket. Usage
metering is enabled only when both `LOAD_BALANCER_URL` and `SPACE_ID` are set;
local development remains unmetered.

Hosted authentication and limits live in `auth.py` and `limiter.py`. Search
proxying and hosted session routing live in `server.py`; neither changes the
local direct WebSocket contract.

## Main files

| File | Role |
|---|---|
| `server.py` | Static app, settings, provider inventory/validation, hosted session and search proxies |
| `main.js` | UI state, configuration acknowledgement, tools, diagnostics, and provider selection |
| `ws/s2s-ws-client.js` | Realtime protocol, tool ordering, cancellation, and adaptive playback policy |
| `worklets/mic-capture.js` | Capture, resampling, gate, Native/Strict fallback behavior |
| `worklets/aec3/` | Pinned AEC3 WASM, authenticated loader, source recipe, smoke test, and license |
| `worklets/audio-playback.js` | Continuous generation-tagged PCM FIFO and safe-start state machine |
| `ui/chat.js` | Durable conversation and per-call tool/camera cards |

## Limits

- Native audio.cpp streaming and adaptive playback learning are experimental.
- AEC3 still needs device-pair-specific speaker-loopback and double-talk
  validation.
- Provider `Auto` language support is reported from the explicit provider
  capability policy, not inferred from the general speech-validation request.
  Only audio.cpp is currently allowlisted; Faster and Groxaxo remain
  unsupported or unverified. Clone-reference language never substitutes for
  conversational response language.
- The managed launcher does not own Gemma, FasterQwen3TTS, Groxaxo, or
  audio.cpp model lifecycle.

## Credits

- Backend: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
- Original browser experience: `amir-tfrere/minimal-conversation-app-s2s-backend`
- Local direct-audio and provider integration: [Omni-NexusAI](https://github.com/Omni-NexusAI)
