---
title: HF Realtime Voice
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Voice chat over WebSocket against a HF speech-to-speech
hf_oauth: true
---

# Minimal Conversation App (S2S backend, **WebSocket** transport)

Drop-in alternative to [`amir-tfrere/minimal-conversation-app-s2s-backend`](https://huggingface.co/spaces/amir-tfrere/minimal-conversation-app-s2s-backend)
that uses the **WebSocket** route of the Hugging Face speech-to-speech
backend instead of the WebRTC SDP proxy. Same load balancer, same
`/session` handshake, same UI, same orb. Just a different wire.

## How it works

1. App POSTs `<lb_url>/session` (empty JSON body).
2. The LB picks a ready compute (round-robin) and returns:
   ```json
   {
     "session_id": "...",
     "websocket_url": "wss://<compute>/v1/realtime",
     "connect_url": "wss://<compute>/v1/realtime?session_token=<JWT>",
     "session_token": "<JWT>",
     "pending_timeout_s": 60
   }
   ```
3. App opens a WebSocket **directly** on `connect_url` (no rewrite to
   `https://`; unlike the WebRTC client which POSTs an SDP offer).
4. Server pushes `session.created` on connect. Client replies with
   `session.update` (OpenAI Realtime **GA** schema: `session.audio.input`,
   `session.audio.output`, `session.output_modalities`).
5. Client streams mic audio as PCM16 16 kHz mono base64 chunks
   (`input_audio_buffer.append`, one frame every ~40 ms).
6. Server pushes `response.output_audio.delta` (pipeline-native PCM16 16 kHz
   mono base64) after any provider-native audio is adapted by the Realtime
   pipeline.
   and transcript deltas.

The backend exposes one concurrent session per compute (same as WebRTC
mode); the LB pins the session via a signed `session_token`.

## Why WebSocket instead of WebRTC

| | WebRTC (original) | WebSocket (this) |
|---|---|---|
| Transport | UDP + Opus 48 kHz + ICE/STUN | TCP + raw PCM16 |
| NAT traversal | needs STUN, can fail on corporate / cellular | none, works everywhere TCP is allowed |
| Audio quality | excellent (Opus, jitter buffer, FEC) | good (raw PCM, simple ring buffer) |
| Latency | lowest (~50-150 ms) | low (~150-300 ms typical) |
| Echo cancellation | browser AEC active on the WebRTC track | browser AEC plus playback-reference Adaptive/Strict guard |
| Debuggability | needs `chrome://webrtc-internals` | `wscat` / DevTools network tab |
| Mobile data | sometimes blocked (UDP) | always works (HTTPS+WSS) |

## Backend requirement

This app talks to the WebSocket route `@app.websocket("/v1/realtime")`
defined in
[`websocket_router.py`](https://github.com/huggingface/speech-to-speech/blob/feat/webrtc-transport/src/speech_to_speech/api/openai_realtime/websocket_router.py)
on the **`feat/webrtc-transport`** branch. The same compute serves both
the WebRTC POST and the WebSocket upgrade on the same path; no backend
change required.

Smoke-test from the shell:

```bash
LB="https://kaa1l6rplzb1gg3y.us-east-1.aws.endpoints.huggingface.cloud"
curl -X POST "$LB/session" -H "Content-Type: application/json" -d '{}'
# -> { "connect_url": "wss://<compute>/v1/realtime?session_token=..." }
# Feed connect_url into a wscat / websocat and you should get a
# session.created event back immediately.
```

## Tools

The assistant can call two tools mid-conversation (toggle them from the **Tools**
button, top-right):

- **Web search** — Google results via Serper.dev, proxied server-side so the key
  never reaches the browser. Set `SERPER_API_KEY` as a Space secret. Without it,
  the tool is disabled unless the user pastes their own key in the Tools panel.
- **Camera** — while enabled, a live self-view shows bottom-left; when the model
  calls the tool, the current frame is sent to the vision-language model so it can
  see what you're showing it.

## Connecting to a backend

The app connects **directly** to a speech-to-speech server's realtime WebSocket —
no load balancer, no `/session` step. Set the URL in two ways:

- **`LOAD_BALANCER_URL` env** (served via `/api/config`) provides the default URL
  shown in Settings — handy for the deployed Space.
- **Settings → Speech-to-speech server URL** lets you override it: paste a full
  `connect_url` (`wss://host/v1/realtime?...`) or a bare host like `localhost:8080`
  (the app adds `/v1/realtime`).

**Settings → Restart** reconnects with the current voice, instructions and URL.

## Usage limits

Conversation time is metered per UTC day by sign-in tier (see `limiter.py` /
`auth.py`), but **only on the deployed Space** — metering turns on only when BOTH
`LOAD_BALANCER_URL` and `SPACE_ID` (injected automatically by the HF Space
runtime) are present. Running locally — even with `LOAD_BALANCER_URL` exported —
leaves the app unmetered. Tunable via env:

| Env | Default | What |
|-----|---------|------|
| `LIMIT_ANON_SEC` | `300` | Daily seconds for anonymous visitors (5 min) |
| `LIMIT_FREE_SEC` | `600` | Daily seconds for signed-in non-PRO users (10 min) |
| `UNLIMITED_ORGS` | _(adds to defaults)_ | Extra HF org names whose members get **unlimited** usage, like PRO |
| `USAGE_HASH_SECRET` | _(random)_ | HMAC secret for hashing identity keys + signing the anon cookie |

PRO members are always unlimited. Members of `cerebras`, `HuggingFaceM4`,
`smolagents`, and `pollen-robotics` are unlimited out of the box (shown as
"Team", not "PRO"); set `UNLIMITED_ORGS=my-team` to add more. Matched
case-insensitively against the user's organisations from HF OAuth.

## Run locally

The app is now a small FastAPI server (it serves the front-end *and* the search
proxy from one container).

```bash
pip install -r requirements.txt
export SERPER_API_KEY=...        # optional; web search is disabled without it
export LOAD_BALANCER_URL=...     # optional; default s2s server URL (set it in Settings otherwise)
uvicorn server:app --reload --port 7860
# or, matching production: docker build -t s2s . && docker run -p 7860:7860 -e SERPER_API_KEY=... -e LOAD_BALANCER_URL=... s2s
```

Then open <http://localhost:7860/>, click the orb, allow the mic, talk.

> Browsers require **HTTPS or `localhost`** for `getUserMedia()` (mic + camera).
> `127.0.0.1` and `localhost` both work; plain `http://192.168.x.y` does NOT.

## Settings (stored in `localStorage`)

| Key | What |
|-----|------|
| Load balancer URL | Base URL of your S2S deployment. App POSTs `<lb>/session`. |
| Voice | Qwen3-TTS speaker name (Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu, Vivian) |
| Instructions | System prompt sent in `session.update` once the WS opens |

LocalStorage keys are namespaced `s2s.ws.*` so this app's settings do
NOT collide with the WebRTC variant.

## Files

| File | Role |
|------|------|
| `index.html` | Single page, orb + settings modal (identical UI to the WebRTC app) |
| `main.js` | State machine, settings, tools, camera, noise-gate UI wiring |
| `ui/chat.js` | `ChatView`: history panel, ephemeral bubbles, transcript/tool streaming |
| `ui/account.js` | `Account`: HF login chip + popover, daily-limit modal |
| `ui/dom.js` | Shared helpers: `$`, `escHtml`, `truncateError`, `DEBUG` |
| `auth.py` | HF OAuth + per-request identity (tier, hashed keys) |
| `limiter.py` | SQLite per-day talk-time budget (chunked server-clock reservation) |
| `ws/s2s-ws-client.js` | WebSocket handshake + OpenAI Realtime GA protocol |
| `ws/codec.js` | base64 <-> PCM helpers + transcript extraction (pure) |
| `ws/orb-visualizer.js` | `OrbVisualiser`: FFT bands -> orb CSS custom properties |
| `worklets/mic-capture.js` | Native-AEC/Strict fallback worklet: resamples capture to 16 kHz PCM16 without a custom predictor |
| `worklets/aec3/` | SHA-verified pinned Sonora/WebRTC AEC3 WASM, loader, reference-aware capture worklet, build recipe, and license |
| `worklets/audio-playback.js` | AudioWorklet: continuous generation-tagged Float32 FIFO, profile priming, resampling, end flush, and stale-tail rejection |
| `style.css` | Orb animations, layout, dark theme (verbatim from the WebRTC app) |

## Audio pipeline notes

- **Input**: `getUserMedia({ echoCancellation, noiseSuppression, autoGainControl })`
  feeds the `mic-capture` worklet at the `AudioContext` rate. The worklet
  resamples to 16 kHz (boxcar lowpass + decimation on the 48 -> 16 fast
  path, linear interpolation fallback for odd rates) and packs Int16 LE.
- **Echo guard**: Native browser AEC is the safe default. Adaptive loads the
  pinned Sonora/WebRTC AEC3 WASM only after its manifest ABI and SHA-256 pass,
  then processes aligned 10 ms render-before-capture frames using the exact PCM
  sent to the playback graph. Any module failure resolves truthfully to Native.
  Strict applies stronger reference-aware suppression and fails closed by
  omitting uncertain frames through the playback tail; it never inserts zero
  PCM. Delay, suppression, leakage, and double-talk sensitivity are saved per
  microphone/output-device pair. The static voice-clone recording is never used
  as the echo reference, and Native remains the default until physical
  speaker-loopback and real human barge-in tests pass.
- **Output**: `response.output_audio.delta` decodes to Int16 -> Float32
  and is posted to one `audio-playback` FIFO across phrase chunks and same-turn
  tool continuations. Validated native audio.cpp playback primes Low Latency,
  Balanced, and Quality at 800, 1280, and 1760 ms; a custom resolved target is
  capped at 2000 ms. Faster, Groxaxo, and buffered fallback remain immediate.
  The visible speaking state begins only when the worklet renders a first sample
  and ends only when that queue drains; network PCM receipt and `response.done`
  keep their separate transport and response-lock semantics.
  Priming never consumes the head, a real underrun re-primes to the full target,
  and `response.output_audio.done` flushes a short final stream immediately.
- **Barge-in**: when the server VAD detects user speech mid-response
  (`input_audio_buffer.speech_started` while `ai-speaking`), the client
  advances the playback generation and clears the queue once. Late PCM retains
  its old generation and is rejected by the worklet. Stop, replacement,
  disconnect, and response cancellation use the same stale-tail boundary.

## Credits

- Backend: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) on `feat/webrtc-transport`
- UI verbatim from `amir-tfrere/minimal-conversation-app-s2s-backend` (Pollen Robotics × Hugging Face)
