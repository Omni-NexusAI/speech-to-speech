# Web UI

## Purpose

- Contains the HF Realtime Voice browser testing UI integrated into this repo.

## Local Contracts

- Keep the UI as a Realtime client for the speech-to-speech backend rather than a second backend implementation.
- Default local testing should point at the local `/v1/realtime` server.
- Voice settings should list saved Voice Studio `Base` clone profiles only, using `clone:<profile_id>` values.
- Leave external web-search functionality disabled unless explicitly configured by the user.
- Preserve AudioWorklet PCM16 capture/playback behavior. Capture averages every
  available microphone channel before one shared, stateful anti-aliased 16 kHz
  conversion; PCM16 output saturates rather than wrapping.
- The About/status surfaces should reflect the live local stack: Gemma endpoint/model, Qwen3TTS endpoint/backend/model, selected clone, VAD mode, and tool status.
- The diagnostics UI should consume realtime `pipeline.metric` events and expose queue/latency signals without becoming a second pipeline implementation.

## Child DOX Index

- `web/hf-realtime-voice/AGENTS.md` covers the vendored Space files once imported.
