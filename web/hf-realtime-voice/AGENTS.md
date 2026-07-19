# HF Realtime Voice UI

## Purpose

- Vendored browser client from `smolagents/hf-realtime-voice` for local testing against this repo's `/v1/realtime` backend.

## Local Contracts

- `/api/config` exposes `apiVersion`; the browser warns when its expected API contract differs from the running Python UI server.
- Model provider settings are persisted locally, validated before mic activation, and applied to the next conversation. Never render or return a stored bearer key; only expose `api_key_set`.
- Diagnostics consume backend `pipeline.metric` and `pipeline.runtime` events plus browser mic/playback telemetry; missing runtime identity or turn metrics is a stale-backend warning.
- Camera remains tool-triggered: only a `camera_snapshot` call captures and attaches one frame.

- Keep this as a frontend client only; backend speech pipeline changes belong under `src/`.
- Default local testing should connect to `ws://127.0.0.1:8765/v1/realtime` through the UI settings.
- Do not enable external search or cloud services by default.
- Settings, About, and Diagnostics should show the real local runtime state, not hosted-demo model labels.
- Keep the upstream `Built by` credit intact. The top identity row must source local Gemma and FasterQwen3TTS provider status from `/api/local-pipeline`, and place the Omni-NexusAI fork credit in a separate `Modified by` row.
- When the backend emits `pipeline.metric`, keep the UI rendering lightweight and diagnostic-only; do not infer pipeline state by duplicating backend logic in the browser.
- Local runtime toggles and TTS backend selection use `pipeline.config.update`, leaving OpenAI-compatible `session.update` for standard voice/instructions/tool fields.
- Send tool output, optional camera image, and one follow-up `response.create` immediately in hosted order. The backend owns the call-ID barrier; never replay a rejected create on a later user turn.
- Speech stop reserves persistent user chronology. Replace it with validated transcript metadata when available or persistent `[User audio]` when absent; the placeholder is display-only and transcript availability never controls assistant or tool UI.
- The live-transcription setting controls only the temporary floating user bubble. Final user text always updates the persistent conversation panel.
- Diagnostics label the always-on final stage `Transcription` and show retained history tokens against the context window detected from llama.cpp.
- Camera preview and Diagnostics should not occupy the same desktop corner; keep the camera self-view clear when diagnostics are open.
- Settings list live Faster and Groxaxo status. Provider changes apply to the next conversation, and an unavailable selected provider blocks start without silent fallback.
- Keep model inference controls separate from TTS controls, and place Voice directly beneath TTS Backend in the vertically scrolling settings layout.
- Stop invalidates the active client before asynchronous teardown; closed-client mic, playback, tool, and WebSocket events must never change the idle UI or enter a replacement conversation.
- Feed the exact playback worklet PCM into the capture worklet as a non-audible reference. Adaptive suppresses correlated echo but admits sustained double-talk; Strict suspends upload through the echo tail; Off preserves capture.
- Keep native `echoCancellation`, `noiseSuppression`, and `autoGainControl` enabled and expose echo correlation, suppression, double-talk, and mode in diagnostics.
- Adaptive echo guard may briefly confirm uncorrelated double-talk, but it must buffer and replay the accepted onset in order. The response-length setting is stored locally and sent as `max_response_tokens` with the next/local pipeline update; it does not change transcript or history behavior.

## Child DOX Index

- No child AGENTS.md files currently.
