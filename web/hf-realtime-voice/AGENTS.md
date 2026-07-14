# HF Realtime Voice UI

## Purpose

- Vendored browser client from `smolagents/hf-realtime-voice` for local testing against this repo's `/v1/realtime` backend.

## Local Contracts

- `/api/config` exposes `apiVersion`; the browser warns when its expected API contract differs from the running Python UI server.
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
- Speech stop reserves persistent user chronology. Replace it with a validated final transcript before assistant or tool UI; if final transcription fails, show only a temporary UI notice and never insert fabricated text or a canned spoken retry.
- The live-transcription setting controls only the temporary floating user bubble. Final user text always updates the persistent conversation panel.
- Diagnostics label the always-on final stage `Transcription` and show retained history tokens against the context window detected from llama.cpp.
- Camera preview and Diagnostics should not occupy the same desktop corner; keep the camera self-view clear when diagnostics are open.
- Settings list live Faster and Groxaxo status. Provider changes apply to the next conversation, and an unavailable selected provider blocks start without silent fallback.

## Child DOX Index

- No child AGENTS.md files currently.
