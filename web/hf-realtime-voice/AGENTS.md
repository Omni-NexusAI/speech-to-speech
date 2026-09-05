# HF Realtime Voice UI

## Purpose

- Vendored browser client from `smolagents/hf-realtime-voice` for local testing against this repo's `/v1/realtime` backend.

## Local Contracts

- `/api/config` exposes `apiVersion`; the browser warns when its expected API contract differs from the running Python UI server.
- Model provider settings are persisted locally, validated before mic activation, and applied to the next conversation. Never render or return a stored bearer key; only expose `api_key_set`.
- Diagnostics consume backend `pipeline.metric` and `pipeline.runtime` events plus browser mic/playback telemetry; missing runtime identity or turn metrics is a stale-backend warning.
- Camera remains tool-triggered: enabling `camera_snapshot` advertises it independently of preview-stream readiness, and only a tool call captures and attaches one frame.

- Keep this as a frontend client only; backend speech pipeline changes belong under `src/`.
- Default local testing should connect to `ws://127.0.0.1:8765/v1/realtime` through the UI settings.
- Do not enable external search or cloud services by default.
- Settings, About, and Diagnostics should show the real local runtime state, not hosted-demo model labels.
- Keep the upstream `Built by` credit intact. The top identity row must source the persisted selected Gemma provider, endpoint, and model plus the currently selected TTS provider from `/api/local-pipeline`; it must not fall back to the launcher's local Gemma identity when Remote is selected. Place the Omni-NexusAI fork credit in a separate `Modified by` row.
- When the backend emits `pipeline.metric`, keep the UI rendering lightweight and diagnostic-only; do not infer pipeline state by duplicating backend logic in the browser.
- Local runtime toggles and TTS backend selection use `pipeline.config.update`, leaving OpenAI-compatible `session.update` for standard voice/instructions/tool fields.
- Initial capture remains gated until `pipeline.config.updated` acknowledges the
  complete conversation-scoped configuration. A missing acknowledgement times
  out after 15 seconds; any server error before it is fatal, visible, and closes
  the socket so the backend releases the pipeline slot.
- Send tool output, optional camera image, and one follow-up `response.create` immediately in hosted order. The backend owns the response-create call-ID barrier; the browser keeps bounded per-connection accepted call-ID tombstones so a replayed function-call terminal cannot execute an external tool twice, while its one output acknowledgement remains idempotent. Never replay a rejected create on a later user turn.
- Speech stop reserves persistent user chronology. Replace it with validated transcript metadata when available or persistent `[User audio]` when absent; the placeholder is display-only and transcript availability never controls assistant or tool UI.
- The live-transcription setting controls only the temporary floating user bubble. Final user text always updates the persistent conversation panel.
- Diagnostics label the always-on final stage `Transcription` and show retained history tokens against the context window detected from llama.cpp.
- Camera preview and Diagnostics should not occupy the same desktop corner; keep the camera self-view clear when diagnostics are open.
- Settings list every configured TTS provider even while unavailable. Provider
  changes apply to the next conversation, and an unavailable selected provider
  blocks start without silent fallback.
- Voice inventory and profile controls are scoped to the selected backend.
  Reconcile Faster's live voices with its configured writable library, use the
  candidate-private audio.cpp profile API, and keep Groxaxo inventory-only
  unless its existing API explicitly advertises safe mutation support.
- Treat each selected backend's current no-store inventory as authoritative;
  do not hide a live audio.cpp profile merely because another provider uses the
  same ID. Clear and disable the voice selector before awaiting a backend
  switch, ignore stale inventory responses, and expose no clone IDs while the
  active backend is unavailable. A new audio.cpp clone visibly requires its
  own explicit validation before a conversation can start.
- Persist one selected Base clone per backend. Restore it on backend switches;
  if missing, choose the provider default or first live Base clone and make the
  fallback visible.
- Restore server-managed settings before the first backend inventory request.
  On backend or clone changes, await the server-backed selection save before
  refreshing identity-scoped validation; do not show a previous provider or
  previous clone's readiness while the new selection is settling.
- Opening Settings refreshes both selected-backend inventory and live backend
  readiness; a transient startup probe must not leave a healthy provider
  labeled unavailable for the rest of the page session.
- Faster model inventory is truthful: its fixed clone-only API currently has no load/switch/unload endpoints, so controls must stay unavailable instead of simulating a model lifecycle.
- The audio.cpp candidate stays separate and is selectable only after an
  explicit probe proves health, a resident compatible model, speech, and a
  synchronized private clone profile. Native incremental PCM requires a live
  capability advertisement, the exact
  `X-TTS-Streaming-Mode: native-incremental-pcm` response header, and at least
  two nonempty PCM chunks. A missing or mismatched header, one completed body,
  or a failed native request is buffered-only evidence; persist that truthful
  delivery mode per model/clone. Never start, switch, or replace
  Faster/Groxaxo automatically.
- Persist local UI preferences (including the selected TTS backend) atomically
  through `/api/ui-settings` so an environment/browser reset can restore them.
  Never retain model or service API keys there; those remain browser-session
  settings.
- General provider validation checks health, resident model, capabilities,
  selected clone presence, and a short synthesis without changing model
  residency. Persist and retrieve results by backend/model/clone so switching
  between previously validated candidate clones does not erase either record;
  experimental audio.cpp still requires explicit validation.
- Persist non-secret backend endpoints, prompts, selected backend, and
  per-backend clone selections atomically through server-managed runtime state.
  Await saves in the Settings UI and keep all API keys browser-session-only.
- History compaction is a conversation-scoped `pipeline.config.update` setting
  with only `enabled`, `trigger_ratio`, `target_ratio`, and `recent_turns`.
  Persist the bounded local preference and render only backend-reported history
  token telemetry; render acknowledged backend policy/status and bounded budget
  fields only, never estimate a compaction budget, success, or failure.
  On startup, remove the legacy local model-key slot without reading it; the
  managed settings reader likewise filters and rewrites legacy private fields
  without returning or logging their values.
- Keep model inference controls separate from TTS controls, and place Voice directly beneath TTS Backend in the vertically scrolling settings layout.
- Stop invalidates the active client before asynchronous teardown; closed-client mic, playback, tool, and WebSocket events must never change the idle UI or enter a replacement conversation.
- Local response ownership uses monotonic `input_epoch` and `response_epoch`.
  Accept an epoch-only pending `pipeline.response` for diagnostics, then bind
  playback only once its OpenAI response ID exists. Reject stale epoch text,
  PCM, metrics, and terminal events. Browser clients declare rendered-playback
  acknowledgement capability at startup; the worklet's first rendered sample
  sends exactly one local `pipeline.playback.started` acknowledgement per
  response/epoch, while nonbrowser peers settle only after the first non-empty
  PCM frame is successfully sent.
- An epoch-only terminal `pipeline.response` without an OpenAI response ID is
  a one-shot browser terminal: clear the pending response/create state and roll
  back its paired provisional user row. A tool-call response ID commits its
  originating user/tool chronology even when it emits no PCM, so its later
  audible follow-up retains visible origin. Treat response terminal IDs as
  idempotent too, so duplicate completion cannot settle a later user row.
- Retired response IDs are terminal even when a compatible peer later omits
  epoch metadata on transcript frames. Drop only that stale-ID transcript;
  preserve unrelated legacy epoch-less transcript IDs.
- A local epoch-owned response stays provisional until rendered playback is
  acknowledged. Roll back an unheard terminal response and its paired user
  transaction; retain an audible cancelled response once, marked interrupted.
  Epoch-less standard OpenAI-compatible text completions retain their normal
  transcript history.
- A completed response that never reaches audible playback is retired at the
  same rollback boundary. Its bounded response-ID tombstone rejects delayed
  epoch-less compatible transcript frames, while unrelated legacy IDs remain
  accepted.
- The local `pipeline.input_audio.speech_started` decision is available only
  after the browser advertises rendered-playback acknowledgement support. It
  precedes the stock speech-start event and tells the browser whether to clear
  playback; nonbrowser and OpenAI SDK clients receive only the stock event.
- audio.cpp playback continuity defaults to Adaptive. It may alter only the
  bounded (two-second maximum) browser startup reservoir for both native PCM
  and the buffered-phrase fallback while preserving the existing PCM sample
  clock, pitch, duration, and quality. Fast start retains the fixed profile
  target. When the cap or backend RTF proves realtime cannot be sustained,
  expose that condition and retain the explicit full-buffer fallback.
- A playback configuration acknowledgement changes only source-clock and
  continuity-reservoir defaults for later responses. Every response freezes its
  16/24 kHz source rate, native/Adaptive-or-Fast-start mode, effective prime
  target, and two-second ceiling at pending ownership; queued PCM/end messages
  cannot be retimed or re-primed by a later acknowledgement.
- The local `pipeline.response` lifecycle event captures the complete
  response-owned playback policy by epoch before an OpenAI response ID exists,
  then binds it to the first response snapshot before PCM; a later config
  acknowledgement may only set the next response's policy.
- `pipeline.config.update` carries the bounded local `playback_policy`
  extension (`prime_target_ms`, `continuity_mode`, `native_streaming`,
  `max_prime_ms`) for server-side admission freezing. Source rate remains the
  separate server-owned `audio_output_sample_rate`; never duplicate it in the
  strict playback-policy request. The matching enriched `pipeline.response`
  copy is authoritative across delayed acknowledgement/lifecycle ordering;
  retain the legacy rate-only fallback for an older local server.
- `response.done` is protocol completion, not browser audibility. When PCM is
  primed but the worklet has not rendered its first sample, retain the
  provisional ChatView transaction and defer its one terminal event. Settle it
  audible only on the matching worklet `started`; settle it unheard only on a
  definitive `drained` or `cleared` boundary, then retire the response so a
  late worklet message cannot revive it.
- Starting a conversation claims a monotonic attempt token and renders
  `connecting` before any await. Every asynchronous preflight, mic acquire,
  and connect completion must abandon and close stale resources rather than
  allowing a fast repeat click to create a second mic, socket, or client.
- Initial local identity is an uncached, latest-request-wins parallel probe of
  the selected provider and local pipeline. Retry boundedly on a transient
  failure and render explicit unavailability; never use it to manage services.
  Render a settled local-pipeline result before a slower backend/clone
  inventory has returned, so static `Checking...` text cannot persist.
- Feed the exact generated playback PCM into the capture worklet as a non-audible reference; never substitute the static clone recording. Native browser AEC is the default. Adaptive uses only the SHA-verified, import-free bundled AEC3 module; any manifest, ABI, hash, compile, or worklet failure resolves truthfully to Native. Strict suspends uncertain upload through the echo tail without inserting zero PCM.
- Keep native `echoCancellation`, `noiseSuppression`, and `autoGainControl` enabled and expose requested/effective mode, module availability, calibration, reference wiring, and double-talk status in diagnostics. The response-length setting remains independent.
- Persist delay, strict suppression, leakage, and double-talk calibration by microphone/output-device pair. Feed the worklet the active AudioContext output latency and keep Sonora's derived `aec3-output-evidence` label distinct from WebRTC's private internal double-talk state.

## Child DOX Index

- Realtime Audio Diagnostics follows the live graph and waterfall and stays
  collapsed by default. Its compact summary shows provider, Realtime-selected
  profile, delivery mode, and first-PCM latency; nested bounded Advanced
  overrides stay aligned with the candidate schema and are grouped by latency
  and phrase dispatch, conditioning and TTS sampling, and expert safety.
  Realtime persists its canonical selection under `scope=realtime`, keeps
  temporary overrides page-scoped, and can save the effective values as a new
  immutable-built-in-safe custom profile without changing Voice Studio's
  selection. Show named, override, and effective values separately; require an
  explicit unsafe unlock before editing the 25-frame decoder context. Never
  send audio.cpp tuning to another TTS provider. Expanded metrics include LLM first stable phrase, TTS first PCM,
  first playback, synthesis RTF, end-to-end time, model/profile, GPU headroom,
  paired reference source/requested/used duration, limit-applied state,
  pairing mode, truthful delivery mode,
  requested/effective echo mode, verified AEC3 identity, device calibration,
  and truthful Native fallback reasons.
- After resolving a candidate profile, carry its complete frozen effective
  Base/full-ICL tuning snapshot plus profile revision in session `tts_tuning`;
  use effective lookahead/flush locally and treat legacy `resolved` values as a
  bounded compatibility fallback only. Candidate supervisor REST payloads retain
  `scope=realtime`; strict WebSocket tuning accepts only the candidate provider,
  profile/revision, complete safe effective fields, and bounded overrides.
  Candidate preflight must complete one live profile resolution and freeze that
  exact result for the initial socket update; it fails visibly rather than
  silently opening a candidate session without `tts_tuning`. Provider switches
  clear foreign voice/tuning state.

- No child AGENTS.md files currently.

## Candidate snapshot and failure boundaries

- Studio snapshot proxies expose only validated GET model/clone snapshots and
  the bounded outcome lookup, never lifecycle mutations. Tuning resolve and
  snapshot responses, including failures/conflicts, are no-store.
- A candidate phrase is complete only after its opaque request outcome confirms
  successful engine termination. Missing/failed outcomes clear the exact owned
  playback queue and stop remaining phrases without replay or provider fallback.
- The Studio diagnostic WAV picker is explicit, local, and separate from live
  microphone capture. A selected clip goes through the normal LLM/TTS path but
  bypasses microphone/VAD admission; do not label it a physical microphone test.
