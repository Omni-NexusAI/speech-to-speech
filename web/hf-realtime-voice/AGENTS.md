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
- Keep the upstream `Built by` credit intact. The top identity row must source local Gemma and FasterQwen3TTS provider status from `/api/local-pipeline`, and place the Omni-NexusAI fork credit in a separate `Modified by` row.
- When the backend emits `pipeline.metric`, keep the UI rendering lightweight and diagnostic-only; do not infer pipeline state by duplicating backend logic in the browser.
- Local runtime toggles and TTS backend selection use `pipeline.config.update`, leaving OpenAI-compatible `session.update` for standard voice/instructions/tool fields.
- Initial capture remains gated until `pipeline.config.updated` acknowledges the
  complete conversation-scoped configuration. A missing acknowledgement times
  out after 15 seconds; any server error before it is fatal, visible, and closes
  the socket so the backend releases the pipeline slot.
- Send tool output, optional camera image, and one follow-up `response.create` immediately in hosted order. The backend owns the call-ID barrier; never replay a rejected create on a later user turn.
- Speech stop reserves persistent visual user chronology. Replace it with validated transcript metadata when available or persistent `[User audio]` when absent; both are UI-only representations. The backend separately owns one semantic user anchor, upgrading audio to validated text when available, and transcript availability never controls assistant or tool UI.
- The live-transcription setting controls only the temporary floating user bubble. Final user text always updates the persistent conversation panel.
- Diagnostics label the always-on final stage `Transcription` and show retained history tokens against the context window detected from llama.cpp.
- Camera preview and Diagnostics should not occupy the same desktop corner; keep the camera self-view clear when diagnostics are open.
- Reject missing, non-string, wrong-type, and extra tool arguments before execution without displaying raw values. Preserve output/image/single-`response.create` ordering.
- `web_search` accepts required `query` plus optional `mode=auto|web|news` and `freshness=none|day|week|month|year`. Auto uses news when freshness is requested and web otherwise; Serper receives the matching endpoint and qdr filter. Only zero-result news selected by `mode=auto` may make one same-filter web fallback; explicit news stays news with zero results.
- Search tool output is versioned structured JSON with retrieval time, requested/effective mode, filter/fallback state, and bounded results carrying title, snippet, URL, date, source, and position. Retrieval time is not publication time; never label undated results as today, and suppress unscoped answer panels for news or freshness-filtered requests.
- The shared direct/post-tool search policy permits one initial search and at most one distinct narrower refinement per accepted turn. Bind that budget to the server item ID so cumulative VAD revisions cannot reset it. Normalized duplicates or broader retries are terminal. The first successful search follow-up exposes only `web_search` with response-scoped automatic choice; all terminal search and non-search follow-ups use response-scoped `tool_choice: none`. Never mutate session policy. Search diagnostics contain only fixed modes/states, booleans, counts, timings, and bounded error classes—never queries, result content, or provider bodies.
- Render every real camera invocation as a distinct generation-keyed card, including unavailable captures. Carry bounded content-free accepted-turn, response, item, call, card, capture-generation, capture-status, and output-acknowledgement identities through requested, completed, unavailable, and rejected states; missing or malformed correlation stays visibly missing and never suppresses the card. Requests about the current view or what changed require a fresh snapshot; never imply that a stale frame is live.
- Settings list every configured TTS provider even while unavailable. Provider
  changes apply to the next conversation, and an unavailable selected provider
  blocks start without silent fallback.
- Voice inventory and profile controls are scoped to the selected backend.
  Reconcile Faster's live voices with its configured writable library, use the
  candidate-private audio.cpp profile API, and keep Groxaxo inventory-only
  unless its existing API explicitly advertises safe mutation support.
- Resolve the shared Faster clone library from `VOICE_LIBRARY_DIR`; when it is
  unset, use `~/.speech-to-speech/qwen3-tts-voices`, matching the pipeline
  handler. A missing portable library is an empty inventory, never permission
  to borrow profiles from another provider or a developer checkout.
- No clone is privileged or undeletable. Preserve an explicit backend-scoped
  selection; otherwise use a valid `selected_profile.json`, then the first live
  Base profile, and clear/disable voice state when the inventory is empty.
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
  Never retain model or service API keys there; those remain in browser-local
  device storage and are excluded from every UI-server persistence payload.
- General provider validation checks health, resident model, capabilities,
  selected clone presence, and a short synthesis without changing model
  residency. Persist and retrieve results by backend/model/clone so switching
  between previously validated candidate clones does not erase either record;
  experimental audio.cpp still requires explicit validation.
- Persist non-secret backend endpoints, prompts, selected backend, and
  per-backend clone selections atomically through server-managed runtime state.
  Await saves in the Settings UI and keep all API keys in browser-local device
  storage only, excluded from every UI-server persistence payload and response.
- Keep model inference controls separate from TTS controls, and place Voice directly beneath TTS Backend in the vertically scrolling settings layout.
- The loopback `/api/config` response carries only the managed launch revision, dirty flag, runtime-source SHA-256 fingerprint, and four browser asset generations. Snapshot this bounded identity once per frontend process and compare it exactly with `pipeline.runtime`; never expose paths, branches, environment values, config content, prompts, media, or credentials.
- Stop invalidates the active client before asynchronous teardown; closed-client mic, playback, tool, and WebSocket events must never change the idle UI or enter a replacement conversation.
- Feed the exact generated playback PCM into the capture worklet as a non-audible reference; never substitute the static clone recording. Adaptive is the default and uses only the SHA-verified, import-free bundled AEC3 module; any manifest, ABI, hash, compile, or worklet failure resolves truthfully to Native. Strict suspends uncertain upload through the echo tail without inserting zero PCM.
- Native-fallback and AEC3 capture share one stateful, phase-indexed
  windowed-sinc conversion to 16 kHz. Average all available channels to finite
  mono before capture/reference processing, retain filter state across browser
  render blocks, support a bounded idempotent clean-endpoint tail flush for
  explicit callers, and saturate PCM16 at `-32768..32767`. Normal session
  teardown aborts and discards the tail. Anti-alias filtering must not change
  VAD, gate, AEC, echo-mode, or reference-tail decisions. A route reset resolves
  the old route once, then clears FIR, partial-chunk, and reference-level state
  before accepting samples from the replacement route.
- Keep one generation-tagged playback worklet FIFO across phrase chunks and same-turn tool continuations. Validated native audio.cpp playback freezes one acknowledged model/clone/profile/effective-settings signature for the complete accepted user turn, including tool continuations; Faster, Groxaxo, and buffered fallback remain immediate. The cold ceiling is the acknowledged first plus steady decoder-block duration, with conservative 800/1280/1760 ms built-in fallbacks and a 2000 ms custom fallback when metadata is invalid. Learn warm starts only from logical decoder-block sample boundaries after two clean full-prime responses, using the last-eight p95 gap plus 64 ms jitter and never less than first-block duration plus 160 ms. A real underrun immediately restores the full ceiling and requires three clean full-prime recoveries. Preserve every input sample, flush a short ended stream, re-arm cancellation only after accepted current-generation PCM, and reject stale-generation tails. Retire completed response snapshots into a bounded tombstone set so long sessions do not leak memory or reopen late PCM.
- Adaptive safe-start diagnostics are content-free and response-scoped: expose cold ceiling, effective target/mode, latest logical block gap, p95, fixed 64 ms jitter, queued duration, underrun/re-prime counters, learning state, and a bounded fallback reason. Never expose the raw signature, model, clone, prompt, transcript, or audio content through playback metrics or persisted learning state.
- Treat worklet `started` and `drained` as the exclusive audible/UI speaking lifecycle. Network PCM, transcript, content-part, and `response.done` events may advance protocol state or release response/tool locks, but must not claim or end audible playback.
- Keep native `echoCancellation`, `noiseSuppression`, and `autoGainControl` enabled and expose requested/effective mode, module availability, calibration, reference wiring, and double-talk status in diagnostics. The response-length setting remains independent.
- Reduce resolved microphone/output routes immediately to a domain-separated
  SHA-256 fingerprint; raw device IDs, group IDs, and labels are transient
  digest input only and must never be retained, logged, persisted, rendered, or
  emitted in diagnostics. Fail closed without a persistence key when either
  physical route cannot be resolved. A changed route advances an async-guarded
  epoch, resets AEC state and the measurement cohort, and reloads only that
  route's calibration; an unchanged fingerprint refreshes latency without a
  reset. Remove route listeners during teardown.
- Persist at most 16 finite, normalized opaque-route calibrations. Legacy raw
  keys are discarded in browser-local load/save, public settings, and server
  GET/PUT paths. Delay remains available to Adaptive and Strict; suppression,
  leakage, and double-talk controls are Strict-only, while `echoTailMs` is
  bounded to 350–1000 ms. "Use measured" requires at least 20 quiet,
  playback-active, no-double-talk samples spanning at least two seconds and an
  accepted median/p95/jitter result after current output-latency adjustment.
  Keep Sonora's derived `aec3-output-evidence` label distinct from WebRTC's
  private internal double-talk state.

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
  explicit unsafe unlock before editing the 72-frame decoder context. Never
  send audio.cpp tuning to another TTS provider. Expanded metrics include LLM first stable phrase, TTS first PCM,
  first playback, synthesis RTF, end-to-end time, model/profile, GPU headroom,
  requested/effective response language and declared provider Auto capability,
  paired reference source/requested/used duration, limit-applied state,
  pairing mode, truthful delivery mode,
  requested/effective echo mode, verified AEC3 identity, device calibration,
  and truthful Native fallback reasons.
- After resolving a candidate profile, carry only its bounded
  `text_lookahead`/`phrase_flush_ms` snapshot in session `tts_tuning` so HF
  Realtime can apply the same phrase queue without hardcoding profile defaults.
  Drop stale resolved values when the selected profile changes.
- Candidate supervisor REST payloads retain `scope=realtime`; WebSocket
  `tts_tuning` is strictly limited to `provider`, `profile_id`, `overrides`, and
  optional `resolved`. Candidate preflight must complete one live profile
  resolution, and provider switches clear foreign voice/tuning state.

- No child AGENTS.md files currently.
