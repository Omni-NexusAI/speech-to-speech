# Package Code

## Purpose

- Contains the installable `speech_to_speech` package: Realtime API, VAD/STT/LLM/TTS handlers, connection modes, and pipeline messages.

## Local Contracts

- Direct-audio Gemma requests include session instructions and a snapshot of bounded chat history with thinking disabled; the current user message carries only audio plus an optional camera frame, never internal control text.
- Progressive Gemma transcript previews are opt-in and ephemeral. Final transcript metadata comes only from the primary direct-audio response; missing or malformed metadata must never start a fallback request, suppress assistant/tool output, or enter model history as a placeholder.
- Direct-audio sessions retain exact settled context without an arbitrary turn cap. Token-budgeted history maintenance triggers at 70% of the selected model window, targets 50%, and protects the six most recent settled exchanges. Its compressed memory is serialized distinctly from dialogue; missing transcript metadata may retain only the actual assistant answer, never a fabricated user transcript. Opaque accepted-turn provenance groups a transcript-less assistant/tool/final sequence as one compactable exchange. Maintenance uses the shared coordinator only while idle, is preemptible by speech, and leaves history unchanged on failure.
- The realtime backend publishes runtime identity through `/v1/pool` and `pipeline.runtime`; local UI diagnostics use it to detect stale backend code.
- Runtime echo metadata declares Native as the default and the browser as owner.
  Adaptive is effective only when the client reports a validated Sonora AEC3
  WASM module with device-pair calibration; module failure falls back to Native,
  while Strict remains fail-closed. Do not advertise the removed server-side
  NLMS filter.
- A Remote model endpoint must own every model operation for its conversation, including audio/transcription, vision/tools, follow-ups, tokenization, context discovery, and identity. Never probe or fall back to Local for that session.
- Soft VAD endpoints settle for 250 ms. Continuation requires 192 ms of confirmed speech and uses a fixed horizon anchored to the first soft endpoint; it is bounded to eight revisions and 30 seconds of combined audio. A newer uncommitted revision cancels only the obsolete transport while retaining captured audio.
- Direct audio, optional preview, and post-tool generation share one conversation-scoped model-operation coordinator. Optional previews drop while occupied; required operations serialize.
- Cancellable model streams use async-task cancellation behind the synchronous handlers. Do not replace this with cross-thread `httpx.Client.close()`: on Windows it can return locally while llama.cpp keeps the inference slot busy.
- Direct-audio cancellation identity must propagate through `DirectAssistantResponse`, `DirectAssistantRequest`, LLM chunks, response end, TTS input, and PCM output. A stale generation must be discarded before it can occupy external TTS.
- The external FasterQwen3TTS PCM stream also uses the async-task-owned cancellation transport. Keep phrase coalescing bounded so a delayed long response cannot become one oversized blocking request.
- `max_response_tokens` is conversation-scoped through `pipeline.config.update`, defaults to 384, and constrains assistant generation only. Token-budgeted direct-history maintenance includes it as a response reserve; it must not change the protected recent-exchange count or optional 96-token live previews.

- Preserve the OpenAI Realtime-compatible `/v1/realtime` protocol shape.
- Keep `--stt gemma-audio` as the local direct-audio bypass mode; it must not load Parakeet or another ASR model.
- Keep `--qwen3_tts_backend openai-api` as an external server mode for Dockerized FasterQwen3TTS; it must not import or initialize in-process FasterQwen3TTS.
- In `openai-api` mode, default to `http://127.0.0.1:8881/v1`, require `1.7B-Base`, and accept clone voices as `clone:<profile_id>`.
- The `qwen3-tts-faster` API resolves clones by stable profile ID or display name and advertises native PCM streaming through `/health`; the adapter must stream each PCM chunk immediately when that capability is present.
- Prefer additive backend options over removing upstream handlers.
- Keep Realtime audio PCM16 and route it through the existing pipeline queues. Faster/Groxaxo and microphone/VAD remain 16 kHz; only the isolated audio.cpp candidate may carry a frozen PCM16/24 kHz `AudioOutput.source_sample_rate` through the router to a server-acknowledged browser clock. Freeze the selected output clock when a response owner is claimed, before Gemma/TTS begins; a live configuration acknowledgement applies only to the next response. Legacy implicit audio paths freeze after their first successful conversion.
- Direct-audio Gemma mode must honor Realtime session instructions, tools, and multimodal context instead of bypassing the session contract.
- Direct-audio Gemma may return optional user transcript metadata alongside its assistant response; phrase-sized assistant chunks may stream to TTS before final completion, with full-buffer behavior kept as a fallback.
- Direct-audio completion events must close the user transcription without queuing the normal `GenerateResponseRequest`; otherwise one Gemma answer can create duplicate TTS playback and block later tool turns.
- Provisional transcript bubbles are fail-closed: emit them only from a strict transcript-only Gemma preview, never from the direct assistant response formatter.
- Pipeline stage timings should be emitted as realtime `pipeline.metric` events for VAD, Gemma, TTS, playback, and end-to-end diagnostics. Response-scoped Gemma/TTS metrics must carry an unambiguous response epoch; only the router's own terminal cancellation result may describe a just-staled response.
- Tool-only direct responses must drain their assistant/tool side-channel event before the separately queued audio completion sentinel closes the response. A response-ID terminal barrier requires both text/tool completion and audio completion before one `response.done`.
- Client tool outputs, optional camera images, and one `response.create` are accepted immediately and bound to their `call_id`; the backend starts that follow-up only after the originating response closes. Keep a bounded connection-local tombstone for completed call IDs so repeated function-call terminal events or duplicate outputs cannot append history, execute twice, or start another follow-up.
- Normalize opaque llama.cpp function-call IDs to the Realtime `call_*` contract at the direct-audio Gemma adapter boundary; keep the global chat validator strict.
- Full-buffer Gemma mode keeps a cancellable streaming HTTP transport and buffers text locally so session Stop can abort in-flight generation.
- Realtime config uses `pipeline.config.update` for model endpoint, `full_buffer_tts`, live-preview state, and the conversation-scoped TTS provider; retain `local.pipeline.update` only as a compatibility alias.
- Validate every pipeline-config field, candidate tuning snapshot, and model
  endpoint against a proposed copy before mutating session runtime state. Commit
  the provider, tuning, flags, and model endpoint together, normalize the legacy
  `audio-cpp` provider alias, and clear candidate tuning when another provider is
  selected.
- Stop, disconnect, barge-in, and replacement sessions close active direct and post-tool Gemma streams plus active TTS HTTP streams before releasing the pipeline slot.
- A model transport that does not close within two seconds is detached generation-safely. Its late events are rejected, while the conversation, context, and next response remain usable.
- Realtime response ownership is conversation-local and uses monotonic
  `input_epoch`/`response_epoch` values. Claim `pending` at accepted VAD stop,
  invalidate the old epoch synchronously when newer speech starts, and reject
  stale text, terminal, and metric events before protocol side effects.
  A completed owner remains current only for an idempotent rendered-playback
  acknowledgement; it is no longer output-admissible, so late same-epoch text,
  PCM, metrics, and duplicate terminals are discarded. A no-protocol terminal
  may cancel only an output-admissible owner with no response ID; duplicate
  finish/cleanup after completion is a strict no-op and cannot roll history back.
  `response.done` is not audible: only the idempotent local
  `pipeline.playback.started` acknowledgement (`response_id`, `response_epoch`)
  commits a provisional chat transaction. Unheard responses roll back their
  user/assistant/pending-tool writes; heard interrupted turns remain history.
  Browser clients declare rendered-ack support with
  `pipeline.playback.capability`; nonbrowser clients retain settlement after the
  first non-empty PCM frame is successfully sent. Safe client inserts wait behind a provisional response and flush
  exactly once on acknowledgement or pre-audible rollback, while stale tool
  outputs are discarded. Additive `pipeline.response` events expose pending,
  producing, priming, completed, and cancelled lifecycle states. An accepted
  audio turn that produces no response ID, text, tool, or PCM emits one
  epoch-only cancelled lifecycle terminal to opted-in browsers; repeated
  terminal sentinels must not duplicate it.
- Direct-Gemma history writes use the response-ownership transaction guard:
  admission plus the short user/assistant/function-call `Chat` mutation is
  atomic with VAD invalidation and provisional rollback. Never hold that guard
  across Gemma, HTTP, queue waits, or another backend's work.
- Downstream admission is epoch-first: direct Gemma, local and OpenAI-compatible
  LLM streams, LM output processing, all TTS inputs, PCM chunks, and terminal
  sentinels must honor an explicit current `response_epoch` before consulting
  speculative turn state. Speculative identity is only the fallback for legacy
  messages without an epoch authority. A non-interrupt successor may wait behind
  an audible owner; both normal LLM entry points must wait for promotion rather
  than reject the queued epoch. Refresh its provisional chat checkpoint at
  promotion before model output begins so later rollback preserves the heard
  owner's commits. Token usage is admitted by the same response epoch and must
  not be revoked merely because a newer speculative input is waiting.
- Post-tool Chat Completions stream from the selected endpoint and emit the first stable sentence to TTS without waiting for the complete answer.
- Direct and post-tool transport failures must emit one failed `EndOfResponse`, release model/response ownership, and produce no canned assistant speech.
- Native tool calls always carry a concise spoken acknowledgement before the call. Prefer a model-provided `ASSISTANT_PREAMBLE` or assistant lead-in, with a small per-tool fallback when omitted.
- Disconnect cleanup flushes every intermediate handler queue before propagating `SESSION_END`, so abandoned speculative turns cannot delay a new session.
- TTS provider selection is session-scoped: Faster on `8881` remains the default, while Groxaxo on `8882` is accepted only when Voice Studio already has a Base model loaded.
- Construct TTS handlers without contacting remote providers. Probe the
  selected TTS backend at generation/session time and return provider-specific
  readiness errors without preventing model-free pipeline startup.
- `qwen3tts-audiocpp` is an explicit candidate provider at `8890/v1`. It may be selected only after the UI validates a resident Base model and synchronizes the selected Base clone into the candidate's private library; it never starts, switches, or unloads candidate models from the realtime pipeline.
- The audio.cpp candidate requests `stream=true` raw PCM only when live status advertises native incremental PCM. Stream each verified chunk immediately; a native failure is surfaced for that response and must never trigger a silent buffered replay. Buffered phrase PCM remains a separate explicit user-selected mode, and Faster remains the default realtime provider.
- A punctuation-complete audio.cpp phrase is stable and starts TTS immediately;
  candidate look-ahead and flush timing apply only to incomplete fragments. Phrase
  coalescing may never cross a provider or cancellation-generation boundary, and
  cancellation while awaiting a flush must exit without occupying external TTS.
- Freeze one immutable synthesis snapshot for every `response_epoch`: provider,
  endpoint/model lifecycle, clone/reference fingerprint, profile/revision,
  effective candidate tuning, language, and seed. A missing profile seed becomes
  one response-local uint32 sent only as `tuning.overrides.seed`; later phrases
  must reuse it and may never coalesce across `input_epoch` or `response_epoch`.
  Candidate clone snapshots additionally freeze the Base profile's content
  revision/hash and exact reference bytes/excerpts, because a clone ID is a
  mutable library slot. Release the snapshot only on terminal response or
  cancellation (not a phrase-local/runaway provider error) so live settings
  changes take effect on the next answer, never mid-answer. Session teardown
  clears every snapshot, phrase count, and first-phrase metric key before epochs
  restart for a new connection. TTS metrics carry
  `input_epoch`, `response_epoch`, and `response_id` as top-level event fields;
  do not recover ownership from optional metric detail.
- A candidate response without a modern validated tuning snapshot still sends a
  minimal nested tuning envelope containing the frozen response seed (and an
  optional profile ID). Legacy `resolved` phrase-queue metadata must never be
  promoted or forwarded as an engine `effective` policy.
- Create the lightweight provider/profile/voice/language/tuning admission copy
  when the owner is claimed, not at first TTS phrase. It must contain only
  synthesis values (never Chat, locks, callbacks, or secrets), honour explicit
  `response.create` voice/rate overrides, and be released on completion,
  cancellation, or pre-audible supersession.
- For `qwen3tts-audiocpp`, lifecycle identity is the health-derived pair
  `supervisorInstanceId` plus non-negative `engineEpoch`, not a mutable
  runtime-setting value. Freeze and submit both on every phrase; candidate
  admission returns 409 when either changes. An older candidate that omits the
  instance ID fails closed, while Faster and Groxaxo receive neither private
  field.
- Give only a cold native audio.cpp endpoint/model/lifecycle epoch a bounded
  45-second first-PCM allowance. The first complete native PCM sample marks that
  epoch warm, restoring the normal 12-60-second budget; a same-model reload or
  supervisor restart changes the status-event epoch and must become cold again.
  Faster, Groxaxo, buffered fallback, and cancellation behavior never inherit
  this allowance.
- audio.cpp clone requests preserve the selected candidate-private clone ID and
  must not retry a Faster fallback clone when that candidate request fails.
- The Realtime router accepts `qwen3tts-audiocpp` (and normalizes the legacy
  `audio-cpp` alias) as a conversation-scoped TTS provider. Candidate speech
  payloads must carry the model identity returned by its live resident-model
  status; never reuse the handler's Faster model name for that request.
- Explicitly selected audio.cpp completed-phrase mode has a longer cancellable
  response budget than native PCM delivery; retain prompt Stop/barge-in
  cancellation for that offline request without converting a native timeout
  into an implicit retry.
- A buffered candidate request cancelled by Stop, barge-in, or replacement
  before its first completed phrase emits `tts/cancelled_before_audio`, not a
  normal zero-audio completion. A non-cancelled zero-byte provider response
  emits `tts/empty_audio` and a warning so missed playback is diagnosable.
- A provider transport failure is terminal for the remaining phrases of that
  assistant response. Preserve its frozen diagnostic snapshot until the
  response terminal arrives, but suppress every later phrase instead of
  producing a truncated, fused, or restarted spoken answer.

## Verification

- Candidate tuning is session-scoped as `tts_tuning` and accepted only for
  `qwen3tts-audiocpp`. New WebSocket snapshots require provider, profile id,
  positive profile revision, complete bounded `effective` Base/full-ICL fields,
  and bounded temporary overrides. A built-in profile may set `model=null` to
  follow the health-derived resident Base model; the response must still freeze
  that concrete model plus supervisor lifecycle identity, and arbitrary model
  IDs remain invalid. Legacy `resolved` phrase timing is fallback only. The
  response freezes `effective` for sampling and phrase timing, while
  Faster/Groxaxo retain their existing ready-queue behavior. Tuning never
  changes candidate residency and is omitted from other providers.
- Candidate diagnostics keep first stable phrase, TTS request, first PCM,
  browser-owned first playback, synthesis RTF, and end-to-end timing as distinct
  measurements rather than treating buffered completion as first PCM.
- Candidate `tts.done` diagnostics consume only bounded response metadata:
  reference source/requested/used seconds, whether a paired limit applied,
  `full` versus `matched-excerpt` pairing, and the truthful delivery mode.
  Never carry reference transcript content into pipeline metrics, and clear
  response metadata between provider requests so it cannot cross providers.

- Run focused pytest tests for modified handlers before broader checks.

## Child DOX Index

- No child AGENTS.md files currently.
