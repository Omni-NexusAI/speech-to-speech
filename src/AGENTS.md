# Package Code

## Purpose

- Contains the installable `speech_to_speech` package: Realtime API, VAD/STT/LLM/TTS handlers, connection modes, and pipeline messages.

## Local Contracts

- Direct-audio Gemma requests include session instructions and a snapshot of bounded chat history with thinking disabled; the current user message carries only audio plus an optional camera frame, never internal control text. Create exactly one semantic user anchor for every accepted turn before committing assistant text or tool calls. Upgrade it in place to validated transcript text when available; otherwise retain and later serialize its session-only WAV `input_audio` shape. Key ownership by session plus logical turn: a newer cumulative VAD revision replaces that same anchor, and stale revision cleanup must not clear the newer owner.
- Accepted direct audio may be any language or code-switched. Treat the whole accepted turn as semantic input, infer its likely natural intent, and keep transcripts optional. Spoken replies follow the current utterance unless the user or session requests another language: use one explicit language for a monolingual answer and `Auto` for mixed or unspecified output. Never default implicitly to English or derive conversation language from clone reference metadata; assistant language is turn-scoped and retained through the same turn's tool continuation.
- Direct-audio prompts must not volunteer discussion of transcription, audio quality, garbling, attached audio, or internal processing; an explicit user question about that topic may still be answered. Reject only exact failure-only transcript sentinels as absent metadata; never use that rejection to suppress an assistant response, tool call, or the single primary model request, and do not add canned response normalization.
- Progressive Gemma transcript previews are opt-in and ephemeral. Final transcript metadata comes only from the primary direct-audio response; validated text may upgrade the existing user anchor in place. Missing or malformed metadata must never start a fallback request, suppress assistant/tool output, delete its retained audio anchor, or enter model history as a placeholder.
- Local history retains 30 complete semantic user turns without automatic summarization and emits content-free context metrics when committed, upgraded, or trimmed. Shared and legacy LLM generation logs report only lifecycle counts; never log assistant text, tool names, arguments, outputs, transcripts, or audio.
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
- `max_response_tokens` is conversation-scoped through `pipeline.config.update`, defaults to 384, and constrains assistant generation only. It must not alter the 30-turn exact-history policy or optional 96-token live previews.

- Preserve the OpenAI Realtime-compatible `/v1/realtime` protocol shape.
- Keep `--stt gemma-audio` as the local direct-audio bypass mode; it must not load Parakeet or another ASR model.
- Keep `--qwen3_tts_backend openai-api` as an external server mode for Dockerized FasterQwen3TTS; it must not import or initialize in-process FasterQwen3TTS.
- In `openai-api` mode, default to `http://127.0.0.1:8881/v1`, require `1.7B-Base`, and accept clone voices as `clone:<profile_id>`.
- The `qwen3-tts-faster` API resolves clones by stable profile ID or display name and advertises native PCM streaming through `/health`; the adapter must stream each PCM chunk immediately when that capability is present.
- Prefer additive backend options over removing upstream handlers.
- Keep audio exchanged with the Realtime client as PCM16 and route through the existing pipeline queues.
- Direct-audio Gemma mode must honor Realtime session instructions, tools, and multimodal context instead of bypassing the session contract.
- Direct-audio Gemma may return optional user transcript metadata alongside its assistant response; phrase-sized assistant chunks may stream to TTS before final completion, with full-buffer behavior kept as a fallback.
- Direct-audio completion events must close the user transcription without queuing the normal `GenerateResponseRequest`; otherwise one Gemma answer can create duplicate TTS playback and block later tool turns.
- Provisional transcript bubbles are fail-closed: emit them only from a strict transcript-only Gemma preview, never from the direct assistant response formatter.
- Pipeline stage timings should be emitted as realtime `pipeline.metric` events for VAD, Gemma, TTS, playback, and end-to-end diagnostics. Accepted-input diagnostics are content-free: duration, RMS, peak, near-silence, clipping, and revision count only. TTS diagnostics report requested/effective response language and advertise Auto support only where the active provider preserves it end to end.
- Tool-only direct responses must drain their assistant/tool side-channel event before the separately queued audio completion sentinel closes the response. A response-ID terminal barrier requires both text/tool completion and audio completion before one `response.done`.
- Client tool outputs, optional camera images, and one `response.create` are accepted immediately and bound to their `call_id`; the backend starts that follow-up only after the originating response closes.
- A camera frame is a point-in-time request attachment. Retire the exact image IDs consumed by a continuation on success, failure, or cancellation; remove an image-only attachment's empty user item and turn count, while preserving images injected after the request snapshot for the next operation.
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
- Post-tool Chat Completions stream from the selected endpoint and emit the first stable sentence to TTS without waiting for the complete answer.
- Direct and post-tool transport failures must emit one failed `EndOfResponse`, release model/response ownership, and produce no canned assistant speech.
- Native tool calls always carry a concise spoken acknowledgement before the call. Prefer request-specific, naturally varied model wording in `ASSISTANT_PREAMBLE` or an assistant lead-in; use varied per-tool fallbacks only when omitted.
- Disconnect cleanup flushes every intermediate handler queue before propagating `SESSION_END`, so abandoned speculative turns cannot delay a new session.
- TTS provider selection is session-scoped: Faster on `8881` remains the default, while Groxaxo on `8882` is accepted only when Voice Studio already has a Base model loaded.
- Construct TTS handlers without contacting remote providers. Probe the
  selected TTS backend at generation/session time and return provider-specific
  readiness errors without preventing model-free pipeline startup.
- `qwen3tts-audiocpp` is an explicit candidate provider at `8890/v1`. It may be selected only after the UI validates a resident Base model and synchronizes the selected Base clone into the candidate's private library; it never starts, switches, or unloads candidate models from the realtime pipeline.
- The audio.cpp candidate requests `stream=true` raw PCM only when live status advertises native incremental PCM. Stream each verified chunk immediately; if native fails before any PCM is emitted, retry the same candidate-private clone once with the explicit buffered-phrase fallback. Never replay a request after any PCM has reached the pipeline, and preserve Faster as the default realtime provider.
- A punctuation-complete audio.cpp phrase is stable and starts TTS immediately;
  candidate look-ahead and flush timing apply only to incomplete fragments. Phrase
  coalescing may never cross a provider or cancellation-generation boundary, and
  cancellation while awaiting a flush must exit without occupying external TTS.
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
- audio.cpp completed-phrase fallback has a longer cancellable response budget
  than native PCM delivery; retain prompt Stop/barge-in cancellation instead
  of dropping a valid offline phrase on the native-stream timeout.
- A buffered candidate request cancelled by Stop, barge-in, or replacement
  before its first completed phrase emits `tts/cancelled_before_audio`, not a
  normal zero-audio completion. A non-cancelled zero-byte provider response
  emits `tts/empty_audio` and a warning so missed playback is diagnosable.

## Verification

- Candidate tuning is session-scoped as `tts_tuning` and accepted only for
  `qwen3tts-audiocpp`; it contains a validated profile id plus bounded temporary
  overrides and a bounded supervisor-resolved phrase-queue snapshot. The TTS
  handler may use that snapshot for candidate-only text look-ahead and flush
  timing; Faster and Groxaxo retain their existing ready-queue behavior. Tuning
  never changes candidate residency and is omitted from other providers.
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
