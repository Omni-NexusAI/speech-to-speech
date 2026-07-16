# Package Code

## Purpose

- Contains the installable `speech_to_speech` package: Realtime API, VAD/STT/LLM/TTS handlers, connection modes, and pipeline messages.

## Local Contracts

- Direct-audio Gemma requests include session instructions and a snapshot of bounded chat history with thinking disabled; the current user message carries only audio plus an optional camera frame, never internal control text.
- Progressive Gemma transcript previews are opt-in and ephemeral; final transcription is mandatory regardless of preview mode. Missing primary transcripts use one transcript-only fallback, and assistant text must never become user history.
- Local history retains 30 complete turns without automatic summarization and emits content-free context metrics when committed or trimmed.
- The realtime backend publishes runtime identity through `/v1/pool` and `pipeline.runtime`; local UI diagnostics use it to detect stale backend code.
- A Remote model endpoint must own every model operation for its conversation, including audio/transcription, vision/tools, follow-ups, tokenization, context discovery, and identity. Never probe or fall back to Local for that session.
- Soft VAD endpoints settle for 250 ms. A newer uncommitted audio revision cancels the obsolete transport while retaining and resubmitting combined captured audio.

- Preserve the OpenAI Realtime-compatible `/v1/realtime` protocol shape.
- Keep `--stt gemma-audio` as the local direct-audio bypass mode; it must not load Parakeet or another ASR model.
- Keep `--qwen3_tts_backend openai-api` as an external server mode for Dockerized FasterQwen3TTS; it must not import or initialize in-process FasterQwen3TTS.
- In `openai-api` mode, default to `http://127.0.0.1:8881/v1`, require `1.7B-Base`, and accept clone voices as `clone:<profile_id>`.
- The `qwen3-tts-faster` API resolves clones by stable profile ID or display name and advertises native PCM streaming through `/health`; the adapter must stream each PCM chunk immediately when that capability is present.
- Prefer additive backend options over removing upstream handlers.
- Keep audio exchanged with the Realtime client as PCM16 and route through the existing pipeline queues.
- Direct-audio Gemma mode must honor Realtime session instructions, tools, and multimodal context instead of bypassing the session contract.
- Direct-audio Gemma should return a user transcript plus assistant response; phrase-sized assistant chunks may stream to TTS before final completion, with full-buffer behavior kept as a fallback.
- Direct-audio completion events must close the user transcription without queuing the normal `GenerateResponseRequest`; otherwise one Gemma answer can create duplicate TTS playback and block later tool turns.
- Provisional transcript bubbles are fail-closed: emit them only from a strict transcript-only Gemma preview, never from the direct assistant response formatter.
- Pipeline stage timings should be emitted as realtime `pipeline.metric` events for VAD, Gemma, TTS, playback, and end-to-end diagnostics.
- Tool-only direct responses must drain their assistant/tool side-channel event before the separately queued audio completion sentinel closes the response. A response-ID terminal barrier requires both text/tool completion and audio completion before one `response.done`.
- Client tool outputs, optional camera images, and one `response.create` are accepted immediately and bound to their `call_id`; the backend starts that follow-up only after the originating response closes.
- Normalize opaque llama.cpp function-call IDs to the Realtime `call_*` contract at the direct-audio Gemma adapter boundary; keep the global chat validator strict.
- Full-buffer Gemma mode keeps a cancellable streaming HTTP transport and buffers text locally so session Stop can abort in-flight generation.
- Realtime config uses `pipeline.config.update` for model endpoint, `full_buffer_tts`, live-preview state, and the conversation-scoped TTS provider; retain `local.pipeline.update` only as a compatibility alias.
- Stop, disconnect, barge-in, and replacement sessions close active direct and post-tool Gemma streams plus active TTS HTTP streams before releasing the pipeline slot.
- Disconnect cleanup flushes every intermediate handler queue before propagating `SESSION_END`, so abandoned speculative turns cannot delay a new session.
- TTS provider selection is session-scoped: Faster on `8881` remains the default, while Groxaxo on `8882` is accepted only when Voice Studio already has a Base model loaded.

## Verification

- Run focused pytest tests for modified handlers before broader checks.

## Child DOX Index

- No child AGENTS.md files currently.
