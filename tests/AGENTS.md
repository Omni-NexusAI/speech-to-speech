# Tests

## Purpose

- Contains pytest coverage for pipeline handlers, Realtime protocol behavior, and backend adapters.

## Local Contracts

- Realtime tests should cover instruction acknowledgement, direct-audio history serialization, final-only context commits, content-free trim counters, backend runtime identity, and PowerShell 5.1 lifecycle ordering.
- Direct-audio regression coverage must prove that optional transcript metadata never starts a second request or gates a response and that an explicit post-tool `response.create` still starts one follow-up. Tool-only direct responses must drain their call event before completion, and missing transcripts must retain their function-call/output pair without inserting placeholder user text.
- Cover final transcript persistence independently from the optional floating live bubble, display-only `[User audio]` fallback, Unicode-safe output, tool acknowledgement after response close, and dynamic history-token/context-window diagnostics.
- Cover immediate tool output/create ordering, session-scoped provider selection, Groxaxo Base-model detection, active-stream cancellation, and bounded TTS runaway handling.
- Direct-audio tool tests must cover opaque, already-prefixed, missing, duplicate, streaming, and buffered llama.cpp call IDs.
- Runtime descriptor tests must keep `/v1/pool` echo metadata aligned with the
  client-owned contract: Native default, validated Sonora AEC3 Adaptive with
  client-reported availability and device-pair calibration, and Strict
  fail-closed, with no legacy NLMS claim.
- Teardown coverage must prove intermediate handler queues are flushed and closed-client events/tool continuations cannot reactivate a stopped conversation.
- Remote-provider coverage must prove authenticated endpoint routing, no local probe/fallback, key redaction, zero OpenAI retries, dynamic context discovery, and superseded-revision transport cancellation.
- Model-operation tests must prove serialization, busy-preview dropping, active transport closure, two-second-style detachment, and stale-release isolation. Browser DSP tests must cover Native-default migration, manifest/hash/ABI-authenticated AEC3 loading, truthful Adaptive-to-Native fallback on module failure, render-before-capture ordering, Strict upload pause, device-pair calibration persistence, real echo reduction, near-end barge-in preservation, and untouched fallback mic PCM.
- VAD continuation tests must prove the first soft endpoint anchors the horizon, sub-threshold fragments create no pending reopen state, and revision/audio caps force a new turn.
- Timeout tests must prove direct and post-tool failures emit one failed completion, no canned speech, and leave the next response usable.
- Transport cancellation tests must prove a stream blocked before or during model output exits promptly when closed, including cancellation requested before its worker starts.
- Direct-audio cancellation coverage must assert that its generation survives the STT notifier and LLM pass-through into TTS. External PCM byte-stream cancellation and bounded phrase coalescing require deterministic mocked coverage.

- Prefer mocked HTTP and monkeypatched model setup for backend-adapter tests.
- Tests for the local Gemma/FasterQwen3TTS path must not require running local models or Docker containers.
- HFRT provider tests must cover backend-scoped live Base-clone inventory,
  per-backend selection restore/fallback, contextual mutation permissions,
  identity-scoped validation, atomic non-secret persistence, and model-free
  launcher/handler construction.
- Candidate inventory tests must mirror the selected candidate's uncached live
  inventory without cross-filtering IDs against Faster or another provider.
  Buffered cancellation tests
  must distinguish expected pre-audio cancellation from an abnormal empty
  provider response.
- Candidate/HFRT bridge tests must cover native chunk relay (`stream=true`),
  pre-first-PCM buffered fallback, no replay after emitted PCM, disconnect and
  barge-in cancellation, selected-clone and resident-model preservation,
  tuning-profile forwarding, bounded resolved look-ahead/flush behavior,
  Faster/Groxaxo phrase-queue isolation, timing metrics, accepted realtime
  provider identity/routing, no cross-provider fallback on candidate synthesis
  failure, and the native cold budget lifecycle: 45 seconds before first PCM,
  normal budget after warm PCM, and cold again after same-model reload or
  supervisor restart without leaking the allowance to other providers.
- Native phrase tests must prove a punctuation-complete first phrase bypasses
  candidate look-ahead delay, provider PCM is yielded before its stream
  completes, later phrases remain ordered exactly once, and cancelled or
  foreign-provider queued input cannot be coalesced into the active request.
- Candidate reference-metric tests must cover source/requested/used duration,
  limit-applied, pairing, and delivery headers; reject invalid numeric values,
  exclude transcript content, and prove stale metadata cannot cross into a
  different TTS provider.
- Browser contract tests must cover settings-before-inventory startup ordering,
  stale backend inventory rejection, and save-before-validation refresh for
  backend and clone changes.
- Realtime startup tests must cover the exact candidate REST-versus-WebSocket
  tuning schemas, a successful pipeline-config acknowledgement, the 15-second
  fail-closed deadline, fatal pre-ack errors, transactional rollback after an
  invalid update, legacy provider normalization, and removal of foreign tuning
  when providers switch.
- Keep live smoke testing documented separately from automated tests.
- Launcher/status regressions must reject transient container IDs and Docker
  lifecycle calls from speech-to-speech launchers while retaining the stable
  `qwen3-tts-faster` runtime identity.

## Child DOX Index

- No child AGENTS.md files currently.
