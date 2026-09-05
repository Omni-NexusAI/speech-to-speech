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
- Idle direct-Gemma pool descriptors must report token-compaction policy and
  unmeasured usage, never a legacy turn cap; non-direct descriptors retain
  their existing visible-trim/compact shape.
- Teardown coverage must prove intermediate handler queues are flushed and closed-client events/tool continuations cannot reactivate a stopped conversation.
- Remote-provider coverage must prove authenticated endpoint routing, no local probe/fallback, key redaction, zero OpenAI retries, dynamic context discovery, and superseded-revision transport cancellation.
- Model-operation tests must prove serialization, busy-preview dropping, active transport closure, two-second-style detachment, and stale-release isolation. Browser DSP tests must cover Native-default migration, manifest/hash/ABI-authenticated AEC3 loading, truthful Adaptive-to-Native fallback on module failure, render-before-capture ordering, Strict upload pause, device-pair calibration persistence, real echo reduction, near-end barge-in preservation, and untouched fallback mic PCM.
- VAD continuation tests must prove the first soft endpoint anchors the horizon, sub-threshold fragments create no pending reopen state, and revision/audio caps force a new turn.
- Timeout tests must prove direct and post-tool failures emit one failed completion, no canned speech, and leave the next response usable.
- Transport cancellation tests must prove a stream blocked before or during model output exits promptly when closed, including cancellation requested before its worker starts.
- Response-ownership tests must cover monotonic input/response epochs,
  completed-but-unheard rollback, exact-once browser playback acknowledgement,
  stale terminal suppression, forced/revised input supersession, and provisional
  user/assistant/tool context rollback without replaying a tool follow-up.
  Include a deterministic VAD-versus-direct-Gemma interleaving: the shared
  response transaction must prevent a stale user, assistant, or function-call
  history write from escaping after its provisional checkpoint rolls back.
  Include synchronous synthetic-final ordering, lifecycle-state emission,
  deferred-client-item flush on rollback/ack, nonbrowser first-PCM settlement,
  browser capability opt-in, and identity propagation through both supported
  OpenAI-compatible LLM implementations. Cover ambiguous response-metric
  rejection while retaining the router's authoritative cancellation metric,
  plus duplicate call-ID terminal/output rejection and exact-once follow-up.
  Prove that a current audible response epoch remains admissible through direct
  Gemma, both LLM implementations, LM output processing, each TTS backend, PCM,
  and its terminal even after a newer speculative input appears. Also prove that
  a completed epoch accepts only its late rendered-playback acknowledgement,
  rejects every later model/TTS/terminal output, and a sequential manual
  `response.create` receives a higher epoch that an old acknowledgement cannot target.
  Also prove that
  promotion refreshes a queued successor's rollback checkpoint so superseding it
  cannot erase history committed by the earlier audible owner, and that both
  normal LLM implementations wait for that promotion instead of dropping the
  accepted request. Token-usage coverage must retain the audible owner's usage
  when newer non-interrupting input advances speculative identity.
  An accepted provider turn with no response ID or output must emit exactly one
  epoch-only cancelled browser lifecycle terminal and release its provisional
  checkpoint.
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
  explicit native failure without silent buffered replay, disconnect and
  barge-in cancellation, selected-clone and resident-model preservation,
  tuning-profile forwarding, bounded resolved look-ahead/flush behavior,
  Faster/Groxaxo phrase-queue isolation, timing metrics, accepted realtime
  provider identity/routing, no cross-provider fallback on candidate synthesis
  failure, and the native cold budget lifecycle: 45 seconds before first PCM,
  normal budget after warm PCM, and cold again after same-model reload or
  supervisor restart without leaking the allowance to other providers.
- A phrase-local provider failure must suppress all later phrases with the same
  response identity while still allowing its terminal sentinel to close the
  response; no implicit mode retry may make a partial answer appear fused.
- PCM transport tests must prove a response freezes its negotiated output
  sample rate when ownership is claimed, before Gemma/TTS starts: a live
  16/24 kHz configuration update may affect only the next response and never
  retime its first or later PCM, or its terminal boundary. Legacy implicit
  audio may freeze after its first successful conversion.
- Response-snapshot tests must prove an owner-time copy of provider, clone,
  profile/revision, language, and seed survives a live update before the first
  LLM/TTS phrase; explicit `response.create` voice/rate overrides win at
  admission, while cancellation/supersession releases only the old snapshot.
- Native phrase tests must prove a punctuation-complete first phrase bypasses
  candidate look-ahead delay, provider PCM is yielded before its stream
  completes, later phrases remain ordered exactly once, and cancelled or
  foreign-provider queued input cannot be coalesced into the active request.
- Candidate-native tests must prove that Buffered phrase PCM remains the
  displayed default even when the native override is enabled, while native PCM
  stays an explicit experimental choice. Cache tests must distinguish reusable
  immutable encoded clone conditioning/reference codes and speaker embeddings
  from fresh phrase-specific talker prefill and generated speech state; do not
  describe the current cache as phrase-independent talker-KV reuse or decoder
  continuity.
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
- Candidate late-error tests must exercise actual outcome routes, strict
  opaque IDs, bounded expiry, immutable terminal results, and content-free
  stage/codec telemetry. Prove shared native/buffered generated-audio limits
  are distinct from wall time, aborted audio cannot be exported as complete,
  exact failed response playback is cleared, and the next turn survives.
  Offline Full WAV remains independent of live phrase-duration controls.
- Mounted Studio route tests must assert GET-only model/clone snapshots,
  uncached tuning-resolution responses including conflicts, and no arbitrary
  lifecycle proxy. Latency diagnostics require explicit opt-in and cannot
  label legacy EOF as engine-proven completion.
- Browser playback ownership tests cover `pipeline.response` epoch binding,
  stale-event rejection, idempotent first-render acknowledgement, worklet queue
  clearing, bounded Adaptive reservoir behavior, and launcher URL output
  without implicit browser launch. Cover epoch-only no-response-ID terminal
  rollback, committed tool-only no-PCM history preservation, bounded duplicate
  browser tool call-ID rejection with normal output acknowledgement, delayed
  start-attempt supersession, and response-scoped 16/24 kHz worklet clocks
  plus immutable Adaptive/Fast-start priming policy across pre-ID config races.
  Prove buffered-phrase audio.cpp receives the same bounded Adaptive reservoir,
  while the local speech-start decision remains capability-gated away from
  nonbrowser/OpenAI SDK clients.
- The native candidate acceptance harness uses a fake HTTP server in
  `test_native_audio_cpp_probe.py`; it must prove identity-only admission,
  raw sample alignment/timing, SSE event-identity exact-once behavior,
  intentional client abort plus clean follow-up, and refusal to switch models.
  These tests never contact Docker.

## Child DOX Index

- No child AGENTS.md files currently.
