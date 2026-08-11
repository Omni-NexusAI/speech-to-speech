# Tests

## Purpose

- Contains pytest coverage for pipeline handlers, Realtime protocol behavior, and backend adapters.

## Local Contracts

- Realtime tests should cover instruction acknowledgement, exact historical `input_audio` serialization when both transcript and semantic memory are absent, same-primary-response `USER_MEMORY` durability, final-only context commits, content-free trim counters, backend runtime identity, and PowerShell 5.1 lifecycle ordering.
- Direct-audio regression coverage must prove every accepted turn retains exactly one semantic user anchor before assistant/function-call/output state. Validated transcript text upgrades that anchor in place and wins over `USER_MEMORY`; a bounded validated memory may upgrade only when transcript is absent, while absent or rejected metadata retains `input_audio`. Memory remains hidden/display-only, never reaches logs, and never gates assistant or tool output. Streamed, buffered, cancellation, revision, session-reuse, tool-order, and language-continuation tests must preserve that rule. A newer cumulative revision in the same session/turn replaces that anchor without duplicating history, stale cleanup cannot release its newer owner, and reused turn IDs in another session cannot alias it. Metadata never starts a second request. An explicit post-tool `response.create` still starts one follow-up, and missing metadata retains its function-call/output pair without inserting placeholder user text.
- Multilingual direct-audio coverage must prove there is no implicit English fallback, monolingual responses preserve their explicit language, mixed or unspecified responses retain `Auto`, stale turn language is cleared, and the effective language survives same-turn tool continuation. Exact failure-only transcript sentinels are absent metadata while legitimate user sentences containing the same words remain intact; neither case may change response or tool delivery.
- Accepted-turn diagnostic coverage must prove duration, RMS, peak, near-silence, clipping, and revision count are emitted without transcript or response content. Shared and legacy LLM log tests must use secret sentinels to prove assistant and tool content never reaches logs. Provider metrics must distinguish requested/effective TTS language and report Auto support truthfully rather than inferring it from clone metadata.
- The deterministic accepted-turn gate must exercise at least 100 turns and prove exactly one primary Gemma request per turn even when optional transcript metadata is absent. Model-level clarification-rate listening validation remains a separate live evidence gate.
- Clarification-rate probe tests must prove the 100 normal turns are divided evenly across the five declared acoustic/language cohorts, every normal challenge and expected value is unique across retained history, meaningless input remains outside the normal denominator, SAPI receives explicit UTF-8 bytes through stdin and returns WAVs only in memory, a real Windows System.Speech runner accepts a multilingual fixture under the active process encoding, failed voice enumeration is isolated to a separate catalog process, and a clean synthesis child uses a fresh default synthesizer per scenario with false culture/alternate coverage when no safe match is available. Seeded noise is deterministic, multilingual fixtures survive an explicit UTF-8/codepoint round trip, clarification and garbling signals take precedence over recovered answer tokens, the exact managed loopback settings handoff rejects redirects before synthesis or model requests, production transcript-versus-memory precedence and historical serialization are reused, ambiguous semantic output fails closed, exact attempted/completed counts survive partial failures, one primary request is counted per accepted turn, and the CLI emits exactly one aggregate JSON object with empty stderr while restoring production logger state. Public success/failure output cannot contain utterances, audio, transcripts, responses, endpoint/model/voice identity, or credentials.
- Cover final transcript persistence independently from the optional floating live bubble, display-only `[User audio]` fallback, single durable semantic user history, Unicode-safe output, tool acknowledgement after response close, and dynamic history-token/context-window diagnostics.
- Cover immediate tool output/create ordering, session-scoped provider selection, Groxaxo Base-model detection, active-stream cancellation, and bounded TTS runaway handling.
- Direct-audio tool tests must cover opaque, already-prefixed, missing, duplicate, streaming, and buffered llama.cpp call IDs. Preserve explicit `ASSISTANT_PREAMBLE` text exactly; omitted preambles execute silently without canned substitution or `ASSISTANT_RESPONSE` rewriting.
- Reject missing, non-string, wrong-type, and extra tool arguments before execution without echoing raw values. Preserve normal output/image/single-`response.create` ordering.
- Give every real camera invocation a distinct visible generation/card, including unavailable captures. Follow-ups asking about the current view or changes require a fresh snapshot rather than silently reusing an old frame.
- Camera cleanup tests must retire exact consumed image IDs on successful, failed, and cancelled continuations, remove image-only empty turns from serialization and turn counts, and preserve a newer image injected after the active request snapshot.
- Runtime descriptor tests must keep `/v1/pool` echo metadata aligned with the
  client-owned contract: Native default, validated Sonora AEC3 Adaptive with
  client-reported availability and device-pair calibration, and Strict
  fail-closed, with no legacy NLMS claim.
- Teardown coverage must prove intermediate handler queues are flushed and closed-client events/tool continuations cannot reactivate a stopped conversation.
- Remote-provider coverage must prove authenticated endpoint routing, no local probe/fallback, key redaction, zero OpenAI retries, dynamic context discovery, and superseded-revision transport cancellation.
- Model-operation tests must prove serialization, busy-preview dropping, active transport closure, two-second-style detachment, and stale-release isolation. Browser DSP tests must cover Native-default migration, manifest/hash/ABI-authenticated AEC3 loading, truthful Adaptive-to-Native fallback on module failure, render-before-capture ordering, Strict upload pause, device-pair calibration persistence, real echo reduction, near-end barge-in preservation, and untouched fallback mic PCM.
- Browser playback tests must deterministically cover native audio.cpp cold and learned startup priming, acknowledgement-scoped immutable signatures, same-turn tool-continuation policy freezing, logical decoder-block timing derived from samples rather than WebSocket packets, fragmented and coalesced block delivery, last-eight p95 plus fixed-jitter learning, conservative invalid-metadata ceilings, persisted-state TTL/LRU and eligibility reconstruction, exact first/last-sample preservation, full-ceiling underrun re-priming and three-clean-response recovery, short-final flush, cancellation/reconnect clearing, current-generation non-mic re-arming, stale-generation rejection, bounded completed-response tombstones, content-free diagnostics, and worklet-started/drained UI lifecycle independent from network response locks. Faster, Groxaxo, and buffered fallback must retain zero-target immediate playback.
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
- Voice-library tests must prove both the pipeline handler and HFRT server use
  the same portable `~/.speech-to-speech/qwen3-tts-voices` fallback while
  preserving `VOICE_LIBRARY_DIR` as the explicit shared override.
- Gemma launcher tests must reject workstation path defaults and require the
  llama.cpp, main-model, draft-model, and mmproj locations to come from explicit
  parameters or environment variables.
- Default-voice tests must cover valid `selected_profile.json`, deterministic
  first-live-Base fallback, explicit overrides, empty-library failure, stale UI
  clearing, and deletion of whichever profile currently happens to be selected.
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
- Content-free Realtime harness tests must prove configuration acknowledgement
  precedes all PCM, the second spoken turn depends on the first, malformed tool
  arguments cause no output/create side effect, function output acknowledgement
  precedes exactly one tool-disabled follow-up create, required tool choice is
  restored to automatic before the next turn, tool results survive a later spoken
  turn, cancellation closes in protocol order, and recovery succeeds without
  serializing any prompt, audio, transcript, response, tool value, endpoint,
  model, voice, or credential.
- Launcher/status regressions must reject transient container IDs and Docker
  lifecycle calls from speech-to-speech launchers while retaining the stable
  `qwen3-tts-faster` runtime identity.

## Child DOX Index

- No child AGENTS.md files currently.
