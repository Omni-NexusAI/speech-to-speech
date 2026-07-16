# Tests

## Purpose

- Contains pytest coverage for pipeline handlers, Realtime protocol behavior, and backend adapters.

## Local Contracts

- Realtime tests should cover instruction acknowledgement, direct-audio history serialization, final-only context commits, content-free trim counters, backend runtime identity, and PowerShell 5.1 lifecycle ordering.
- Direct-audio regression coverage must prove that a final transcript does not enqueue a second normal response and that an explicit post-tool `response.create` still starts one follow-up. Tool-only direct responses must drain their call event before completion, and missing transcripts must retain their function-call/output pair without inserting placeholder user text.
- Cover final transcript persistence independently from the optional floating live bubble, transcript-only fallback, Unicode-safe output, tool acknowledgement after response close, and dynamic history-token/context-window diagnostics.
- Cover immediate tool output/create ordering, session-scoped provider selection, Groxaxo Base-model detection, active-stream cancellation, and bounded TTS runaway handling.
- Direct-audio tool tests must cover opaque, already-prefixed, missing, duplicate, streaming, and buffered llama.cpp call IDs.
- Teardown coverage must prove intermediate handler queues are flushed and closed-client events/tool continuations cannot reactivate a stopped conversation.
- Remote-provider coverage must prove authenticated endpoint routing, no local probe/fallback, key redaction, zero OpenAI retries, dynamic context discovery, and superseded-revision transport cancellation.

- Prefer mocked HTTP and monkeypatched model setup for backend-adapter tests.
- Tests for the local Gemma/FasterQwen3TTS path must not require running local models or Docker containers.
- Keep live smoke testing documented separately from automated tests.

## Child DOX Index

- No child AGENTS.md files currently.
