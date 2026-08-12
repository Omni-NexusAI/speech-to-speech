# Context glossary

Canonical terms for the current local and hosted Realtime browser surface. This
file defines vocabulary; implementation contracts live in `AGENTS.md` and
`DESIGN.md`.

## Local Realtime UI

The browser application served by the managed frontend, normally at
`http://127.0.0.1:7862`. It is a client of the repository's Realtime WebSocket,
not a second speech backend.

## Pipeline

The current local turn path:

`microphone → browser capture/AEC → VAD → Gemma direct audio → optional tools → selected Qwen3-TTS provider → playback worklet`

- **VAD** decides when an utterance is accepted. It is the only audio-admission
  boundary for the direct-audio path.
- **Gemma direct audio** sends the accepted utterance to the conversation's
  selected local or remote OpenAI-compatible audio model without requiring an
  STT transcript first.
- **TTS** speaks stable assistant phrases through FasterQwen3TTS, the
  user-managed Groxaxo provider, or the isolated audio.cpp candidate.

## Accepted user turn

One VAD-completed utterance admitted to Gemma. A cumulative VAD revision may
replace the same logical turn, but it must not create another semantic history
entry or another final response.

## Semantic user anchor

The single history item reserved for an accepted user turn before assistant or
tool state. Validated transcript text has first precedence, bounded validated
same-response `USER_MEMORY` has second precedence, and historical `input_audio`
is retained when neither text field is safe. The browser's `[User audio]` label
is a separate display-only representation.

## Configuration acknowledgement

The matching `pipeline.config.updated` event that commits model provider, TTS
provider, clone, and tuning for a conversation. Microphone upload waits for the
initial acknowledgement. Later changes apply only after acknowledgement and do
not mutate already queued speech.

## Model provider

The conversation-scoped Gemma endpoint. **Local** uses the configured local
llama.cpp service. **Remote** uses one explicit OpenAI-compatible endpoint and
never silently falls back to Local.

## TTS provider

The conversation-scoped speech service, independent from the model provider:

- **FasterQwen3TTS** is the stable clone-only default.
- **Groxaxo** is user-managed and selectable only while compatible and ready.
- **audio.cpp** is the opt-in candidate with private profiles and experimental
  native PCM.

## Base clone profile

A voice-cloning identity backed by a reference WAV, exact paired transcript,
and Base task metadata. Inventories are provider-scoped. A profile visible in
one provider is not borrowed by another provider.

## Native incremental PCM

audio.cpp delivery that produces multiple verified PCM chunks before the
synthesis request completes. A capability flag alone is not proof. A request
that lacks the exact native header or enough chunks remains **buffered
fallback**.

## Adaptive safe-start

The browser's native audio.cpp playback policy. Cold playback uses the full
first-plus-steady block ceiling; learned warm playback uses content-free logical
decoder-block arrival evidence. It is unrelated to the **Adaptive** echo mode.

## Echo modes

- **Native** uses browser AEC and is the default.
- **Adaptive** uses the authenticated AEC3 worklet when available and otherwise
  resolves to Native.
- **Strict** withholds uncertain capture through the echo tail instead of
  manufacturing silence.

The echo reference is exact scheduled playback PCM, never a clone reference
recording.

## Tool

A model-requested browser function. Web search and camera snapshot are the
current tools. Arguments are schema-validated before execution, output is
acknowledged before one post-tool response is created, and the tool call/result
remain ordered inside the originating semantic turn.

## Camera snapshot

One point-in-time frame captured only by a real tool call. Preview is not a
snapshot, and a question about the current view or a change requires a fresh
call. Every invocation receives a distinct durable visible card.

## Web search

A browser tool backed by the same-origin Serper proxy. Its canonical request is
`query` plus optional `mode` (`auto`, `web`, or `news`) and `freshness` (`none`,
`day`, `week`, `month`, or `year`). Output is versioned structured JSON that
distinguishes requested/effective mode, retrieval time, recency filtering, and
the one auto-news-to-web zero-result fallback; explicit news never falls back. Available result dates and sources remain attached to their
results; retrieval time does not imply that an undated result was published
today. One accepted turn may make at most one distinct narrower refinement. The
first result follow-up exposes only search; terminal and non-search follow-ups
disable tools for that response.

## Hosted session

The optional Hugging Face deployment path in which the UI server proxies a
load-balancer `/session` request and returns a signed Realtime WebSocket URL.
This is separate from the managed local direct WebSocket at `8765`.

## Diagnostics

Content-free runtime, queue, latency, context-size, provider, language, and echo
telemetry. Diagnostics may report counts, timings, states, and bounded reasons;
they never log prompt, transcript, semantic memory, tool values, response text,
audio, credentials, or raw adaptive-playback identities.
