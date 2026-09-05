# audio.cpp Candidate

## Purpose

- Holds the one opt-in CUDA audio.cpp Qwen3-TTS candidate container: engine,
  copied Voice Studio, and staged BF16 model assets.

## Local Contracts

- Never share port `8881`, container lifecycle, image, model assets, or profiles with
  `qwen3-tts-faster`; never manage Groxaxo on `8882`.
- Keep `qwen3-tts-base-f16.json` on original BF16 0.6B and 1.7B checkpoints.
  Do not switch this initial candidate to Q8 or another reduced-weight variant.
- Native incremental PCM is staged as an opt-in candidate mode for the original
  raw 0.6B and 1.7B Base checkpoints and is not promoted until live proof. Both models require independent multi-chunk,
  prompt-cancellation, stale-tail, and clean-recovery validation while remaining
  mutually exclusive under the single-resident supervisor. Progressive buffered
  phrase PCM and complete WAV remain selectable rollback/fallback paths; this
  candidate does not replace or change the Faster default.
- Claim native incremental PCM only when live status advertises it and the
  supervisor consumes an engine-owned `speech.decode_mode` SSE event from the
  same upstream iterator before committing downstream headers or relaying PCM.
  A separate direct transport validation must observe at least two nonempty PCM
  chunks. The supervisor may return
  `X-TTS-Streaming-Mode: native-incremental-pcm` only after that proof; a mode
  mismatch, PCM/done before proof, or EOF fails closed. Never synthesize a
  native label from request metadata. Automated transport validation does not prove
  speaker-loopback suppression or human double-talk preservation; those remain
  physical microphone/output-device tests owned by the user.
- audio.cpp lazy loading retains models. Use `supervisor.py` so only one model
  process/config is active; model switches must stop the old child first and
  must reject GPU-busy or insufficient-VRAM conditions before a load.
- Keep load and synthesis GPU guards distinct: model loads require full
  checkpoint headroom, while synthesis checks post-load free VRAM and
  utilization without counting the resident model against itself.
- The Studio may persist candidate-only GPU admission mode (`enforced`,
  `custom`, or explicit `disabled`) under its private `/voices` volume. Custom
  values can reduce coexistence reserves; disabled mode only bypasses the
  preflight and must visibly warn that CUDA OOM remains possible.
- Keep `qwen3_tts.mem_saver=true` for buffered-only deployments and set it to
  `false` only while native incremental PCM is enabled. Expose named lifecycle events
  through both the supervisor log and model status. An unexpected child exit
  is `evicted/error`, not a normal unload.
- Native development moves the one-time lazy decoder cost into an explicit,
  private load warmup. Do not report the
  model `loaded` until the discarded PCM stream completes or a truthful
  skipped/failed result is recorded. Bound warmup time, release the child on
  cancellation, allow same-model retry after skip/failure, suppress warmup from
  normal generation metrics, and reject queued requests whose engine epoch is
  stale after a switch.
- The Docker engine stage has no visible host GPU. Its first CMake configure
  must receive `CUDAARCHS=86` for this RTX 3080 Laptop candidate rather than
  inheriting nvcc's fallback architecture. The pinned engine exposes no
  supported CUDA-graph runtime toggle, so status must report its graph mode as
  engine-default/uncontrolled and must not advertise a cosmetic A/B switch.
- Native clone caches may reuse immutable encoded reference conditioning and
  speaker embeddings. The opt-in `qwen3_tts.talker_prefix_cache_slots` prototype
  may additionally retain the exact immutable leading clone, language, control,
  and reference talker-KV state, plus bounded host-side static prompt material
  keyed by the paired transcript/reference identity. It must still replay every
  phrase-dependent causal suffix; never cache a whole long-reference KV state
  merely to reduce latency. The engine defaults to zero slots and
  `mem_saver` always disables it; the explicit native-development override may
  configure a bounded nonzero slot count for validation. Every request restores
  only that prefix, replays all phrase-dependent suffix rows, and keeps generated
  speech state request-local. Native first-PCM latency remains a measured gate.
- `/health`, `/control/status`, and candidate inventory expose a new random
  `supervisorInstanceId` together with `engineEpoch`. Realtime phrase requests
  that carry expected lifecycle values require the complete pair and reject 409
  on either mismatch; strip both private admission fields before every engine
  forward. The nonce makes a supervisor restart stale even if its numeric epoch
  resets to the same value.
- Full-ICL reference audio and its transcript are an indivisible conditioning
  pair. A reference-duration limit may apply only to an explicitly stored
  cropped audio excerpt with its exact matching transcript; otherwise retain
  the complete pair and report the requested and effective duration truthfully.
- Candidate clone inventory exposes a content revision and length-delimited
  SHA-256 content hash for the actual full-ICL reference/excerpts. A response
  may submit its frozen clone snapshot with exact encoded reference material;
  validate its ID/revision/hash and use those bytes instead of reopening the
  mutable profile slot, then strip the private snapshot before the engine call.
- Both raw model decoders declare a 25-frame causal window (2 seconds at one
  80 ms codec frame). Built-in native profiles retain those 25 frames; a wider
  value only adds decode latency, while smaller Advanced overrides are
  experimental and must warn about progressive quality loss rather than
  masquerading as a safe latency optimization.
- Normalize Gradio's `seed=-1` random-seed sentinel by omitting `seed` at the
  audio.cpp boundary; audio.cpp accepts only unsigned explicit seed values.
- The actual editable Studio is the copied Gradio app at `/voice-studio/`. Keep the root HF Realtime
  page separate for later streaming/integration testing; never relabel it as
  Voice Studio.
- `gradio_voice_studio.py` is the recovered Faster-side copy, adapted only at
  its candidate boundary: preserve its orange custom Studio surface and full
  clone/library/playground layout. Keep model controls visible; Base model
  load/switch/unload are routed through the single-resident supervisor, while
  unsupported candidate modes remain visibly disabled with an explanation.
- Seed Playground profile dropdown choices from the private persistent library
  at app construction and synchronize them on Library load/refresh; Gradio
  validates callback values against server-side choices before handlers run.
- The candidate Dockerfile uses `VOICE_STUDIO_SOURCE_REV` immediately before
  the copied Studio source so a source-only Studio repair refreshes final image
  layers without rebuilding CUDA or re-staging model checkpoints.
- The Playground live surface uses same-origin candidate proxies for microphone,
  VAD, cancellation, llama.cpp turn flow, progressive phrase PCM, and buffered
  playback. Keep its phrase diagnostics/tuning local to the Studio and do not
  use it for native first-chunk benchmarks.
- A frozen Studio response reads its lifecycle, clone snapshot, tuning, and any
  terminal audio outcome through explicit same-origin no-store routes only.
  Snapshot reads are GET-only where applicable; a native stream failure may use
  its exact opaque request ID to report a limited/error/cancelled outcome, then
  cancels that response's remaining phrases and playback without switching TTS
  provider or delivery mode.
- Native Playground playback uses one turn-scoped AudioWorklet queue, releases
  after the selected first block only (`first_block_frames * 80 ms`), and uses
  steady-block cadence only for bounded underrun re-priming. It remains
  continuous across phrases and clears queued/stale PCM on cancellation.
  Selected profile look-ahead and
  flush settings own phrase dispatch; legacy browser-local phrase values may
  not silently override them or split incomplete text mid-word.
- Buffered phrase PCM is the stable Playground default even when the candidate
  native override is enabled. Native incremental PCM remains an explicit
  experimental selection until separate live latency, RTF, continuity,
  cancellation, and listening gates pass for the resident raw Base model.
- Buffered phrase PCM keeps the selected profile's sampler and phrase-queue
  policy but runs one completed `offline_full` engine decode per phrase. It
  must report `buffered-fallback` as its delivery mode and result-derived
  `offline-full-decoder` proof; never run the native block loop and hide it
  behind a completed response.
- Keep the live widget's LLM connection separate from Diagnostics & tuning.
  Voice Studio and HF Realtime expose the candidate profile selector,
  native-streaming status and controls, and delivery/latency metrics against
  the fixed model-native PCM16/24 kHz output contract. Disable native-only block
  controls when native streaming is unavailable and report native, buffered fallback, or
  non-streaming delivery truthfully; never imply an adjustable model bitrate.
- Persist the Studio's non-secret LLM, microphone, VAD, phrase, input-format,
  and input-rate settings atomically in the private `/voices` volume. API keys
  remain browser-only; normalize a bare llama.cpp server URL to
  `/v1/chat/completions` before the container proxy relays a turn.
- Streaming microphone turns use AudioWorklet mono PCM capture. Apply pre-roll
  and VAD boundaries to PCM buffers and emit a complete PCM16 RIFF/WAV for each
  accepted turn. WAV is the lossless default; optional MP3 320 kbps conversion
  happens server-side and llama.cpp metadata must match the actual bytes.
- Non-streaming always uses one complete native WAV master. WAV/PCM return that
  master or its raw frames; other formats are one final high-quality conversion
  with clear container-versus-codec status, never multiple user outputs.
- Live native and buffered phrase requests share candidate-owned, request-local
  generated-codec-frame limits, distinct from wall-clock deadlines. A cap or
  error is never successful EOS or a complete exported clip. Engine cycle
  telemetry is observation-only, not an acoustic degeneration classifier.
  Keep the content-free outcome lookup bounded (256 requests, ten minutes),
  no-store, and keyed by server-generated opaque IDs. It explains failures
  after raw PCM headers without putting metadata into PCM or retrying synthesis.
  Offline Full WAV keeps its existing engine capacity and quality policy.
- Non-streaming Full WAV is an independent offline-full-decoder Quality policy:
  streaming block, context, look-ahead, flush, and session overrides are
  inactive and must not leak between Full WAV and streaming modes.
- Require result-derived `X-AudioCPP-Qwen3-Decode-Mode: offline-full-decoder`
  proof for every Voice Studio Full Quality request. Missing or contradictory
  proof is a hard upstream error; never silently accept an older streaming
  decoder for a non-streaming request.
- Full Quality creates one 24 kHz PCM16 WAV master, then exposes WAV, raw PCM,
  FLAC, MP3, AAC, and Opus as final exports. Keep explicit uint32 seeds active;
  blank, null, and Gradio `-1` request engine randomness.
- Tuning selections are canonical and independent for `voice-studio` and
  `realtime`. Temporary overrides are page-session-only, built-ins are
  immutable, custom-profile imports are conflict-safe, and Save As creates a
  new revisioned custom profile rather than overwriting an existing ID.
- For repaired live rollouts, build `Dockerfile.overlay` explicitly: it compiles
  the pinned patched engine and copies the new binary plus control/UI sources
  onto the retained model-bearing base. Recreate only Compose project
  `audio-cpp` with `--no-build`; select current or rollback through
  `AUDIO_CPP_IMAGE`, preserve the named `/voices` volume, and never use
  `down -v`.
- A source-only live recovery may use `Dockerfile.runtime-overlay` only when its
  base image already carries the validated native engine hash asserted by that
  Dockerfile. It must also verify both embedded BF16 model trees, replace only
  supervisor/Studio/HF runtime sources, and retain `Dockerfile.overlay` as the
  clean pinned-engine rebuild path.
