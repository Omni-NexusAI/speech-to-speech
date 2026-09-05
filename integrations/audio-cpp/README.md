# audio.cpp Qwen3-TTS Candidate

This is one self-contained CUDA candidate container for audio.cpp plus the
completed copied Voice Studio. The image includes unquantized Qwen3-TTS 0.6B
Base and 1.7B Base checkpoints. It does not change, mount, rebuild, or manage
the `qwen3-tts-faster` container.

## Boundary

- The endpoint is `http://127.0.0.1:8890/v1`, separate from Faster on `8881`
  and the user-managed Groxaxo candidate on `8882`.
- Both original BF16 Qwen checkpoints are staged in the candidate image during
  build. The only named volume is private persistent Voice Studio clone data.
- The Compose profile is opt-in. This repository never starts it automatically.
- The regular candidate file pins `AUDIO_CPP_NATIVE_INCREMENTAL_PCM=false`.
  Native development is enabled only by adding `compose.native.yml`; buffered
  phrase PCM and complete WAV remain available in that same candidate. The
  override only changes the runtime flag on the exact image selected through
  `AUDIO_CPP_IMAGE`. Record and retain one explicit rollback tag before a live
  candidate recreation.
- Docker engine builds explicitly set `CUDAARCHS=86` before their first CMake
  configure.  Builder stages do not see the host GPU and otherwise inherit an
  nvcc default (previously observed as sm_52); `86` is the RTX 3080 Laptop
  target used by this isolated candidate.
- The pinned engine currently exposes no supported CUDA-graph runtime switch;
  it runs engine-default graph behavior and reports that control as unavailable
  rather than pretending an environment variable changes it. Explicit **Load**
  includes one private, discarded native warmup using a live paired clone before
  the model reports `loaded`. Status
  exposes `warming`, elapsed time, profile ID, timeout, and retryable
  skipped/failed results; the warmup never appears as a user generation metric.
- The container supervisor keeps exactly one model process/configuration active.
  Switching models terminates the old process before starting the new one; both
  checkpoint files remain in the same image.
- Before a switch or synthesis, the supervisor checks free VRAM and GPU
  utilization. A busy GPU produces a clear `GPU busy/insufficient VRAM` result
  without attempting the operation. Load and synthesis reserves are reported
  separately, and named lifecycle events are available through the model-status
  endpoint and container logs.
- The Studio exposes a candidate-only **GPU admission guard**, defaulting to
  **Disabled** so the expected chat LLM plus one audio.cpp model can coexist.
  Enforced and Custom remain explicit opt-ins. For 0.6B,
  Enforced admission uses the observed ~5,630 MiB load delta rounded up to a
  6,000 MiB measured model-residency reserve plus the unchanged 2,048 MiB synthesis floor:
  8,048 MiB free before load at 85% maximum utilization. For 1.7B, Enforced
  keeps the existing 10,500 MiB **total** threshold until a real residency
  delta is measured; it does not invent and add a second 2,048 MiB reserve.
  Synthesis retains its separate 2,048 MiB / 95% check.
  Custom mode uses its persisted load value as an explicit absolute threshold;
  Disabled mode bypasses preflight only and warns that CUDA OOM remains possible.
- Do not substitute a Q8 or other reduced-weight checkpoint for this initial
  candidate.

## Bring-up and validation

1. A first bootstrap, when no validated local base image exists yet, may use
   the full Dockerfile. It downloads the two original BF16 Base models and
   their tokenizer sidecars, compiles the pinned engine, and creates the
   self-contained candidate image:

   ```powershell
   docker build -f integrations/audio-cpp/Dockerfile -t local/audio-cpp-qwen3-tts-voice-studio:native-development .
   ```

2. A repaired live rollout must use `Dockerfile.overlay` explicitly. It
   compiles the repaired pinned audio.cpp binary in a disposable build stage,
   then copies that binary plus the current supervisor and UI sources onto the
   validated model-bearing base image. It does not re-download either BF16
   checkpoint:

   ```powershell
   $candidateContainer = 'audio-cpp-qwen3-tts-voice-studio'
   $imageRepository = 'local/audio-cpp-qwen3-tts-voice-studio'
   $currentImageId = docker inspect --format '{{.Image}}' $candidateContainer
   if (-not $currentImageId) { throw 'Candidate container has no deployed image ID.' }
   $rollbackTag = "${imageRepository}:rollback-$(Get-Date -Format yyyyMMdd-HHmmss)"
   docker image tag $currentImageId $rollbackTag
   $rollbackImageId = docker image inspect --format '{{.Id}}' $rollbackTag
   if ($rollbackImageId -ne $currentImageId) { throw 'Rollback tag does not match the deployed candidate image.' }

   $releaseTag = "${imageRepository}:stream-reliability-$(Get-Date -Format yyyyMMdd-HHmmss)"
   docker build -f integrations/audio-cpp/Dockerfile.overlay `
     --build-arg AUDIO_CPP_BASE_IMAGE=$rollbackTag `
     --build-arg AUDIO_CPP_BASE_IMAGE_ID=$currentImageId `
     -t $releaseTag .
   $env:AUDIO_CPP_IMAGE = $releaseTag
   docker compose -p audio-cpp -f integrations/audio-cpp/compose.candidate.yml -f integrations/audio-cpp/compose.native.yml --profile audio-cpp-candidate up -d --no-build --no-deps --force-recreate audio-cpp-candidate
   Remove-Item Env:AUDIO_CPP_IMAGE
   ```

   Do not reuse an older tag by name. The sequence above first resolves the
   exact image ID deployed by the candidate container, creates and verifies a
   new rollback tag for that ID, passes both into the build, and records the
   parent ID in the final image's `io.omninexus.rollback.base` label.

   When the running model-bearing image already contains the exact validated
   native engine hash asserted by `Dockerfile.runtime-overlay`, a supervisor or
   Studio-only repair may use that narrower Dockerfile instead. It verifies the
   inherited engine binary and both BF16 model trees, then overlays only the
   current runtime sources; it is not a substitute for the pinned-engine build
   when C++ changes are involved:

   ```powershell
    docker build -f integrations/audio-cpp/Dockerfile.runtime-overlay `
      --build-arg AUDIO_CPP_BASE_IMAGE=local/audio-cpp-qwen3-tts-voice-studio:<rollback-tag> `
      --build-arg AUDIO_CPP_BASE_IMAGE_ID=sha256:<verified-deployed-image-id> `
      -t local/audio-cpp-qwen3-tts-voice-studio:<runtime-overlay-tag> .
   ```

   Keep project name `audio-cpp`; it preserves the existing container and
   private-volume identity. Never run `docker compose down -v` during rollout
   or rollback because `-v` would remove the persisted clone/profile library.
   `AUDIO_CPP_IMAGE` selects the exact existing image consumed by Compose, so a
   rollback does not trigger a build:

   ```powershell
    $env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:<rollback-tag>'
   docker compose -p audio-cpp -f integrations/audio-cpp/compose.candidate.yml --profile audio-cpp-candidate up -d --no-build --no-deps --force-recreate audio-cpp-candidate
   Remove-Item Env:AUDIO_CPP_IMAGE
   ```

3. Open `http://127.0.0.1:8891/voice-studio/` for the candidate's copied
   Gradio Voice
   Studio. It is the orange copied Gradio surface with visible candidate model
   controls. The root page remains the separate HF Realtime integration test UI.
   Confirm the
   private API endpoint `http://127.0.0.1:8890` reports both
   `qwen3-tts-0.6b-base-bf16` and `qwen3-tts-1.7b-base-bf16`.
4. Select a candidate model in **Settings & candidate model controls**, click
   **Load selected model**, and use the Base clone or Playground to run a
   buffered speech probe. **Unload model** terminates the audio.cpp child and
   leaves no model resident.

The image uses a multi-stage CUDA build: compilation happens in a development
stage, while the final image contains only the server binary, required runtime
assets, Python/Gradio/FFmpeg runtime, and model files. It excludes compiler
tooling, CUDA development packages, Nsight tooling, build trees, source/Git
metadata, and tests. Both model directories remain original BF16 Hugging Face
checkpoints; their identical speech-tokenizer weights are stored once through
internal symlinks. This reduces storage without changing checkpoint weights or
runtime math.

The Studio stores Base clone metadata and WAVs in its private volume. Gradio
and the API rescan the same live directories: hidden/cache entries, missing
reference audio, malformed metadata, and non-Base entries are excluded. Safe
legacy Base metadata is normalized atomically without changing its directory
ID or reference recording. Its
candidate adapter resolves `clone:<profile_id>` to that private audio and sends
audio.cpp the documented `voice_ref` and `reference_text` fields. The live
adapter treats those fields as one conditioning pair: it never shortens a WAV
while retaining the full transcript. `max_reference_seconds` can select a
shorter reference only when profile metadata stores that excerpt audio with its
exact excerpt transcript; otherwise the complete pair is used and response
headers report that the requested limit was not applied.
The live
Playground surface can test microphone selection, browser VAD, pre-roll,
cancellation, llama.cpp turn flow, and progressive phrase PCM playback through
same-origin candidate proxies. The normal candidate stays buffered: each phrase
uses one completed `offline_full` decoder pass before playback and is not labeled
native streaming. It retains the selected profile's sampler and phrase-queue
policy while requiring result-derived `offline-full-decoder` proof, so a native
block loop cannot be silently buffered behind the fallback label. The development
override enables the pinned eight-file engine patch, keeps one model resident,
and makes **Native incremental PCM (experimental)** selectable through
`/v1/audio/speech` with `stream:true` and `response_format:"pcm"`. Buffered
phrase PCM remains the Playground default until native passes the live timing,
continuity, cancellation, and listening gates; Full WAV stays selectable as the
offline reference path. Stop/new-turn cancellation closes the browser proxy and
supervisor streams so the engine resets without stale tail playback. Native
promotion requires independent multi-chunk, cancellation/recovery, timing, and
single-resident GPU validation for each raw model. The default clone cache
reuses encoded reference conditioning and speaker embeddings. The engine's
talker-prefix cache defaults to zero slots; the explicit native-development
override configures four experimental slots for the exact immutable leading
clone, language, control, and reference KV state. Each request restores that
prefix, replays every phrase-dependent suffix row, and keeps generated speech
state request-local. Native first-PCM latency therefore remains an
explicit performance gate rather than a solved capability. Physical
speaker-loopback AEC3 suppression and human double-talk preservation remain
user-owned acoustic validation.
The streaming Playground starts by default for evaluation. Non-streaming uses
one complete offline decode and keeps audio.cpp's PCM16 WAV as its full-quality
master. It always uses the shipped Quality sampler policy independently of
streaming selections or session overrides; first/steady/context and phrase
queue controls are disabled while Full WAV is selected. The result pane labels
this path `offline-full-decoder`. The supervisor requests
`qwen3_tts.decode_mode=offline_full`, rejects a contradictory result-derived
decoder-mode header, and never retries through incremental generation. WAV is
returned unchanged from that completed master, raw PCM is extracted directly,
 FLAC is lossless, MP3 and AAC are post-encoded at 320 kbps, and Opus uses the
 codec's maximum supported 256 kbps without resampling or downmixing.
 `/v1/audio/speech` with `stream:false` plus PCM remains
the separately labeled buffered fallback and uses the completed offline decoder;
`/v1/audio/voice-clone` is always the Full Quality offline policy regardless of
export format.
Native and buffered streaming use the selected Voice Studio profile directly.
Punctuation-complete text dispatches immediately; incomplete text uses the
profile look-ahead and idle-flush values, cuts only on safe word/clause
boundaries, and has a profile-derived hard cap. Browser-local legacy phrase
values never override the selected profile. One turn-scoped AudioWorklet queue
spans every phrase. Native playback releases after the selected first block
only, then uses measured steady-block cadence solely for bounded underrun
re-priming; it reports underruns/drain and rejects late PCM after cancellation.
Non-secret Studio controls (LLM endpoint/model/prompt, selected microphone,
VAD, and selected tuning identity) are atomically retained in the private volume, so they
survive candidate recreation. API keys intentionally remain browser-only. A
bare local llama.cpp server address such as `http://127.0.0.1:8818` is accepted
and normalized to `/v1/chat/completions`; a full OpenAI-compatible endpoint is
also accepted. The Streaming Playground captures mono Float32 PCM through an
AudioWorklet, applies VAD and pre-roll over PCM buffers, and creates a complete
PCM16 RIFF/WAV for every accepted turn. Lossless WAV at 16 kHz is the default;
24 kHz and 48 kHz encodes and an explicit server-converted MP3 320 kbps A/B
transport are available. llama.cpp input metadata always matches the bytes
sent. A candidate restart always releases the active model process, so
the Studio must explicitly load its selected model before its next generation.
Quality, Balanced, Low Latency, and custom tuning definitions are shared, but
Voice Studio and Realtime persist independent active selections. Balanced is
the fallback for either surface when it has no valid saved selection.
Built-ins are immutable; editing starts by cloning one. Realtime Save As sends
the complete effective safe schema and can atomically select the new profile
only for the Realtime scope, without changing Voice Studio's selection. Custom
updates require their current revision and reject stale writes.
All built-ins retain the model-required 25 decoder-context frames (2 seconds).
A smaller
custom or temporary value is an explicitly warned Advanced experiment because
reduced causal history can progressively degrade a long native stream. The
 pinned native engine patch uses the same 25-frame causal-context fallback for direct internal
requests, so bypassing profile resolution cannot silently restore the older
quality-degrading context.
Seed accepts null or an unsigned 32-bit integer: blank and Gradio's `-1`
sentinel omit the field for engine randomness, while explicit values are
forwarded. Temperature must be greater than zero. x-vector-only clone mode and
crossfade remain inactive. Full ICL clone reference audio plus its transcript
remains required.

## Diagnostic and failure contracts

The mounted Studio freezes each response through same-origin, uncached tuning,
model-state and clone-snapshot routes. Model/clone reads are GET-only and do not
expose lifecycle mutations. Its explicit local WAV diagnostic picker exercises
the normal LLM/snapshot/TTS path without recording a microphone; physical VAD,
voice identity and prosody remain separate listening gates.

Native experimental and buffered live phrases share request-local codec-frame
bounds and a separate wall-clock deadline. The engine reports generated frames,
capacity, EOS/limit termination and stage timings; cycle observation is not an
automatic acoustic-degeneration detector. A valid EOS exactly at the frame cap
is complete. Limit/error termination is not a successful clip and cannot be
automatically retried through another delivery mode.

Every candidate speech response carries an opaque `X-TTS-Request-Id`. Clients
check `/v1/audio/outcomes/{id}` after PCM delivery so a late failure can clear
only the affected response queue. Outcomes contain no text or audio, expire in
ten minutes, and retain at most 256 entries. Native response headers remain held
until the same iterator proves decoder mode and two PCM events; diagnostics
report that validation hold separately from engine first-PCM latency.

Full Quality WAV keeps its independent offline policy. Temporary tuning never
edits shared saved profiles, and Voice Studio/Realtime retain separate selected
profiles. Explicit benchmark clips and reports stay local and outside version
control; HTTP completion alone does not certify microphone-to-speaker behavior.
