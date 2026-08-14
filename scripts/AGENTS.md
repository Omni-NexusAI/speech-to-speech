# Scripts

## Purpose

- Contains local launch, benchmark, and test clients for speech-to-speech development.

## Local Contracts

- Published scripts must accept workstation-specific executable and model paths
  through parameters or environment variables; do not commit local path defaults.
- Do not commit API keys; read local secrets from environment variables or optional script parameters.
- `launch_gemma_4_12b_16k.ps1` reserves a 2560 MiB llama.cpp fit margin so an
  already-resident isolated audio.cpp TTS candidate retains synthesis headroom.
  It requires llama.cpp, main-model, draft-model, and mmproj paths through its
  four path parameters or the corresponding environment variables.
- Use `start_hf_realtime_frontend.ps1` for the local HF Realtime Voice UI on `http://127.0.0.1:7862`. In a worktree without `.venv`, it and `local_realtime.ps1` use `SPEECH_TO_SPEECH_PYTHON` or the established local shared virtual environment while still serving worktree source.
- Use `start_local_gemma_realtime_backend.ps1` for a foreground local realtime backend on `ws://127.0.0.1:8765/v1/realtime`. It launches this worktree's source only and must never inspect, start, stop, or restart Gemma or TTS services; those dependencies are validated at conversation time.
- Use `local_realtime.ps1 -Action start|stop|restart|status` for tracked background operation. It records launcher and listening child identities, adopts exact matching legacy repo processes, falls back to `netstat` when non-elevated PowerShell cannot query `Get-NetTCPConnection`, and never stops Gemma or FasterQwen3TTS.
- Use `probe_historical_audio_context.py` only as a content-free live capability
  gate. It runs two paired current/historical ABBA cycles with equal-RMS tone
  and seeded-noise WAVs in memory, keeps the current arm audio-only like the
  production direct request, calibrates the endpoint's label mapping from
  the current-audio trials, and passes only when historical results flip and
  remain consistent by stimulus against that mapping. It must never persist or
  print audio, prompts, transcript/response text, endpoints, models, or bearer
  keys.
- Use `probe_user_memory_fallback.py` only as a content-free live capability
  gate for the same-primary semantic-memory fallback. Each of its two equal-RMS
  tone/noise ABBA cycles makes one current-audio primary request and, only after
  allowlisted memory plus the fixed acknowledgement is returned without a
  transcript, one text-only contextual follow-up. It calibrates inverted
  primary label mappings and passes only when every follow-up preserves that
  mapping. It must never persist or print memory, audio, prompts, response text,
  endpoints, models, or bearer keys, and it must never become runtime retry
  behavior.
- Use `probe_realtime_context_tools.py` only as a content-free managed
  WebSocket release gate. It synthesizes fixed spoken scenarios in memory with
  Windows System.Speech, requires `pipeline.config.updated` before the first
  PCM append, and validates contextual counting, one exact tool call/output/
  follow-up transaction, and cancellation recovery in a single session. The
  search turn temporarily requires a tool, its post-result create disables
  recursive tools, and automatic tool choice is acknowledged again before the
  next spoken turn. It reads the remote bearer only from
  `S2S_REMOTE_MODEL_API_KEY`. Fixed non-sensitive fixture phrases remain
  source-only; no runtime-generated or live prompt, audio, transcript,
  response, tool value, endpoint/model/voice identity, or credential may be
  persisted or emitted.
- Use `probe_clarification_rate.py` only as the privacy-safe model-level
  clarification-rate gate. It runs exactly 100 normal accepted direct-audio
  turns across clear speech, hesitation, best-available installed voices,
  deterministic moderate noise, and supported-language code-switching, plus a
  separately reported meaningless cohort. Every normal turn uses a unique
  deterministic numeric challenge and expected value so retained history
  cannot answer a later cohort without attending to its current audio. It generates WAVs through Windows
  System.Speech entirely in memory using explicit UTF-8 bytes on stdin plus a
  UTF-8 PowerShell console input encoding, sends exactly one primary request per turn,
  and reuses the production direct-audio payload, control parser, semantic
  anchor, serializer, and 30-turn history contract. It reads the remote bearer
  only from `S2S_REMOTE_MODEL_API_KEY` and may emit only aggregate counts,
  rates, booleans, timings, exact attempted/completed request counts, and error
  classes. Voice discovery uses a separate bounded PowerShell process because
  failed enumeration may corrupt System.Speech process-wide; a clean synthesis
  process then uses a fresh synthesizer per scenario, falls back to its default
  voice, and reports culture/alternate coverage as false. Missing culture or
  alternate voices fail full coverage truthfully; prompts, audio, transcripts, responses,
  endpoints, models, voice identities, and credentials never reach public
  stdout/stderr or temporary files. This controlled gate is not a WER, room/AEC, VAD, WebSocket,
  tool-order, or open-ended conversation benchmark. Resolve model routing only
  from the exact managed loopback `/api/ui-settings` response, reject redirects,
  and validate that handoff before reading or using the remote credential.
- Managed startup validates only Python and repository configuration. Missing
  Gemma or TTS services are status warnings; the launcher never inspects,
  starts, stops, or restarts model containers.
- Clear inherited `PYTHONPATH` only for managed child launches and restore the
  caller environment afterward so worktree imports remain deterministic.
- Before `Start-Process`, collapse duplicate case-insensitive `Path`/`PATH`
  entries inherited by Windows PowerShell into one `Path` value; duplicate
  keys must not prevent the model-independent managed backend from restarting.
- Managed runtime state and split logs live under the gitignored `.runtime/` directory.
- The clarification probe shares the production direct-audio sampling constants, keeps all utterances and responses process-local, and fails the 100-turn normal-conversation gate above one clarification, any recognition/input-process commentary, any repeated clarification wording, or incomplete safe multilingual voice coverage. Recognition commentary takes precedence over an expected numeric answer. Pure English/Spanish/German/Japanese turns and mixed-language turns remain distinct cohorts; meaningless input is reported separately.
- The installed command reads only `rts.target`. Before repointing it, preserve the previous launcher in the passive local `rts.target.rollback` record; installers and lifecycle commands must never execute or overwrite that record implicitly. Restore it only through an explicit user-authorized target installation after confirming the pool is idle.
- Before a managed start, hash only the checkout's runtime source files and capture the Git revision, dirty state, and browser asset generations. Inject that immutable content-free identity into both child processes, reject healthy listeners whose identity does not match the intended checkout, and make `status` warn prominently when running code is stale relative to current source.

## Child DOX Index

- No child AGENTS.md files currently.
