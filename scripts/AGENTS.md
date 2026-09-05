# Scripts

## Purpose

- Contains local launch, benchmark, and test clients for speech-to-speech development.

## Local Contracts

- Scripts may encode machine-local paths when they are explicitly for this Windows test setup.
- Do not commit API keys; read local secrets from environment variables or optional script parameters.
- Future laptop Gemma tests use only the user's existing normal configuration launcher. Do not prescribe or substitute an ad-hoc launcher, forced 16k context, or other settings; the selected normal launcher is user-owned and must not be inferred.
- The local llama.cpp build uses canonical `--spec-type mtp:n_max=...,p_min=...`
  syntax, `--gpu-layers-draft`, and flag-only `--fit`; do not restore removed
  legacy draft, projector-offload, fit-context, or slots flags.
- Use `start_hf_realtime_frontend.ps1` for the local HF Realtime Voice UI on `http://127.0.0.1:7862`. In a worktree without a complete runtime, `local_realtime.ps1` resolves Python per component: the backend probes its actual worktree package import graph, while the frontend probes FastAPI/httpx/Uvicorn. It then uses `SPEECH_TO_SPEECH_PYTHON` or the established local shared virtual environment while still serving worktree source.
- Use `start_local_gemma_realtime_backend.ps1` for the local realtime backend on `ws://127.0.0.1:8765/v1/realtime`.
- Use `local_realtime.ps1 -Action start|stop|restart|status` for tracked background operation. It records the exact launcher or its direct Windows-venv listener child, includes the worktree app directory in the managed frontend command identity, adopts exact matching legacy repo processes, falls back to `netstat` when non-elevated PowerShell cannot query `Get-NetTCPConnection`, and never stops Gemma or FasterQwen3TTS.
- Compare recorded and live process start times as round-trip UTC instants;
  JSON reload must not reinterpret a matching listener through the local
  timezone and break safe status, stop, or restart ownership checks.
- Stop only the exact ownership-validated listener/launcher PID. If Windows
  `Stop-Process` fails internally across job boundaries, re-resolve that PID,
  treat an already-exited process as success, and use the fresh process
  object's PowerShell 5.1-compatible `Kill()` fallback; never widen the target
  to a process name or unrelated listener.
- Managed startup validates only Python and repository configuration. Missing
  Gemma or TTS services are status warnings; the launcher never inspects,
  starts, stops, or restarts model containers.
- On Windows PowerShell 5.1, Python runtime readiness is determined by the
  native process exit code. Harmless import-time stderr diagnostics must not be
  promoted into a failed preflight, and the caller's error preference must be
  restored afterward.
- Clear inherited `PYTHONPATH` only for managed child launches and restore the
  caller environment afterward so worktree imports remain deterministic.
- Before `Start-Process`, collapse duplicate case-insensitive `Path`/`PATH`
  entries inherited by Windows PowerShell into one `Path` value; duplicate
  keys must not prevent the model-independent managed backend from restarting.
- Managed runtime state and split logs live under the gitignored `.runtime/` directory.
- The managed launcher resolves Gemma identity from the repository JSON by
  default, but a persisted Remote provider endpoint/model takes precedence for
  frontend startup, dependency warnings, and `status`; it never starts or
  manages that remote model service.
- The foreground frontend launcher uses the repository JSON identity for its
  child process and restores the caller environment when the raw console exits.
- `install-rts.ps1` and `uninstall-rts.ps1` must preserve existing PowerShell
  profile text and encoding outside their managed marker block. A User PATH
  update changes the persisted User target while adding/removing only the shim
  entry in the current process PATH; it must not replace the effective process
  PATH or discard machine-level entries. The `rts.target` sidecar is UTF-8 so
  non-ASCII Windows user and folder names remain launchable.
- After a verified frontend health check, `local_realtime.ps1` prints
  `HF Realtime Voice UI: http://127.0.0.1:7862` for start/restart/status. Only
  the explicit `-Open` switch may launch that URL, and it must recheck frontend
  readiness before doing so even when only the backend component was selected.
- Launcher status must not hard-code a turn cap or compaction-disabled claim.
  The active context policy is conversation-scoped and reported by Realtime
  Diagnostics; model reachability is not authenticated generation proof.
- `probe_audio_cpp_native_pcm.py` is a candidate-only acceptance harness. It
  first reads `/health`, `/control/status`, and `/v1/models`, then can issue a
  short already-resident native Base-clone request. It never uses control
  endpoints or changes candidate model/container state, never writes PCM, and
  binds every synthesis request to that engine/instance identity, and reports
  raw byte/sample alignment plus timing; exact-once checks are limited to SSE
  event identities when an SSE server provides them. It also reports
  cancellation-follow-up recovery as compact JSON.

- `measure_candidate_tts_paths.py` defaults to read-only, non-secret identity.
  Only `--execute` starts resident-model diagnostic synthesis; it warms each
  configuration and interleaves three repetitions without saving settings or
  loading models. Refuse contended/hot GPU runs, freeze clone/profile/engine
  identity, distinguish transport cadence from engine boundaries, and verify
  request outcomes before optional explicit audio export. Store diagnostic
  files only in the ignored scratch directory; never record ordinary turns.
  Allow at most five seconds for the previous diagnostic's utilization sample
  to age out only after this diagnostic's own completed request; the first
  dispatch always uses one strict current snapshot. Never relax GPU limits,
  and stop immediately on low memory,
  excessive temperature, or missing telemetry.
  An optional finite 0--60-second inter-request cooldown begins only after a
  completed diagnostic request and before the next strict GPU/identity checks;
  it is excluded from every measured request interval.
- `validate_direct_audio_followups.py --execute` sends only explicitly selected
  local diagnostic WAV clips through the running VAD/Gemma/TTS WebSocket path.
  It never starts models, records microphones, or persists settings/secrets.
  Read credentials only from the named environment variable. Audio export is
  opt-in; synthetic clip delivery is not a microphone or manual listening gate.
- `validate_hfrt_tts_handler_path.py --execute` sends one explicitly selected,
  fixed candidate TTS phrase through the production `TTSInput`/snapshot/process
  handler path after read-only resident health and profile/revision admission.
  It excludes WebSocket, LLM, microphone, and listening validation; it never
  loads/switches models or saves settings, and raw PCM export is opt-in.
  Use only the explicit candidate endpoint without inherited TTS credentials.
  Require frozen lifecycle admission and correlated engine-EOS completion;
  emit allowlisted failure codes instead of raw provider errors. Refuse an
  existing export before dispatch and atomically publish owned temporary PCM
  only after completion proof, cleaning that temporary file on failure.

## Child DOX Index

- No child AGENTS.md files currently.
