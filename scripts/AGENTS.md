# Scripts

## Purpose

- Contains local launch, benchmark, and test clients for speech-to-speech development.

## Local Contracts

- Scripts may encode machine-local paths when they are explicitly for this Windows test setup.
- Do not commit API keys; read local secrets from environment variables or optional script parameters.
- Gemma launcher helpers should mirror `C:\llama.cpp\launch_gemma-4-12B-it-qat-MTP.ps1` unless a script comment says otherwise.
- `launch_gemma_4_12b_16k.ps1` reserves a 2560 MiB llama.cpp fit margin so an
  already-resident isolated audio.cpp TTS candidate retains synthesis headroom.
- Use `start_hf_realtime_frontend.ps1` for the local HF Realtime Voice UI on `http://127.0.0.1:7862`. In a worktree without `.venv`, it and `local_realtime.ps1` use `SPEECH_TO_SPEECH_PYTHON` or the established local shared virtual environment while still serving worktree source.
- Use `start_local_gemma_realtime_backend.ps1` for a foreground local realtime backend on `ws://127.0.0.1:8765/v1/realtime`. It launches this worktree's source only and must never inspect, start, stop, or restart Gemma or TTS services; those dependencies are validated at conversation time.
- Use `local_realtime.ps1 -Action start|stop|restart|status` for tracked background operation. It records launcher and listening child identities, adopts exact matching legacy repo processes, falls back to `netstat` when non-elevated PowerShell cannot query `Get-NetTCPConnection`, and never stops Gemma or FasterQwen3TTS.
- Use `probe_historical_audio_context.py` only as a content-free live capability
  gate. It runs two paired current/historical ABBA cycles with equal-RMS tone
  and seeded-noise WAVs in memory, calibrates the endpoint's label mapping from
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
- Managed startup validates only Python and repository configuration. Missing
  Gemma or TTS services are status warnings; the launcher never inspects,
  starts, stops, or restarts model containers.
- Clear inherited `PYTHONPATH` only for managed child launches and restore the
  caller environment afterward so worktree imports remain deterministic.
- Before `Start-Process`, collapse duplicate case-insensitive `Path`/`PATH`
  entries inherited by Windows PowerShell into one `Path` value; duplicate
  keys must not prevent the model-independent managed backend from restarting.
- Managed runtime state and split logs live under the gitignored `.runtime/` directory.

## Child DOX Index

- No child AGENTS.md files currently.
