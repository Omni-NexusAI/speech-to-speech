# Scripts

## Purpose

- Contains local launch, benchmark, and test clients for speech-to-speech development.

## Local Contracts

- Scripts may encode machine-local paths when they are explicitly for this Windows test setup.
- Do not commit API keys; read local secrets from environment variables or optional script parameters.
- Gemma launcher helpers should mirror `C:\llama.cpp\launch_gemma-4-12B-it-qat-MTP.ps1` unless a script comment says otherwise.
- Use `start_hf_realtime_frontend.ps1` for the local HF Realtime Voice UI on `http://127.0.0.1:7862`.
- Use `start_local_gemma_realtime_backend.ps1` for the local realtime backend on `ws://127.0.0.1:8765/v1/realtime`.
- Use `local_realtime.ps1 -Action start|stop|restart|status` for tracked background operation. It records launcher and listening child identities, adopts exact matching legacy repo processes, and never stops Gemma or FasterQwen3TTS.
- Managed runtime state and split logs live under the gitignored `.runtime/` directory.

## Child DOX Index

- No child AGENTS.md files currently.
