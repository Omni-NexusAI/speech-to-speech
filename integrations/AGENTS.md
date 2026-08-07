# External Integration Patches

## Purpose

- Stores reproducible source patches for local services that are rebuilt outside this Git repository.

## Contracts

- `qwen3-tts-faster-language.patch` documents the only required Faster server divergence: optional request language, advertised support, and per-request clone language conditioning. Preserve explicit `Auto`; clone-reference metadata must never choose the conversational response language.
- Keep the external service source, rebuilt image behavior, `/health` capabilities, and this patch aligned.
- Do not add copied model files, Docker layers, generated audio, or voice-library data here.
