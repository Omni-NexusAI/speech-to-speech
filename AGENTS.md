# Repository Instructions

## DOX Framework

- DOX is installed here as a shallow AGENTS.md hierarchy.
- AGENTS.md files are binding work contracts for their subtrees.
- Before editing, read this root AGENTS.md and every child AGENTS.md on the path to the files you will touch.
- After meaningful edits, update the nearest AGENTS.md and any affected parent/child indexes when structure, contracts, workflows, or durable preferences changed.
- Keep docs concise and operational; document stable contracts rather than change history.

## Local Voice-Agent Testing Contract

- Local realtime defaults retain 30 complete turns with compaction disabled; context is session-scoped and resets with the WebSocket/backend.
- Every completed direct-audio turn produces a validated final user transcript for history and context. The live-transcription toggle controls only the temporary speaking bubble.
- Context diagnostics report retained history tokens against the live llama.cpp context window; turn retention remains a separate 30-turn policy.
- `scripts/local_realtime.ps1` is the managed background entry point; foreground launchers remain available for raw-console debugging.

- This fork is used for local speech-to-speech testing with the OpenAI Realtime-compatible server preserved.
- Keep upstream pipeline behavior intact unless a change is required for the local direct-audio Gemma path.
- The local fast-test path intentionally skips Parakeet/transcription with `--stt gemma-audio` and sends completed audio turns directly to the local Gemma audio model.
- Speech output defaults to the existing Dockerized FasterQwen3TTS-compatible OpenAI API on `8881`; the optional Groxaxo candidate on `8882` is user-managed and must never be started, stopped, or switched by this repo.
- Local inference only for the custom test path: no cloud fallback should be introduced.
- For Gemma 4 12B tests, use the settings from `C:\llama.cpp\launch_gemma-4-12B-it-qat-MTP.ps1` with context reduced to 16k; this repo provides `scripts\launch_gemma_4_12b_16k.ps1` for that.
- The target TTS service is the stable Docker container name `qwen3-tts-faster` on `http://127.0.0.1:8881/v1`; container IDs are transient after authorized rebuilds and must not be used by lifecycle scripts.
- Qwen3 clone output must use the `1.7B-Base` backend model by default; the `qwen3-tts-faster` API exposes native incremental PCM16 clone streaming while preserving buffered formats for non-streaming requests.
- Faster clone requests carry an explicit assistant language when known; keep the reproducible local server patch under `integrations/` in sync with the rebuilt image.
- The realtime UI should expose saved Voice Studio `Base` clone profiles only, with `clone:16d9bb336799` (`J.A.R.V.I.S`) as the default.
- Do not test Gemma/llama.cpp while the local Gemma server is being rebuilt.

## Child DOX Index

- `src/AGENTS.md` covers package code and runtime pipeline contracts.
- `scripts/AGENTS.md` covers local launch/test helper scripts.
- `tests/AGENTS.md` covers test expectations.
- `web/AGENTS.md` covers the vendored HF Realtime Voice browser UI.
- `integrations/AGENTS.md` covers reproducible patches for external local services.

## Release Rules

- Never include `codex` in branch names or pull request titles.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, or generated wheel/sdist files.

## Publishing to PyPI

PyPI publishing is handled by GitHub Actions in `.github/workflows/publish.yml`. The workflow runs on pushed tags that match `v*`, builds the package with `uv build`, checks the artifacts with `twine check --strict`, and publishes through the configured `pypi` environment.

To prepare a release:

1. Confirm the intended version is not already published on PyPI.
2. Bump `version` in `pyproject.toml`.
3. Bump `__version__` in `src/speech_to_speech/__init__.py`.
4. Open and merge a pull request with only the release preparation changes.

To publish after the release PR is merged:

1. Update `main` locally: `git checkout main && git pull origin main`.
2. Create an annotated tag for the version: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
3. Push the tag: `git push origin vX.Y.Z`.
4. Watch the `Publish` GitHub Actions workflow complete successfully.
5. Verify the new version appears at `https://pypi.org/project/speech-to-speech/`.

Only upload manually if the GitHub Actions workflow is unavailable and the maintainers have explicitly chosen that fallback.
