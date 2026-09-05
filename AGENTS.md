# Repository Instructions

## DOX Framework

- DOX is installed here as a shallow AGENTS.md hierarchy.
- AGENTS.md files are binding work contracts for their subtrees.
- Before editing, read this root AGENTS.md and every child AGENTS.md on the path to the files you will touch.
- After meaningful edits, update the nearest AGENTS.md and any affected parent/child indexes when structure, contracts, workflows, or durable preferences changed.
- Keep docs concise and operational; document stable contracts rather than change history.

## Local Voice-Agent Testing Contract

- Managed direct-audio realtime uses context-budget compaction at 70% usage, targeting 50% and preserving up to six recent settled exchanges verbatim subject to capacity. Ordinary and doubled turn-count eviction are disabled on this path; context remains session-scoped and resets with the WebSocket/backend.
- VAD is the only direct-audio admission boundary. A completed accepted turn always reaches Gemma once; transcript metadata is optional and must never gate assistant text, tools, or response completion. The live-transcription toggle controls only the temporary speaking bubble.
- Context diagnostics distinguish estimated and measured usage against the selected model's usable context budget, including instruction, tool, response and multimodal reserves. Compaction protects current/provisional input and unresolved tools, applies atomically only to valid source history, and preserves original history on failure; insufficient capacity is an explicit maintenance state, never silent eviction.
- Genuine assistant output remains eligible for audible-playback history commitment when the optional user transcript is absent. Keep explicit internal provenance without inventing user words, and preserve provisional rollback, interrupted responses and exact-once tool behavior.
- `scripts/local_realtime.ps1` is the managed background entry point; foreground launchers remain available for raw-console debugging.
- Model provider selection is conversation-scoped. Remote completely replaces local Gemma operations without fallback; TTS selection remains independent.
- One conversation may own only one model HTTP operation at a time. Barge-in cancels or detaches that generation within two seconds without ending the WebSocket session or clearing context.
- Accepted speech and assistant responses use monotonic input/response epochs. Claim a response as pending before Gemma starts; newer confirmed speech invalidates the old epoch before cancellation so stale text, TTS, PCM, history, metrics, and terminal events cannot escape. Browser history commits only after an idempotent first-rendered-sample acknowledgement; clients without that capability commit after the first PCM frame is successfully delivered.
- An explicit output-admissible response epoch is authoritative through Gemma, both LLM paths, the output processor, and every TTS backend. A completed owner may remain current only for its late rendered-playback acknowledgement; it cannot emit further model, TTS, PCM, metric, or terminal output. Speculative turn identity is a legacy/pre-ownership fallback only; a queued non-interrupt successor waits for promotion before either LLM is called and refreshes its rollback checkpoint when promoted so it cannot erase the audible owner's trailing history.
- Freeze provider/model lifecycle, clone fingerprint, tuning profile/revision, language, and one response-scoped seed at the first stable assistant phrase. Every later phrase in that answer reuses the snapshot; realtime setting changes apply to the next response.
- audio.cpp Adaptive playback may change only the bounded startup/re-prime PCM reservoir at the native 24 kHz clock. It must not time-stretch, pitch-shift, duplicate, or silently switch to full buffering; report a provider that cannot sustain realtime and retain the explicit buffered-phrase fallback.
- Browser echo protection defaults to Native browser AEC. Exact playback PCM remains available as a reference tap; Adaptive resolves to Native until an actual AEC3 module is loaded and validated, while Strict may fail closed by pausing upload through the echo tail.

- This fork is used for local speech-to-speech testing with the OpenAI Realtime-compatible server preserved.
- Keep upstream pipeline behavior intact unless a change is required for the local direct-audio Gemma path.
- The local fast-test path intentionally skips Parakeet/transcription with `--stt gemma-audio` and sends completed audio turns directly to the local Gemma audio model.
- Speech output defaults to the existing Dockerized FasterQwen3TTS-compatible OpenAI API on `8881`; the optional Groxaxo candidate on `8882` is user-managed and must never be started, stopped, or switched by this repo.
- The custom path defaults to local inference. Explicit private-LAN remote endpoints are allowed, but no automatic cloud or local fallback should be introduced.
- Future laptop Gemma tests must use only the user's existing normal configuration launcher. Do not substitute context or other settings, prescribe an ad-hoc launcher, or infer which normal launcher the user selected.
- The target TTS service is the stable Docker container name `qwen3-tts-faster` on `http://127.0.0.1:8881/v1`; container IDs are transient after authorized rebuilds and must not be used by lifecycle scripts.
- Qwen3 clone output must use the `1.7B-Base` backend model by default; the `qwen3-tts-faster` API exposes native incremental PCM16 clone streaming while preserving buffered formats for non-streaming requests.
- The isolated audio.cpp Qwen3-TTS candidate remains opt-in at `8890`/`8891`: its `qwen3tts-audiocpp` provider is selected only after explicit health, resident-model, private Base-profile, and speech validation. Use native incremental PCM only when the running engine advertises it and the supervisor consumes its engine-owned decoder-mode SSE proof from the same iterator before committing the exact native header or relaying PCM; a direct chunk probe must then verify delivery. Otherwise label and retain the cancellable buffered-phrase fallback. It never changes Faster or Groxaxo lifecycle.
- Buffered phrase PCM remains the stable audio.cpp Playground and HF Realtime
  fallback/default while native PCM is repaired and benchmarked. Native PCM is
  selectable only for explicit candidate testing and is never promoted solely
  because the engine advertises the capability.
- Faster clone requests carry an explicit assistant language when known; keep the reproducible local server patch under `integrations/` in sync with the rebuilt image.
- The realtime UI exposes live `Base` clone profiles from the selected TTS
  backend and retains one valid selection per backend. Faster keeps
  `clone:16d9bb336799` (`J.A.R.V.I.S`) as its default.
- Managed HFRT startup is model-independent: missing Gemma or TTS endpoints are
  warnings, and repository launchers never manage external model containers.
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
