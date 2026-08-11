# Local Gemma Audio + Qwen3-TTS Providers

This fork keeps the upstream Realtime speech-to-speech framework intact while adding one local fast-test path:

- `--stt gemma-audio` skips Parakeet/transcription and sends completed audio turns directly to Gemma.
- `--qwen3_tts_backend openai-api` sends assistant text to a conversation-scoped OpenAI-compatible Qwen3-TTS provider: Faster by default, or an explicitly selected and validated Groxaxo/audio.cpp candidate.

## Services

On this Windows machine, the target TTS container for this framework is:

- container: `qwen3-tts-faster`
- URL: `http://127.0.0.1:8881/v1`
- endpoint: `/audio/speech`

The example config defaults to `8881` and expects the `1.7B-Base` backend model. This container hardcodes `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and exposes native clone PCM streaming. Its local server accepts an optional `language` value and preserves explicit `Auto`, so multilingual requests do not inherit the clone reference's stored language. The reproducible source patch is stored at `integrations/qwen3-tts-faster-language.patch`; Realtime diagnostics continue to label Faster Auto support unverified until that exact adapter is rebuilt and probed.

The Settings panel also lists the optional user-managed Groxaxo candidate on `http://127.0.0.1:8882/v1`. The UI reads `/v1/backend/models` and permits a new conversation only when Voice Studio already has `0.6B-Base` or `1.7B-Base` loaded. The pipeline never starts, stops, loads, unloads, switches, or silently falls back from that candidate.

Gemma should be launched with the local llama.cpp settings from `C:\llama.cpp\launch_gemma-4-12B-it-qat-MTP.ps1`, with context reduced to 16k. This repo includes a wrapper:

```powershell
$env:GEMMA_API_KEY = '<local llama-server key>'
.\scripts\launch_gemma_4_12b_16k.ps1
```

Do not run Gemma/llama.cpp tests while the local server is being rebuilt.

## Run Backend

```powershell
$env:GEMMA_API_KEY = '<local llama-server key>'
python -m speech_to_speech.s2s_pipeline .\examples\local_gemma_fasterqwen3tts.json
```

For a foreground Windows test run, use the backend-only helper:

```powershell
.\scripts\start_local_gemma_realtime_backend.ps1
```

The helper launches this worktree's backend source only. Gemma and every TTS
service remain user-managed and are checked when a conversation starts.

The backend listens on:

```text
ws://127.0.0.1:8765/v1/realtime
```

## Test Clients

Use either the included Python client:

```powershell
python .\scripts\listen_and_play_realtime.py --host 127.0.0.1 --port 8765
```

or the integrated HF Realtime Voice UI under `web/hf-realtime-voice` once dependencies are installed.

Start the integrated UI with:

```powershell
.\scripts\start_hf_realtime_frontend.ps1 -Open
```

The UI runs on `http://127.0.0.1:7862` and should point Settings to `http://127.0.0.1:8765`.

## Remote multimodal model

Settings can switch model inference from **Local** to a conversation-scoped **Remote** OpenAI-compatible endpoint. Enter its `/v1` URL, model name, and optional bearer key, then use **Test model connection** before restarting the conversation. Remote selection replaces local Gemma for direct audio, optional live preview, camera/tool selection, post-tool generation, model identity, tokenization, and context discovery. It never silently falls back to Local; FasterQwen3TTS remains separately selected.

On the model machine, expose llama.cpp to the LAN and enable the tool-compatible chat template, for example:

```powershell
llama-server.exe --host 0.0.0.0 --port 8080 --jinja --api-key '<private-lan-key>' <your model and multimodal projector arguments>
```

Allow that TCP port through the model machine's firewall only for the trusted private network. The selected model must support audio input, camera images, OpenAI-compatible chat completions, and native function tools. Use `http://<model-machine-ip>:8080/v1` in Settings. The browser stores the optional key in localStorage by explicit design; the backend keeps it in connection-scoped memory and exposes only `api_key_set` in status events. Prefer a trusted wired LAN and do not expose an unauthenticated llama-server to untrusted networks.

The realtime backend and UI can start while local Gemma is unavailable. Local endpoint readiness is checked only when a Local conversation starts; Remote is validated before microphone activation.

## Notes

- No separate ASR model is used in `gemma-audio` mode.
- A VAD soft endpoint waits 250 ms before launching Gemma. The seven-second continuation horizon is anchored to the first soft endpoint and never slides forward. Continuations require 192 ms of confirmed speech and are bounded to eight revisions or 30 seconds of combined audio; a bound starts a new turn without dropping the new fragment.
- Direct audio, optional previews, and post-tool generation share one model-operation coordinator. Only one request can occupy the selected Local or Remote endpoint; optional previews are dropped while it is busy. VAD is the sole audio-admission boundary: transcript metadata comes only from the primary request and cannot reject audio, trigger another request, suppress an answer, or block a tool call.
- Direct and post-tool timeouts close the active stream, emit one failed completion, release response ownership, and resume listening without canned speech. Post-tool responses use streamed Chat Completions and dispatch the first complete sentence to TTS. A barge-in actively closes the obsolete stream; if transport shutdown exceeds two seconds, that generation is detached and its late output is rejected without ending the conversation.
- Tool responses may include a model-generated `ASSISTANT_PREAMBLE`. It is spoken and retained before the native function call; when it is omitted, a request-specific varied per-tool fallback keeps the tool call audible without reusing one stock phrase.
- Faster on `8881` is the default. Groxaxo on `8882` remains user-managed, and the isolated audio.cpp candidate on `8890` is opt-in after explicit validation. A Settings provider change is conversation-scoped and takes effect on the next conversation.
- Voice defaults to `clone:16d9bb336799`, displayed as `J.A.R.V.I.S`.
- The realtime UI lists only live `Base` clone profiles from the selected backend. Faster reads its configured `profiles/*/meta.json` library, audio.cpp resolves its private candidate library, and Groxaxo remains inventory-only unless its API advertises safe mutation support.
- The realtime session stores clone IDs, but the frozen `qwen3-tts-faster` API looks up clones by profile name, so the adapter maps `clone:16d9bb336799` to `clone:J.A.R.V.I.S` before calling `/v1/audio/speech`.
- Pretrained Qwen voices are intentionally hidden for this local test path.
- All three OpenAI-compatible TTS providers receive `stream: true`, `response_format: pcm`, their selected backend-scoped clone ID, and an explicit assistant language when Gemma supplies one. Cyrillic, Japanese, Korean, and Chinese script detection is a fallback only. audio.cpp uses native incremental PCM only after its live capability and chunk probe succeed; otherwise it reports and uses the cancellable buffered fallback.
- TTS streams are cancelled on Stop/disconnect and aborted when they exceed `min(60s, max(12s, 3x estimated speech duration + 5s))`.
- Echo control defaults to **Native**, using the browser's built-in AEC. **Adaptive** is available only when the bundled Sonora AEC3 worklet passes manifest, hash, ABI, and load validation; it receives the exact scheduled playback PCM and preserves independent near-end speech for barge-in. Any AEC3 failure resolves truthfully to Native. **Strict** applies the reference-aware path more aggressively and fails closed by withholding uncertain upload through the echo tail. No mode uploads a custom NLMS/predictor residual, and the static voice-clone recording is never used as the echo reference.
# Managed lifecycle

Use the tracked background launcher for routine local testing:

```powershell
.\scripts\local_realtime.ps1 -Action start
.\scripts\local_realtime.ps1 -Action status
.\scripts\local_realtime.ps1 -Action restart -Component backend
.\scripts\local_realtime.ps1 -Action stop
```

Add `-Component frontend` or `-Component backend` to scope an action, and `-Open` to open the UI after startup. State and separate process logs are written under `.runtime`; stop and restart only terminate a process whose PID, executable, and start time match that recorded state. Gemma and FasterQwen3TTS are dependency services and are never stopped by this script.

# Conversation context

The local direct-audio mode keeps the latest 30 complete turns with `compact_history` disabled. System instructions live outside that bounded buffer. Every VAD-accepted turn creates exactly one semantic user anchor before its assistant text or tool state. A newer cumulative VAD revision replaces that anchor within the same session and logical turn instead of creating another history entry. When Gemma returns a validated transcript, that same anchor is upgraded in place to text. If the same primary response has no transcript, a bounded validated `USER_MEMORY` paraphrase may upgrade the anchor instead; it is hidden from the UI, starts no second request, and cannot gate or replace assistant or tool output. If neither field is safe, the session-only WAV `input_audio` remains fail-closed as the historical user item. Progressive previews and the `[User audio]` fallback are visual metadata only and never enter model history. Function calls and outputs remain ordered after the single user anchor for tool continuity. Accepted audio may use any supported language, accent, or code-switching; replies follow the current utterance naturally, use an explicit session language when requested, and retain `Auto` for mixed or unspecified output. Context diagnostics report only lifecycle and size data and never log transcript or semantic-memory content.

# Tool follow-ups

The browser sends `function_call_output`, an optional camera image, and one `response.create` immediately. The backend binds these to the model's original `call_id`, waits for the originating response to close, and starts exactly one continuation. A valid tool call remains anchored to the single accepted user item and executable even if Gemma omits optional transcript metadata; a validated hidden semantic paraphrase can preserve that request for later context, while the UI may still show `[User audio]`. Missing or unsafe memory retains raw audio rather than inserting placeholder text.
