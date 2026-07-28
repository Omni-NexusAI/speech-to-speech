# Local Gemma Audio + FasterQwen3TTS

This fork keeps the upstream Realtime speech-to-speech framework intact while adding one local fast-test path:

- `--stt gemma-audio` skips Parakeet/transcription and sends completed audio turns directly to Gemma.
- `--qwen3_tts_backend openai-api` sends assistant text to the existing Dockerized FasterQwen3TTS-compatible server.

## Services

On this Windows machine, the target TTS container for this framework is:

- container: `qwen3-tts-faster`
- container id: `281c411e5cfe02d0b0d903f67cd3fb712ab38bc705eb3edbedd8a2041c78702b`
- URL: `http://127.0.0.1:8881/v1`
- endpoint: `/audio/speech`

The example config defaults to `8881` and expects the `1.7B-Base` backend model. This container hardcodes `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and exposes native clone PCM streaming. Its local server accepts an optional `language` value so multilingual requests do not inherit the profile's stored English language. The reproducible source patch is stored at `integrations/qwen3-tts-faster-language.patch`.

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

For the Windows local test setup, use the helper:

```powershell
.\scripts\start_local_gemma_realtime_backend.ps1
```

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
- Tool responses may include an optional model-generated `ASSISTANT_PREAMBLE`. It is spoken and retained before the native function call; no scripted fallback is synthesized when it is omitted.
- Faster on `8881` is the default. A Settings provider change is conversation-scoped and takes effect on the next conversation.
- Voice defaults to `clone:16d9bb336799`, displayed as `J.A.R.V.I.S`.
- The realtime UI lists saved Voice Studio profiles from `profiles/*/meta.json` and only exposes `Base` clone profiles.
- The realtime session stores clone IDs, but the frozen `qwen3-tts-faster` API looks up clones by profile name, so the adapter maps `clone:16d9bb336799` to `clone:J.A.R.V.I.S` before calling `/v1/audio/speech`.
- Pretrained Qwen voices are intentionally hidden for this local test path.
- Both providers receive `stream: true`, `response_format: pcm`, the selected clone ID, and an explicit assistant language when Gemma supplies one. Cyrillic, Japanese, Korean, and Chinese script detection is a fallback only.
- TTS streams are cancelled on Stop/disconnect and aborted when they exceed `min(60s, max(12s, 3x estimated speech duration + 5s))`.
- Assistant echo guard defaults to **Adaptive**: native browser AEC remains enabled while the exact generated playback PCM drives a classifier-only echo predictor. Echo and uncertain frames are withheld; 450 ms of confirmed independent speech releases the untouched buffered mic onset. Predictor residuals are diagnostics only and never replace microphone samples. The static voice-clone recording is never used. **Strict** suspends upload during playback/tail; **Off** uses native AEC only.
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

The local direct-audio mode keeps the latest 30 complete turns with `compact_history` disabled. System instructions live outside that bounded buffer. Progressive transcript previews are display-only. A validated primary transcript and its assistant response share session history; when transcript metadata is absent, the UI shows `[User audio]` without adding the placeholder or an assistant-only exchange to model history. Function calls and outputs remain retained for tool continuity. Context diagnostics report counts and trimming events without logging conversation text.

# Tool follow-ups

The browser sends `function_call_output`, an optional camera image, and one `response.create` immediately. The backend binds these to the model's original `call_id`, waits for the originating response to close, and starts exactly one continuation. A valid tool call remains executable even if Gemma omits a trustworthy transcript; that failure produces a temporary UI notice rather than fabricated conversation text.
