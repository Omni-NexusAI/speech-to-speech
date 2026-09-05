# Local Gemma Audio + FasterQwen3TTS

This fork keeps the upstream Realtime speech-to-speech framework intact while adding one local fast-test path:

- `--stt gemma-audio` skips Parakeet/transcription and sends completed audio turns directly to Gemma.
- `--qwen3_tts_backend openai-api` sends assistant text to the existing Dockerized FasterQwen3TTS-compatible server.

## Services

On this Windows machine, the target TTS container for this framework is:

- container: `qwen3-tts-faster`
- URL: `http://127.0.0.1:8881/v1`
- endpoint: `/audio/speech`

The example config defaults to `8881` and expects the `1.7B-Base` backend model. This container hardcodes `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and exposes native clone PCM streaming. Its local server accepts an optional `language` value so multilingual requests do not inherit the profile's stored English language. The reproducible source patch is stored at `integrations/qwen3-tts-faster-language.patch`.

The Settings panel also lists the optional user-managed Groxaxo candidate on `http://127.0.0.1:8882/v1`. The UI reads `/v1/backend/models` and permits a new conversation only when Voice Studio already has `0.6B-Base` or `1.7B-Base` loaded. The pipeline never starts, stops, loads, unloads, switches, or silently falls back from that candidate.

The isolated `qwen3tts-audiocpp` candidate is available at `http://127.0.0.1:8890/v1` only after explicit health, resident-model, selected-clone, and synthesis validation. It owns a separate private profile library and may carry one validated tuning-profile snapshot per conversation. The realtime pipeline never starts, stops, loads, unloads, or switches its model. Native incremental PCM remains opt-in and must be advertised and probed successfully; buffered phrase synthesis remains a separately selected stable mode. A native failure is surfaced instead of silently replaying the phrase through buffered synthesis.

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

The UI runs on `http://127.0.0.1:7862` and should point Settings to `http://127.0.0.1:8765`. The managed launcher prints this UI URL only after frontend readiness is verified; it opens a browser only when `-Open` is supplied.

## Remote multimodal model

Settings can switch model inference from **Local** to a conversation-scoped **Remote** OpenAI-compatible endpoint. Enter its `/v1` URL, model name, and optional bearer key, then use **Test model connection** before restarting the conversation. Remote selection replaces local Gemma for direct audio, optional live preview, camera/tool selection, post-tool generation, model identity, tokenization, and context discovery. It never silently falls back to Local; FasterQwen3TTS remains separately selected.

On the model machine, expose llama.cpp to the LAN and enable the tool-compatible chat template, for example:

```powershell
llama-server.exe --host 0.0.0.0 --port 8080 --jinja --api-key '<private-lan-key>' <your model and multimodal projector arguments>
```

Allow that TCP port through the model machine's firewall only for the trusted private network. The selected model must support audio input, camera images, OpenAI-compatible chat completions, and native function tools. Use `http://<model-machine-ip>:8080/v1` in Settings. The HF Realtime key remains page-session-only; the backend keeps it in connection-scoped memory and exposes only `api_key_set` in status events. Prefer a trusted wired LAN and do not expose an unauthenticated llama-server to untrusted networks.

The realtime backend and UI can start while local Gemma is unavailable. Local endpoint readiness is checked only when a Local conversation starts; Remote is validated before microphone activation.

## Notes

- No separate ASR model is used in `gemma-audio` mode.
- A VAD soft endpoint waits 250 ms before launching Gemma. The seven-second continuation horizon is anchored to the first soft endpoint and never slides forward. Continuations require 192 ms of confirmed speech and are bounded to eight revisions or 30 seconds of combined audio; a bound starts a new turn without dropping the new fragment.
- Direct audio, optional previews, and post-tool generation share one model-operation coordinator. Only one request can occupy the selected Local or Remote endpoint; optional previews are dropped while it is busy. VAD is the sole audio-admission boundary: transcript metadata comes only from the primary request and cannot reject audio, trigger another request, suppress an answer, or block a tool call.
- Direct and post-tool timeouts close the active stream, emit one failed completion, release response ownership, and resume listening without canned speech. Post-tool responses use streamed Chat Completions and dispatch the first complete sentence to TTS. A barge-in actively closes the obsolete stream; if transport shutdown exceeds two seconds, that generation is detached and its late output is rejected without ending the conversation.
- Every accepted input and assistant answer carries monotonic input and response epochs. New confirmed speech invalidates the old epoch before transport cancellation, so late text, PCM, history, metrics, and terminal events cannot escape. With interruption disabled, an accepted successor waits behind the audible owner and reaches the selected LLM exactly once after promotion. Unheard provisional history is rolled back; once the browser reports its first rendered sample, an interrupted answer is retained once. Clients that do not advertise rendered-playback acknowledgements retain compatibility settlement after the first non-empty PCM frame is successfully sent and receive no browser-only speech-start decision event.
- Each assistant answer freezes one synthesis snapshot at its first stable phrase: provider/model lifecycle, clone fingerprint, profile revision, resolved tuning, language, and seed. A profile without an explicit seed gets one random response-local seed reused by every phrase. Live profile changes apply to the next response and cannot split the active answer's voice identity.
- Tool responses may include an optional model-generated `ASSISTANT_PREAMBLE`. It is spoken and retained before the native function call; when omitted, a small deterministic per-tool acknowledgement prevents a silent tool call without claiming a fabricated result.
- Faster on `8881` is the default. A Settings provider change is conversation-scoped and takes effect on the next conversation.
- Voice defaults to `clone:16d9bb336799`, displayed as `J.A.R.V.I.S`.
- The realtime UI lists saved Voice Studio profiles from `profiles/*/meta.json` and only exposes `Base` clone profiles.
- The realtime session stores clone IDs, but the frozen `qwen3-tts-faster` API looks up clones by profile name, so the adapter maps `clone:16d9bb336799` to `clone:J.A.R.V.I.S` before calling `/v1/audio/speech`.
- Pretrained Qwen voices are intentionally hidden for this local test path.
- Streaming providers receive `stream: true`, `response_format: pcm`, the selected clone ID, and an explicit assistant language when Gemma supplies one. Cyrillic, Japanese, Korean, and Chinese script detection is a fallback only. Faster/Groxaxo retain their established 16 kHz pipeline transport; verified native audio.cpp PCM preserves the model-native 24 kHz source rate through browser playback.
- TTS streams are cancelled on Stop/disconnect. Candidate live output has a generated-audio cap of `min(60s, max(12s, 3x estimated speech duration + 5s))`, separate from its 180-second wall deadline; Full Quality WAV retains its independent offline policy.
- Assistant echo protection defaults to **Native browser AEC**. The exact scheduled playback PCM remains wired as a reference tap, but Adaptive resolves truthfully to Native until a validated AEC3 module is loaded; no classifier-only filter is presented as echo cancellation. **Strict** may fail closed by pausing microphone upload through playback and its tail without manufacturing silence. The static voice-clone recording is never used as the echo reference.
- For audio.cpp playback, **Adaptive continuity** changes only the bounded startup/re-prime reservoir at the original 24 kHz clock for native and buffered-phrase PCM; it never stretches, slows, duplicates, pitch-shifts, or resamples audio to hide starvation. **Fast start** retains the fixed low-reservoir path. If the required reserve exceeds two seconds, Diagnostics reports that the provider cannot sustain realtime and leaves the explicit full-buffer fallback available.
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

Managed direct-audio conversations use context-based compaction, not 30-turn or doubled-turn eviction. Compaction starts at 70% of the usable selected-model context budget and targets 50%, preserving up to six recent settled exchanges exactly when capacity allows. Instructions, current audio, response allowance, tool schemas, provisional responses, and unresolved tool transactions are protected. Diagnostics distinguishes measured token counts from estimates and reports compaction status and failures. Settings can adjust the bounded policy.

Progressive transcript previews remain display-only. When optional primary transcription is absent, the actual assistant answer is retained with internal provenance for an untranscribed audio exchange; no user words or placeholder transcript are invented. Multi-part tool transactions remain one exchange. Browser audibility still controls commitment, and superseding an unheard response rolls back its provisional history.

One low-priority summary request uses the conversation's selected endpoint and the existing operation coordinator. New speech cancels or detaches it. A summary is explicitly identified as conversation memory and replaces history atomically only if its source remains current. Failure, interruption, reset, or endpoint changes retain original history. If another request cannot fit safely, the conversation reports context maintenance rather than silently deleting turns or sending an oversized request.

# Candidate diagnostics

The mounted orange Studio uses uncached, same-origin tuning, model, and clone snapshot routes. Its explicit diagnostic WAV picker can exercise the same LLM/TTS path without microphone capture, but that is not a physical-microphone or listening test. Both native experimental and buffered phrase modes freeze the answer's voice configuration once. Full Quality WAV remains independent of live phrase controls.

Candidate live synthesis has separate wall-clock and generated-audio limits. A codec-frame cap is not successful EOS, and failed/aborted output is never exported as a complete clip. A valid EOS exactly at the frame cap remains successful. After raw-PCM headers have been sent, the opaque, short-lived request outcome explains any failure; clients stop only that answer and clear its exact queue, without replay through another mode/provider or unloading the model. Codec-cycle telemetry remains observation-only, not a proven acoustic degeneration detector.

`scripts/measure_candidate_tts_paths.py --execute` performs explicit resident-model diagnostics with frozen conditioning and no saved-setting changes. Its HTTP transport timing does not certify browser playback or engine chunk boundaries. `scripts/validate_direct_audio_followups.py --execute` sends explicitly selected local WAV clips through the managed realtime path; credentials come from an environment variable and audio export is opt-in. Keep artifacts outside version control and reserve microphone, voice identity, prosody, and seam judgments for listening tests.

# Tool follow-ups

The browser sends `function_call_output`, an optional camera image, and one `response.create` immediately. The backend binds these to the model's original `call_id`, waits for the originating response to close, and starts exactly one continuation. A valid tool call remains executable even if Gemma omits a trustworthy transcript; that failure produces a temporary UI notice rather than fabricated conversation text.
