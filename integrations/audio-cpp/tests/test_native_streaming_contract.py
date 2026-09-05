"""Regression checks for the reproducible native/offline Qwen3-TTS patch.

The package is one nine-file patch based on a pinned upstream tree.  The
optional parity test is enabled by ``AUDIO_CPP_AUTHORITATIVE_TREE`` during
release validation; it intentionally does not require CUDA or model assets.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PATCH = Path(__file__).parents[1] / "patches" / "qwen3-native-pcm-streaming.patch"
DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"
PINNED_REV = "238ab6a9e321c17de8e120559f57efeedaeb1345"
TOUCHED_FILES = (
    "app/server/runtime.cpp",
    "include/engine/models/qwen3_tts/session.h",
    "include/engine/models/qwen3_tts/talker.h",
    "include/engine/models/qwen3_tts/tokenizer_speech_decoder.h",
    "include/engine/models/qwen3_tts/types.h",
    "src/models/qwen3_tts/loader.cpp",
    "src/models/qwen3_tts/session.cpp",
    "src/models/qwen3_tts/talker.cpp",
    "src/models/qwen3_tts/tokenizer_speech_decoder.cpp",
)


class NativeStreamingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = PATCH.read_text(encoding="utf-8")

    def test_one_authoritative_patch_covers_all_engine_sources(self) -> None:
        headers = [line for line in self.patch.splitlines() if line.startswith("diff --git ")]
        self.assertEqual(len(headers), len(TOUCHED_FILES))
        for path in TOUCHED_FILES:
            self.assertIn(f"a/{path} b/{path}", self.patch)
        self.assertNotIn("qwen3-native-pcm-streaming-cancel-fix.patch", DOCKERFILE.read_text(encoding="utf-8"))

    def test_callback_delivery_is_final_result_only(self) -> None:
        self.assertIn("policy.output = runtime::StreamingOutputKind::FinalResult;", self.patch)
        self.assertIn("return std::nullopt;", self.patch)
        # PullEvents would make app/streaming/streaming.cpp forward returned
        # events in addition to callback delivery.  There must be one direct
        # sink dispatch site in the engine patch, not a second queued path.
        self.assertEqual(self.patch.count("stream_event_sink_(event);"), 1)

    def test_talker_emits_only_complete_codec_frames(self) -> None:
        completed = self.patch.index("++out.generated_codes.frames;")
        callback = self.patch.index("on_generated_frame(generated_frame);")
        self.assertLess(completed, callback)

    def test_callback_abort_resets_without_flushing_a_tail(self) -> None:
        self.assertIn("Socket callbacks are permitted to abort", self.patch)
        self.assertIn("Do not decode/emit an EOS tail", self.patch)
        self.assertIn("(void) talker_step_->release_cached_step_graph();", self.patch)

    def test_offline_full_reuses_streaming_session_without_pcm_events(self) -> None:
        self.assertIn('constexpr const char * kDecodeModeOption = "qwen3_tts.decode_mode";', self.patch)
        self.assertIn('constexpr const char * kOfflineFullMode = "offline_full";', self.patch)
        self.assertIn("streaming_result_ = run_offline_request(request);", self.patch)
        self.assertIn("streaming_result_ = run_streaming_base_request(request);", self.patch)
        offline_branch = self.patch.index("if (decode_mode == Qwen3RequestDecodeMode::OfflineFull)")
        native_branch = self.patch.index("streaming_result_ = run_streaming_base_request(request);")
        callback = self.patch.index("stream_event_sink_(event);")
        self.assertLess(offline_branch, native_branch)
        self.assertGreater(callback, native_branch)
        self.assertEqual(self.patch.count("stream_event_sink_(event);"), 1)

    def test_offline_full_result_is_one_shot_and_truthfully_reported(self) -> None:
        self.assertIn("runtime::TaskResult result = std::move(*streaming_result_);", self.patch)
        self.assertIn("streaming_result_.reset();", self.patch)
        self.assertIn('attach_decode_mode(*streaming_result_, "offline-full-decoder");', self.patch)
        self.assertIn('attach_decode_mode(*streaming_result_, "native-incremental-pcm");', self.patch)
        self.assertIn('"X-AudioCPP-Qwen3-Decode-Mode"', self.patch)

    def test_unknown_qwen_decode_mode_fails_explicitly(self) -> None:
        self.assertIn(
            '"qwen3_tts.decode_mode must be native_incremental or offline_full"',
            self.patch,
        )

    def test_fixed_tail_and_reference_trim_are_packaged(self) -> None:
        native = self.patch.split("runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1)[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        self.assertIn(
            'parse_block_frames("qwen3_tts.stream_left_context_frames", 25)',
            self.patch,
        )
        self.assertIn("decoder_capacity_frames = left_context_frames + std::max(", native)
        self.assertIn("First, steady, and EOS-tail blocks use one padded causal decoder", native)
        self.assertNotIn("decode_and_trim_reference(*voice_prompt.reference_codes, pending)", native)
        self.assertIn("audio = speech_decoder_->decode_padded(decoder_input, decoder_capacity_frames);", native)
        self.assertIn("const int64_t drop_samples = context_frames * 1920;", native)
        self.assertIn("exact context samples are discarded", native.lower())
        self.assertIn("decode_padded(", self.patch)
        self.assertIn("std::vector<int32_t> padded_codes", self.patch)
        self.assertIn("graph_->run(padded_codes.data(), padded_codes.size())", self.patch)
        self.assertIn("codec_codes.frames * kDecodeSamplesPerCode", self.patch)

    def test_stream_sse_proof_is_engine_owned_and_precedes_pcm(self) -> None:
        self.assertIn("qwen3_stream_decode_mode_artifact", self.patch)
        self.assertIn('attach_decode_mode(event, "native-incremental-pcm");', self.patch)
        # The C++ JSON literal is escaped inside the unified patch.
        self.assertIn('\\"type\\":\\"speech.decode_mode\\"', self.patch)
        self.assertIn("streaming Qwen3 speech produced no decoder-mode proof", self.patch)
        self.assertIn('model.config.family == "qwen3_tts"', self.patch)
        self.assertIn("if (require_qwen_decode_mode && !wrote_decode_mode)", self.patch)
        self.assertIn("speech.audio.done", self.patch)

    def test_generic_speech_sse_does_not_require_qwen_artifacts(self) -> None:
        self.assertIn("const bool require_qwen_decode_mode", self.patch)
        self.assertIn("if (require_qwen_decode_mode && !wrote_decode_mode)", self.patch)
        self.assertIn("if (!wrote_audio)", self.patch)
        self.assertIn("if (require_qwen_decode_mode) {", self.patch)

    def test_decoder_context_uses_upstream_25_frame_window_everywhere(self) -> None:
        self.assertIn('parse_block_frames("qwen3_tts.stream_left_context_frames", 25)', self.patch)
        self.assertIn("decoder_capacity_frames = left_context_frames + std::max(", self.patch)
        self.assertIn("must not exceed 300 codec frames", self.patch)

    def test_voice_prompt_cache_is_reused_without_claiming_phrase_decoder_state(self) -> None:
        native = self.patch.split("runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1)[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        self.assertIn("resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder)", native)
        self.assertIn("voice_prompt_cache_", self.patch)
        self.assertIn("no decoder-state", native)
        self.assertIn("continuity between phrase requests", native)
        self.assertNotIn("static Qwen3", native)

    def test_clone_conditioning_cache_is_keyed_and_truthful(self) -> None:
        # The default cache owns immutable clone conditioning. Generated state
        # stays phrase-local even when the opt-in prefix prototype is present.
        self.assertIn("resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder)", self.patch)
        self.assertIn("qwen3_tts.voice_prompt_ms", self.patch)
        self.assertNotIn("talker_clone_conditioning_equal", self.patch)

    def test_opt_in_prefix_cache_reuses_clone_conditioned_prefix_across_phrase_text(self) -> None:
        self.assertIn('"qwen3_tts.talker_prefix_cache_slots"', self.patch)
        self.assertIn("immutable_prefix_cache_slots = 0", self.patch)
        self.assertIn("talker_prefix_cache_slots_(mem_saver_from_options(options) ? 0", self.patch)
        self.assertIn("state.immutable_prefix_steps <= prompt_steps", self.patch)
        self.assertNotIn("state.immutable_prefix_steps == prompt_steps", self.patch)
        self.assertIn("cached_prefill_steps = state.immutable_prefix_steps", self.patch)
        self.assertIn("key.prefix_embeddings.assign(state.prompt.begin(), prefix_end);", self.patch)
        self.assertIn("run_prefill_embeddings_with_state(key.prefix_embeddings, cached_prefill_steps)", self.patch)
        self.assertIn("cached_prefill_steps < prompt_steps", self.patch)
        self.assertIn("current = cached_step_graph_->run_step(row_at(state.prompt, row, config.hidden_size));", self.patch)
        self.assertIn("reference audio is not a safe boundary", self.patch)

    def test_opt_in_prefix_cache_key_preserves_exact_clone_and_language_conditioning(self) -> None:
        # The cache lives in one model runtime and its exact, unquantized fused
        # prefix includes clone/reference and language control rows.  A phrase
        # suffix is replayed locally; generated speech never becomes a cache key.
        self.assertIn("std::vector<float> prefix_embeddings;", self.patch)
        self.assertIn("lhs.prefix_embeddings == rhs.prefix_embeddings", self.patch)
        self.assertIn("They include clone/reference conditioning and the", self.patch)
        self.assertIn("language control rows", self.patch)
        self.assertIn("no generated row is retained", self.patch)

    def test_opt_in_prefix_cache_reimports_only_prefill_state(self) -> None:
        self.assertIn("Never export\n+        // the cache after generated frames", self.patch)
        self.assertIn("const runtime::TransformerKVState * cached_state = &prefill_output.state;", self.patch)
        self.assertIn("cached_graph_has_state = false", self.patch)
        self.assertIn("immutable_prefix_cache_.put(key, std::move(entry));", self.patch)
        self.assertIn("replay every phrase-dependent row for", self.patch)

    def test_clone_prompt_material_cache_reuses_only_exact_static_conditioning(self) -> None:
        # Base-clone prompt text is fused into early reference-code rows, so a
        # whole-reference KV reuse would be incorrect. The packaged cache keeps
        # only host-side clone material and the already-safe leading KV prefix.
        self.assertIn("struct VoiceClonePromptMaterial", self.patch)
        self.assertIn("struct VoiceClonePromptMaterialKey", self.patch)
        self.assertIn("reference_audio_hash", self.patch)
        self.assertIn("reference_sample_count", self.patch)
        self.assertIn("reference_text == rhs->reference_text", self.patch)
        self.assertIn("speaker_embedding_equal(lhs.speaker_embedding, rhs.speaker_embedding)", self.patch)
        self.assertIn("voice_clone_prompt_material_cache_", self.patch)
        self.assertIn("build_voice_clone_prompt_state_from_material", self.patch)
        self.assertIn("cannot be reused without changing model", self.patch)
        self.assertIn("rebuild/replay only the required causal suffix", self.patch)

    def test_patch_applies_and_matches_authoritative_tree(self) -> None:
        authoritative = os.environ.get("AUDIO_CPP_AUTHORITATIVE_TREE")
        if not authoritative:
            self.skipTest("set AUDIO_CPP_AUTHORITATIVE_TREE for clean-tree package parity validation")
        source = Path(authoritative).resolve()
        self.assertTrue(source.is_dir(), source)
        with tempfile.TemporaryDirectory(prefix="audio-cpp-native-patch-") as temporary:
            clean = Path(temporary) / "clean"
            subprocess.run(["git", "clone", "--no-checkout", str(source), str(clean)], check=True)
            subprocess.run(["git", "-C", str(clean), "checkout", "--detach", PINNED_REV], check=True)
            subprocess.run(["git", "-C", str(clean), "apply", "--check", str(PATCH)], check=True)
            subprocess.run(["git", "-C", str(clean), "apply", str(PATCH)], check=True)
            for relative in TOUCHED_FILES:
                self.assertEqual(
                    (clean / relative).read_bytes(),
                    (source / relative).read_bytes(),
                    relative,
                )


if __name__ == "__main__":
    unittest.main()
