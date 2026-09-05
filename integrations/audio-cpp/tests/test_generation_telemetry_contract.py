"""Static contract for the engine-owned Qwen3 generation guard telemetry.

These checks deliberately inspect the distributable patch: they need no model,
CUDA device, container, or audio synthesis run.
"""
from __future__ import annotations

import unittest
from pathlib import Path


PATCH = Path(__file__).parents[1] / "patches" / "qwen3-native-pcm-streaming.patch"


class GenerationTelemetryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = PATCH.read_text(encoding="utf-8")

    def test_request_cap_is_positive_and_below_fixed_native_capacity(self) -> None:
        self.assertIn('"qwen3_tts.max_generated_frames"', self.patch)
        self.assertIn("*value <= 0 || *value >= config.max_new_tokens", self.patch)
        self.assertIn("options.max_new_tokens = *max_generated_frames + 1", self.patch)
        self.assertIn("cannot be combined with max_tokens", self.patch)

    def test_talker_reports_complete_codec_frames_and_terminal_reason(self) -> None:
        self.assertIn("enum class Qwen3GenerationTermination", self.patch)
        self.assertIn("int64_t generation_cap = 0", self.patch)
        self.assertIn("out.generation_cap = max_new_tokens - 1", self.patch)
        self.assertIn("out.termination = Qwen3GenerationTermination::Eos", self.patch)
        self.assertLess(
            self.patch.index("++out.generated_codes.frames;"),
            self.patch.index("on_generated_frame(generated_frame);"),
        )

    def test_native_limit_does_not_flush_a_success_like_tail(self) -> None:
        self.assertIn("Only EOS has a valid final causal tail", self.patch)
        self.assertIn("codes.termination == Qwen3GenerationTermination::Eos", self.patch)
        self.assertIn("pending.codes.clear();", self.patch)
        self.assertIn('"qwen3_tts.generation_cancelled"', self.patch)

    def test_native_generation_result_outlives_the_callback_guard(self) -> None:
        native = self.patch.split(
            "runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1
        )[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        self.assertIn("const auto codes = [&]()", native)
        self.assertLess(
            native.index("const auto codes = [&]()"),
            native.index("if (codes.termination == Qwen3GenerationTermination::Eos)"),
        )

    def test_offline_explicit_cap_is_request_total_and_returns_no_truncated_master(self) -> None:
        self.assertIn("requested_generation_cap", self.patch)
        self.assertIn("remaining_generated_frames", self.patch)
        self.assertIn(
            "capped_chunk_request.options[kMaxGeneratedFramesOption] = std::to_string(remaining_generated_frames)",
            self.patch,
        )
        self.assertIn("generation_cap += codes.generation_cap", self.patch)
        self.assertIn("requested_generation_cap.value_or(generation_cap)", self.patch)
        self.assertIn("not a successful offline master", self.patch)
        self.assertIn("qwen3_tts.max_generated_frames_exhausted", self.patch)
        self.assertIn("release_talker_cached_step_graph();", self.patch)

    def test_completed_results_carry_headers_and_same_iterator_sse_telemetry(self) -> None:
        for header in (
            "X-AudioCPP-Generated-Frames",
            "X-AudioCPP-Generation-Cap",
            "X-AudioCPP-Termination",
        ):
            self.assertIn(header, self.patch)
        self.assertIn('"type\\\":\\\"speech.generation\\\"', self.patch)
        self.assertIn('"termination\\\":', self.patch)
        self.assertIn('"complete\\\":', self.patch)
        self.assertLess(
            self.patch.index('"type\\\":\\\"speech.generation\\\"'),
            self.patch.index('"type\\\":\\\"speech.audio.done\\\"'),
        )


if __name__ == "__main__":
    unittest.main()
