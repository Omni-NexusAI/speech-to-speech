from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GemmaAudioSTTHandlerArguments:
    gemma_audio_model_name: str = field(
        default="gemma-4-12b-it-qat",
        metadata={"help": "Model name to send to the local Gemma audio-capable OpenAI-compatible endpoint."},
    )
    gemma_audio_base_url: str = field(
        default="http://127.0.0.1:8818/v1",
        metadata={"help": "OpenAI-compatible base URL for local llama.cpp Gemma audio server."},
    )
    gemma_audio_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for the local Gemma server. If unset, GEMMA_API_KEY or LLAMA_CPP_API_KEY is used."},
    )
    gemma_audio_stream: bool = field(
        default=True,
        metadata={"help": "Stream Gemma chat completion deltas where supported. Default is true."},
    )
    gemma_audio_timeout_s: float = field(
        default=120.0,
        metadata={"help": "HTTP timeout for direct audio-to-Gemma requests."},
    )
    gemma_audio_format: str = field(
        default="wav",
        metadata={"help": "input_audio format label sent to the OpenAI-compatible endpoint. Default is wav."},
    )
    gemma_audio_prompt: str = field(
        default=(
            "Listen to the attached user audio and respond directly as a concise voice assistant. "
            "Do not include a transcript unless the user asks for one."
        ),
        metadata={"help": "Text instruction sent alongside each audio turn."},
    )
    gemma_audio_system_prompt: str = field(
        default="You are a local low-latency voice assistant. Answer naturally for speech synthesis.",
        metadata={"help": "System prompt for Gemma direct-audio turns."},
    )
