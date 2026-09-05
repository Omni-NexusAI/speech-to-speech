"""Convenience `TypeAlias` definitions for pipeline handler generics.

These aliases keep handler declarations readable (e.g. `BaseHandler[STTIn, STTOut]`)
without hiding what actually flows between stages (see `pipeline/messages.py` and
`pipeline/queue_types.py` for the full story, including control + sentinel items).
"""

from __future__ import annotations

# ruff: noqa: I001

from typing import TypeAlias

import numpy as np

from speech_to_speech.pipeline.messages import (
    AudioOutput,
    DirectAssistantRequest,
    DirectAssistantResponse,
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    PartialTranscription,
    TTSInput,
    TokenUsage,
    Transcription,
    VADAudio,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

# â”€â”€ VAD stage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
VADIn: TypeAlias = bytes | tuple[bytes, RuntimeConfig]
VADOut: TypeAlias = VADAudio

# â”€â”€ STT stage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STTIn: TypeAlias = VADAudio
STTOut: TypeAlias = PartialTranscription | Transcription | DirectAssistantResponse

# â”€â”€ LLM stage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLMIn: TypeAlias = GenerateResponseRequest | DirectAssistantRequest
LLMOut: TypeAlias = LLMResponseChunk | TokenUsage | EndOfResponse

# â”€â”€ TTS stage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TTSIn: TypeAlias = TTSInput | EndOfResponse
TTSOut: TypeAlias = bytes | np.ndarray | AudioOutput
