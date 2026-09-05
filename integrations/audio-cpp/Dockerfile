# One isolated candidate: audio.cpp, the copied Voice Studio, and unquantized
# BF16 Base checkpoints in one runtime container.  Build tools never reach the
# final image and the candidate never shares ports or storage with Faster/Groxaxo.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS build

ARG AUDIO_CPP_REPOSITORY=https://github.com/0xShug0/audio.cpp.git
ARG AUDIO_CPP_REF=238ab6a9e321c17de8e120559f57efeedaeb1345
# Docker build stages do not see the host GPU, so CMake otherwise accepts
# nvcc's compiler-default architecture (observed as 52).  This isolated
# candidate is validated for the RTX 3080 Laptop GPU (sm_86); set CUDAARCHS
# before the first configure so CMake caches the intended target.
ARG AUDIO_CPP_CUDA_ARCHITECTURES=86

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates cmake git ninja-build build-essential python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --no-checkout "${AUDIO_CPP_REPOSITORY}" /opt/audio.cpp \
    && cd /opt/audio.cpp \
    && git fetch --depth 1 origin "${AUDIO_CPP_REF}" \
    && git checkout --detach "${AUDIO_CPP_REF}"
COPY integrations/audio-cpp/patches/qwen3-native-pcm-streaming.patch /tmp/qwen3-native-pcm-streaming.patch
WORKDIR /opt/audio.cpp
RUN git apply --check /tmp/qwen3-native-pcm-streaming.patch \
    && git apply /tmp/qwen3-native-pcm-streaming.patch
RUN test -n "$AUDIO_CPP_CUDA_ARCHITECTURES" \
    && CUDAARCHS="$AUDIO_CPP_CUDA_ARCHITECTURES" bash scripts/build_linux.sh --backend cuda --deployment-build \
    --model-set custom --models qwen3_tts --target audiocpp_server

FROM build AS model-stage
RUN python3 -m pip install --no-cache-dir "huggingface_hub>=0.32" \
    && python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", local_dir="/opt/models/Qwen3-TTS-12Hz-0.6B-Base")
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="/opt/models/Qwen3-TTS-12Hz-1.7B-Base")
PY
# Both Base checkpoints contain byte-identical tokenizer weights.  Keep one
# immutable copy and expose it at each model's expected path.
RUN cmp /opt/models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/model.safetensors \
        /opt/models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer/model.safetensors \
    && mv /opt/models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer /opt/models/qwen3-tts-speech-tokenizer \
    && rm -rf /opt/models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer \
    && ln -s ../qwen3-tts-speech-tokenizer /opt/models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
    && ln -s ../qwen3-tts-speech-tokenizer /opt/models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer \
    && rm -rf /root/.cache/huggingface

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_server /usr/local/bin/audiocpp_server
COPY --from=build /opt/audio.cpp/model_specs /opt/audio.cpp/model_specs
COPY --from=build /opt/audio.cpp/assets /opt/audio.cpp/assets
COPY --from=model-stage /opt/models /opt/models
COPY integrations/audio-cpp/qwen3-tts-base-f16.json /config/qwen3-tts-base-f16.json

WORKDIR /opt/voice-studio
COPY web/hf-realtime-voice/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt "gradio>=5,<6"
ARG VOICE_STUDIO_SOURCE_REV=voice-studio-ui-and-hfrt-reliability-20260801
RUN test -n "$VOICE_STUDIO_SOURCE_REV"
COPY web/hf-realtime-voice/ .
COPY integrations/audio-cpp/gradio_voice_studio.py /opt/voice-studio/gradio_voice_studio.py
COPY integrations/audio-cpp/profile_library.py /opt/voice-studio/profile_library.py
COPY integrations/audio-cpp/supervisor.py /opt/audio-cpp-supervisor.py
COPY integrations/audio-cpp/profile_library.py /opt/profile_library.py
COPY integrations/audio-cpp/entrypoint.sh /usr/local/bin/audio-cpp-candidate
RUN chmod +x /usr/local/bin/audio-cpp-candidate

ENV AUDIOCPP_BACKEND=cuda \
    AUDIO_CPP_SERVER_BIN=/usr/local/bin/audiocpp_server
EXPOSE 7860 8080
ENTRYPOINT ["/usr/local/bin/audio-cpp-candidate"]
