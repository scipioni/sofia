# stt + tts — the GPU half. One image, two commands (sofia-stt / sofia-tts).
#
# NVIDIA and AMD differ by exactly one thing: which PyTorch wheel index we pull
# from. The application code is identical, because torch's ROCm build exposes
# the same torch.cuda API as the CUDA build.
#
#   NVIDIA : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   AMD    : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2
#   CPU    : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
#
# The CUDA/ROCm runtime libraries ship inside those wheels, so a plain slim base
# is enough — the host only needs its driver plus the right container runtime.

# Header source, selected per profile: "with-rocrand" (AMD, via the rocm-ctx
# named context compose.rocm.yaml provides) or "without-rocrand" (empty stub).
# Must be declared before the first FROM to be usable in FROM interpolation.
ARG WITH_ROCRAND_HEADERS=without-rocrand

# --- ROCm headers (AMD only) -------------------------------------------------
# MIOpen JIT-compiles some kernels at first use (the LSTM dropout inside
# Kokoro, among others); those sources #include <rocrand/…> and <hip/…>
# headers the PyTorch ROCm wheel does not ship — first inference dies with
# HIPRTC_ERROR_COMPILATION. The ROCm profile copies the whole include tree
# from the host's matching /opt/rocm through a named build context; every
# other profile gets an empty stub. Header-only, no runtime linkage: the
# compiled kernels link against the libraries inside the wheel. Clang's
# default search (which the JIT uses) covers /usr/local/include.
FROM busybox AS with-rocrand
COPY --from=rocm-ctx include/ /usr/local/include/

FROM busybox AS without-rocrand
WORKDIR /opt/rocm/include/rocrand

FROM ${WITH_ROCRAND_HEADERS} AS rocrand-headers

FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ffmpeg  : how transformers decodes whatever audio container arrives
# espeak-ng: Kokoro's fallback grapheme-to-phoneme for out-of-vocabulary words
# libstdc++-12-dev: on AMD, MIOpen JIT-compiles kernels with the bundled clang
#   at first use; those device sources include C++ standard headers (<utility>
#   …) that slim images do not carry. Harmless on CPU/NVIDIA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg espeak-ng libsndfile1 libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first and on its own layer: it is by far the biggest download, and it
# should not be re-fetched every time an application dependency moves.
RUN uv pip install --system --no-cache --index-url "${TORCH_INDEX_URL}" torch

COPY --from=rocrand-headers /usr/local/include/ /usr/local/include/

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache '.[audio]'

# Kokoro's English grapheme-to-phoneme (misaki) loads this spaCy model at
# startup, and pip does not pull it in. Without it the tts service crashes on
# boot for any en_* voice.
RUN python -m spacy download en_core_web_sm

RUN useradd --create-home --uid 10001 sofia

# Model weights land here. Mount a volume so they survive a rebuild — otherwise
# every image change re-downloads several gigabytes from Hugging Face.
#
# The directory must exist *in the image*, owned by sofia: Docker seeds a fresh
# named volume from the image path it covers, ownership included. Without this
# the volume arrives root-owned and the non-root process cannot write to it.
ENV HF_HOME=/home/sofia/.cache/huggingface \
    SOFIA_MODELS_DIR=/home/sofia/models
# miopen-cache: MIOpen's JIT kernel cache, volume-mounted so each tensor shape
# is compiled once per deployment, not once per boot.
RUN mkdir -p "${HF_HOME}" "${SOFIA_MODELS_DIR}" /home/sofia/.cache/miopen \
    && chown -R 10001:10001 /home/sofia/.cache /home/sofia/models

USER sofia

EXPOSE 8100 8200

CMD ["sofia-stt"]
