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

# Compute backend for the streaming ASR engine (parakeet.cpp / ggml), chosen
# independently of TORCH_INDEX_URL above: Vulkan runs the same binary on AMD
# and NVIDIA alike (see design.md D1 in add-parakeet-streaming-asr), so it is
# the default even though the host here is ROCm. HIP and CUDA are recognised
# values, but ggml's HIP/CUDA backends need hipcc/nvcc at *build* time, not
# just a runtime library — wiring those in means swapping this stage's builder
# base for a ROCm or CUDA devel image, which this change does not do. Passing
# PARAKEET_BACKEND=hip or =cuda today will fail the build with a clear cmake
# error (no such compiler) rather than silently falling back to something else.
ARG PARAKEET_BACKEND=vulkan
ARG PARAKEET_COMMIT=e75de9b6b9b688fd293aa22f7e27aa724ea286f8

# --- parakeet.cpp (streaming ASR engine) builder ----------------------------
# Built from source because upstream ships no prebuilt libparakeet.so; the
# GGUF model weights it loads at runtime ARE prebuilt (see ensure_model() in
# audio/streaming.py) so this stage never touches model weights, only ~2 MB of
# C++ source plus the vendored ggml submodule.
FROM python:3.12-slim AS parakeet-builder
ARG PARAKEET_BACKEND
ARG PARAKEET_COMMIT

# glslc/glslang-tools/spirv-headers/libvulkan-dev: Vulkan backend only, but
# installed unconditionally — they are small and PARAKEET_BACKEND=cpu still
# needs cmake/build-essential/git/ninja-build regardless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build build-essential \
        glslc glslang-tools spirv-headers libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --recursive https://github.com/mudler/parakeet.cpp . \
    && git checkout "${PARAKEET_COMMIT}"

# One cmake bool per backend name; PARAKEET_SHARED=ON is what makes cmake
# build libparakeet.so instead of a static libparakeet.a (ctypes needs the
# former). GGML_NATIVE=OFF: this image is built once and run on whatever host
# pulls it, so it must not bake in the *build* host's ISA extensions.
RUN case "${PARAKEET_BACKEND}" in \
        vulkan) BACKEND_FLAG="-DPARAKEET_GGML_VULKAN=ON" ;; \
        cpu)    BACKEND_FLAG="" ;; \
        hip)    BACKEND_FLAG="-DPARAKEET_GGML_HIP=ON" ;; \
        cuda)   BACKEND_FLAG="-DPARAKEET_GGML_CUDA=ON" ;; \
        *) echo "PARAKEET_BACKEND must be vulkan, cpu, hip or cuda; got '${PARAKEET_BACKEND}'" >&2; exit 1 ;; \
    esac \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DPARAKEET_SHARED=ON \
        -DGGML_NATIVE=OFF \
        ${BACKEND_FLAG} \
    && cmake --build build --target parakeet -j"$(nproc)"

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
# The final stage's COPY --from=rocrand-headers /usr/local/include/ ... needs
# this path to exist here even when there's nothing to copy — a COPY whose
# source is entirely absent (not just empty) is a hard build error, not a
# no-op.
RUN mkdir -p /usr/local/include

FROM ${WITH_ROCRAND_HEADERS} AS rocrand-headers

FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG WITH_ROCRAND_HEADERS

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ffmpeg  : how transformers decodes whatever audio container arrives
# espeak-ng: Kokoro's fallback grapheme-to-phoneme for out-of-vocabulary words
# libvulkan1/mesa-vulkan-drivers: the Vulkan loader plus Mesa's RADV/lavapipe
#   ICDs, needed at runtime by libggml-vulkan.so regardless of which vendor's
#   GPU is behind /dev/dri — NVIDIA hosts get their own ICD from the driver the
#   container runtime mounts in (see NVIDIA_DRIVER_CAPABILITIES in
#   compose.nvidia.yaml), Mesa's own ICD is simply unused there.
# libgomp1: ggml's CPU backend is OpenMP-parallel even when Vulkan does the
#   matmuls — the mel front end and tokenizer run on CPU regardless of backend.
#
# libstdc++-12-dev is added conditionally below, gated on the same
# WITH_ROCRAND_HEADERS build arg as the rocrand include tree above: on AMD,
# MIOpen JIT-compiles kernels with the bundled clang at first use, and those
# device sources include C++ standard headers (<utility> …) that slim images
# do not carry. It is a -dev package (hundreds of MB), not a runtime lib, so
# CPU/NVIDIA builds — which never touch MIOpen's JIT path — skip it rather
# than carrying it "harmlessly."
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg espeak-ng libsndfile1 \
        libvulkan1 mesa-vulkan-drivers libgomp1 \
        $( [ "$WITH_ROCRAND_HEADERS" = "with-rocrand" ] && echo libstdc++-12-dev ) \
    && rm -rf /var/lib/apt/lists/*

# parakeet.cpp runtime: libparakeet.so plus the ggml shared libraries it links
# against (built separately, not statically linked in). Streaming ASR only —
# tts never touches these, but the image is shared between both commands. The
# 0.13.0 in these filenames is the vendored ggml submodule's version at
# PARAKEET_COMMIT above, not something to track independently — it moves only
# when that pinned commit changes.
COPY --from=parakeet-builder /src/build/libparakeet.so /usr/local/lib/
COPY --from=parakeet-builder /src/build/third_party/ggml/src/libggml.so.0.13.0 /usr/local/lib/libggml.so.0
COPY --from=parakeet-builder /src/build/third_party/ggml/src/libggml-cpu.so.0.13.0 /usr/local/lib/libggml-cpu.so.0
COPY --from=parakeet-builder /src/build/third_party/ggml/src/libggml-base.so.0.13.0 /usr/local/lib/libggml-base.so.0
COPY --from=parakeet-builder /src/build/third_party/ggml/src/ggml-vulkan/libggml-vulkan.so.0.13.0 /usr/local/lib/libggml-vulkan.so.0
RUN ldconfig

WORKDIR /app

# torch first and on its own layer: it is by far the biggest download, and it
# should not be re-fetched every time an application dependency moves.
RUN uv pip install --system --no-cache --index-url "${TORCH_INDEX_URL}" torch

COPY --from=rocrand-headers /usr/local/include/ /usr/local/include/

COPY pyproject.toml README.md ./
COPY src ./src
# DEV=true installs editable instead: the package's .pth then points at this
# same /app path, so a compose.dev.yaml bind mount of ./src over /app/src
# later is what actually gets (re-)imported — including by uvicorn's own
# reload watcher (SttSettings.reload / TtsSettings.reload) — with no reinstall
# needed at container start. False (default) is the production, immutable
# install this image has always done.
ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        uv pip install --system --no-cache -e '.[audio]'; \
    else \
        uv pip install --system --no-cache '.[audio]'; \
    fi

# Kokoro's English grapheme-to-phoneme (misaki) loads this spaCy model at
# startup, and pip does not pull it in. Without it the tts service crashes on
# boot for any en_* voice.
RUN python -m spacy download en_core_web_sm

RUN useradd --create-home --uid 10001 sofia

# Model weights land here. Mount something so they survive a rebuild —
# otherwise every image change re-downloads several gigabytes from Hugging
# Face. compose.yaml bind-mounts host directories here (browseable, path set
# via SOFIA_MODELS_HOST_DIR / SOFIA_HF_CACHE_HOST_DIR in .env; see Taskfile.yml
# `_up` for why those need creating and opening up before first use — a bind
# mount does not inherit the ownership below the way a fresh named volume
# would).
#
# The directory must exist *in the image* regardless, owned by sofia: this is
# what makes a plain named volume (the miopen one below, or hf-cache/models if
# you point compose at one instead of a host path) come up owned by sofia
# rather than root — Docker seeds a fresh named volume from the image path it
# covers, ownership included, but has nothing to seed from for a bind mount.
ENV HF_HOME=/home/sofia/.cache/huggingface \
    SOFIA_MODELS_DIR=/home/sofia/models
# miopen-cache: MIOpen's JIT kernel cache, volume-mounted so each tensor shape
# is compiled once per deployment, not once per boot.
RUN mkdir -p "${HF_HOME}" "${SOFIA_MODELS_DIR}" /home/sofia/.cache/miopen \
    && chown -R 10001:10001 /home/sofia/.cache /home/sofia/models

USER sofia

EXPOSE 8100 8200

CMD ["sofia-stt"]
