# s2s.service — the LiveKit worker. Pure Python, no GPU.
# The only weights it carries are the tiny VAD and turn-detector ONNX models,
# baked in at build time so a cold start never blocks on a download.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# DEV=true installs editable, for compose.dev.yaml's bind mount + `sofia-s2s
# dev`'s own hot reload (livekit-agents' built-in watchfiles-based watcher) —
# see docker/audio.Dockerfile's own DEV arg for the full explanation.
ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        uv pip install --system --no-cache -e '.[s2s]'; \
    else \
        uv pip install --system --no-cache '.[s2s]'; \
    fi

RUN useradd --create-home --uid 10001 sofia
USER sofia
ENV HF_HOME=/home/sofia/.cache/huggingface

# Silero VAD + the multilingual turn-detector model.
RUN sofia-s2s download-files

# The worker's own health endpoint (livekit-agents serves it on 8081).
EXPOSE 8081

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8081/ || exit 1

CMD ["sofia-s2s", "start"]
