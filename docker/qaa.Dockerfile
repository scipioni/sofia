# qaa-agent.service — the brain. Pure Python, no GPU, no model weights.
# Identical on NVIDIA and AMD hosts.
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

# Dependency layer first: source edits then rebuild in seconds, not minutes.
COPY pyproject.toml README.md ./
COPY src ./src
# DEV=true installs editable, for compose.dev.yaml's bind mount + hot reload —
# see docker/audio.Dockerfile's own DEV arg for the full explanation.
ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        uv pip install --system --no-cache -e '.[qaa]'; \
    else \
        uv pip install --system --no-cache '.[qaa]'; \
    fi

RUN useradd --create-home --uid 10001 sofia
USER sofia

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["sofia-qaa"]
