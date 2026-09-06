# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Tasks run with [go-task](https://taskfile.dev) (`task`), not make. `Taskfile.yml` loads `.env` via `dotenv`.

```bash
task setup                       # uv venv + editable install with every extra
task check                       # lint + test — what CI runs
task lint                        # ruff check + ruff format --check on src tests
task fmt                         # ruff check --fix + ruff format
task test                        # pytest -q
task test -- -k realtime -v      # anything after `--` is passed through
task config                      # `docker compose config -q` for all three overlays
task up:rocm | up:nvidia | up:cpu   # build and run the stack
task up:rocm:dev | up:nvidia:dev | up:cpu:dev   # same, + compose.dev.yaml hot reload
task down                        # stop
task logs -- s2s                 # follow one service's logs
```

Probes that need no LiveKit room:

```bash
task console                # full voice loop in the terminal (sofia-s2s console)
task ask Q="che ore sono?"  # brain only, over HTTP against localhost:8000
task say T="ciao"           # tts only, writes out.wav
```

A single test file directly: `uv run pytest tests/test_engine.py -v`.
`tests/test_streaming_asr.py` self-skips unless `SOFIA_STT_SHERPA_MODEL_DIR` points at a real
sherpa-onnx model directory.

## Architecture

Four containers; a LiveKit server and an OpenAI-compatible LLM are **provided by the user**, not
shipped here.

```
LiveKit room → s2s (VAD + turn detection, no weights)
                 ├── stt :8100   POST /v1/audio/transcriptions | WS /v1/realtime
                 ├── tts :8200   POST /v1/audio/speech
                 └── qaa-agent :8000  POST /v1/chat/completions → your LLM
```

The load-bearing design decision: **every internal interface is the OpenAI protocol.** `qaa-agent`
*serves* Chat Completions while *consuming* Chat Completions upstream, so s2s wires to it with
plain `livekit.plugins.openai.LLM(base_url="http://qaa-agent:8000/v1")` — no glue code, token
streaming end to end, and any component swappable for a hosted equivalent or drivable with `curl`.
Preserve this when changing anything at a service boundary; a bespoke endpoint would forfeit it.

Consequences worth knowing before editing:

- **Tool calls go both ways.** Tools defined in `qaa/tools.py` run server-side inside the engine
  and are invisible to s2s. Tools declared by the caller (LiveKit `function_tool`s) are passed
  through as an external toolset and handed back for the caller to execute. `SOFIA_QAA_MAX_TOOL_ROUNDS`
  caps the loop at 3, because each round is silence on the call.
- **The agent loop is pydantic-ai's, not ours** (`qaa/engine.py`, major-pinned). The framework owns
  model requests, tool dispatch, streamed fragment folding, retries and the request budget; the
  engine keeps only the protocol glue (OpenAI messages in, three stream events out) and the
  voice-agent policies: the round cap (`UsageLimits`), the token ceiling (`max_tokens`, pinned to
  the legacy wire field for vLLM/llama.cpp/Ollama), and the never-silent failure path. Keep it that
  way — the facade (`TextDelta | ToolCallsDelta | Done`) is what contains framework churn inside one
  container.
- **Wire schemas are `extra="allow"`** (`qaa/schemas.py`). s2s adds `sofia_room` / `sofia_language`
  via `extra_body`; upstream LLMs ignore them. Don't tighten this.
- **qaa-agent is stateless.** LiveKit resends full history each turn; there is no persistence layer.
- **Streaming is the point of `qaa/engine.py`.** Text deltas are emitted the instant they arrive.
  The run uses `end_strategy='graceful'` so text-then-tool-call responses still speak the text *and*
  run the tools. When upstream fails the engine still speaks `UPSTREAM_ERROR_REPLY` —
  silence is the worst failure mode for a voice agent.
- **`stt` serves two backends at once** (`SOFIA_STT_BACKEND=both`): batch Nemotron 3.5 over HTTP
  (punctuated, 40 locales incl. Italian) and streaming over websocket. Streaming itself has two
  engines (`SOFIA_STT_STREAMING_ENGINE`): sherpa-onnx (no Italian, no punctuation, CPU-only) or
  parakeet.cpp/ggml (same Nemotron weights as batch, so punctuated and covering Italian, but needs
  a GPU-capable image — see below). s2s picks streaming vs batch with `SOFIA_S2S_STT_USE_REALTIME`,
  so switching is a restart, not a redeploy. sherpa and batch run well on CPU; parakeet does not.
  See [docs/details.md](docs/details.md#streaming-vs-batch-asr) for the measured trade-off.
- **Turn latency is set by policy, not compute.** `SOFIA_STT_ENDPOINT_SILENCE` (0.8 s) dominates
  once streaming ASR is on — true for both streaming engines: Nemotron has no end-of-utterance
  signal of its own (that belongs only to the separate, English-only
  `nvidia/parakeet_realtime_eou_120m-v1`), so the parakeet engine implements the same kind of
  trailing-silence rule sherpa's own endpoint detection does, rather than a model signal.
- **`tts` streams PCM by default.** `s2s`'s `tts_response_format` defaults to `pcm`:
  `tts_app.py`'s `/v1/audio/speech` yields each Kokoro segment as soon as it's synthesized instead
  of waiting for the whole reply, cutting time-to-first-audio by ~60% on long replies (see
  docs/benchmark.md). `wav`/`flac` requests are unchanged — still a single flush, since a WAV
  header must declare a total length that isn't known until synthesis finishes. `s2s` has its own
  Dockerfile (`docker/s2s.Dockerfile`), separate from `tts`'s — rebuilding `tts` without rebuilding
  `s2s` silently leaves this off.
- **`tts`'s ROCm torch busy-spins one CPU core at idle** — a known, currently-open upstream ROCm
  bug (the pip-bundled HSA runtime's `AsyncEventsLoop` background thread never sleeps after any GPU
  op). `stt`, on the same runtime, doesn't show it. Fixed with `GPU_MAX_HW_QUEUES=1` on the `tts`
  service in compose.yaml — an env var, not an application-code change; see docs/benchmark.md for
  how it was diagnosed and confirmed.
- Audio code **never branches on GPU vendor at the application-code level**: ROCm PyTorch exposes
  the same `torch.cuda` API, so `audio/config.py:resolve_device` covers both. The only NVIDIA/AMD/
  CPU difference is which torch wheel index `docker/audio.Dockerfile` installs — hence `torch` is
  deliberately absent from `pyproject.toml`. The parakeet engine keeps this at the *build*, not
  runtime, level too: `PARAKEET_BACKEND` picks a `cmake` flag (Vulkan by default — one build runs
  unchanged on AMD and NVIDIA; `hip`/`cuda` are recognised but need a different builder base image
  this Dockerfile does not provide), and the Python ctypes binding is identical regardless.

### Layout

One package `src/sofia_galileo/`, four console scripts (`sofia-s2s`, `sofia-qaa`, `sofia-stt`,
`sofia-tts`), three install extras (`s2s`, `qaa`, `audio`) matching the images. `audio` builds one
image serving two commands.

### Configuration

Every service has a `config.py` with a pydantic-settings class and its own env prefix:
`SOFIA_S2S_`, `SOFIA_QAA_`, `SOFIA_STT_`, `SOFIA_TTS_`. Fields map 1:1 to env vars; add a setting
by adding a field, then document it in `.env.example`. `LIVEKIT_*` vars are intentionally *not* in
`S2SSettings` — the livekit-agents CLI reads them from the environment itself.

The system prompt is `qaa/prompts.py`, overridable at runtime via `SOFIA_QAA_SYSTEM_PROMPT_FILE`.

## Conventions

- Ruff, line length 100, `py311`, lint set `E,F,I,UP,B,ASYNC,RUF,BLE,SLF`.
- pytest with `asyncio_mode = "auto"` — async tests need no marker.
- Logging is structlog via `core/logging.py` (`get_logger(__name__)`), JSON by default.
- Comments in this codebase explain *why* a choice was made, often citing a measurement or a
  failure mode. Match that register rather than narrating what the code does.
- Tests target the parts that actually break: the tool loop against a stub upstream over real HTTP,
  and the realtime websocket wire format against a fake recogniser. See
  [docs/details.md](docs/details.md#what-the-tests-actually-protect).
- When creating commits, do **not** add a `Co-Authored-By` trailer.

## OpenSpec

This repo uses OpenSpec (`openspec/`, schema `spec-driven`) with `/opsx:*` slash commands and
matching skills for proposing, applying, and archiving changes. Use them when the user asks for a
structured change workflow; specs live in `openspec/specs/`, in-flight changes in
`openspec/changes/`.
