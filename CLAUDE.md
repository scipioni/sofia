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

- **Tool calls go both ways.** Tools registered in `qaa/tools.py` run server-side inside the engine
  and are invisible to s2s. Tools declared by the caller (LiveKit `function_tool`s) are passed
  through and handed back for the caller to execute — `qaa/engine.py` dispatches on ours-vs-theirs.
  `SOFIA_QAA_MAX_TOOL_ROUNDS` caps the loop at 3, because each round is silence on the call.
- **Wire schemas are `extra="allow"`** (`qaa/schemas.py`). s2s adds `sofia_room` / `sofia_language`
  via `extra_body`; upstream LLMs ignore them. Don't tighten this.
- **qaa-agent is stateless.** LiveKit resends full history each turn; there is no persistence layer.
- **Streaming is the point of `qaa/engine.py`.** Text deltas are emitted the instant they arrive.
  Tool-call fragments arrive split across SSE chunks and are folded by index in
  `_accumulate_tool_calls`. When upstream fails the engine still speaks `UPSTREAM_ERROR_REPLY` —
  silence is the worst failure mode for a voice agent.
- **`stt` serves two backends at once** (`SOFIA_STT_BACKEND=both`): batch Nemotron 3.5 over HTTP
  (punctuated, 40 locales incl. Italian) and streaming sherpa-onnx over websocket (interim
  transcripts, no Italian, no punctuation). s2s picks with `SOFIA_S2S_STT_USE_REALTIME`, so
  switching is a restart, not a redeploy. Both run well on CPU. See
  [docs/details.md](docs/details.md#streaming-vs-batch-asr) for the measured trade-off.
- **Turn latency is set by policy, not compute.** `SOFIA_STT_ENDPOINT_SILENCE` (0.8 s) dominates
  once streaming ASR is on.
- Audio code **never branches on GPU vendor**: ROCm PyTorch exposes the same `torch.cuda` API, so
  `audio/config.py:resolve_device` covers both. The only NVIDIA/AMD/CPU difference is which torch
  wheel index `docker/audio.Dockerfile` installs — hence `torch` is deliberately absent from
  `pyproject.toml`.

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
