# sofia-galileo

A vocal agent that joins a LiveKit room, listens to people, and talks back.

Bring your own OpenAI-compatible LLM. Everything else runs in four containers,
on NVIDIA, AMD, or plain CPU.

> Design rationale, measurements and production notes: **[docs/details.md](docs/details.md)**
> Latency and throughput benchmarks on the reference AMD deployment: **[docs/benchmark.md](docs/benchmark.md)**

## Schema

```
        ┌──────────────────────────────┐
        │        LiveKit room          │   a human, on WebRTC
        └───────────────┬──────────────┘
                        │ audio tracks
                        ▼
   ┌────────────────────────────────────────┐
   │              s2s.service               │   no GPU, no weights
   │  Silero VAD → semantic turn detection  │
   └───┬───────────────┬────────────────┬───┘
       │               │                │
       │ WS  /v1/      │ POST           │ POST
       │  realtime     │ /v1/chat/      │ /v1/audio/
       │ (or batch     │ completions    │ speech
       │  POST)        │                │
       ▼               ▼                ▼
  ┌─────────┐   ┌──────────────┐   ┌─────────┐
  │   stt   │   │  qaa-agent   │   │   tts   │
  │ :8100   │   │    :8000     │   │  :8200  │
  └─────────┘   └──────┬───────┘   └─────────┘
                       │ POST /v1/chat/completions
                       ▼
              ┌──────────────────┐
              │  your LLM        │  vLLM / llama.cpp / Ollama /
              │ (OpenAI-compat)  │  TGI / a hosted API
              └──────────────────┘
```

Every internal interface is an **OpenAI-compatible API**. Nothing here speaks a
bespoke protocol, so any component can be swapped for a hosted equivalent — or
driven with `curl` — without touching the others.

| Service | Endpoint | Contract |
|---|---|---|
| `qaa-agent` | `POST /v1/chat/completions` | OpenAI Chat Completions, SSE streaming, tool calls |
| | `GET /v1/models`, `GET /healthz` | |
| `stt` | `POST /v1/audio/transcriptions` | OpenAI transcriptions (multipart) → `{"text": …}` |
| | `WS /v1/realtime` | OpenAI realtime transcription: 24 kHz PCM16 in, incremental deltas out |
| | `GET /healthz` | |
| `tts` | `POST /v1/audio/speech` | OpenAI speech → `wav` / `pcm` / `flac` @ 24 kHz |
| | `GET /healthz` | |
| `s2s` | `GET :8081/` | livekit-agents worker health |

`qaa-agent` is the interesting one: it *serves* the OpenAI protocol while
*consuming* it upstream. s2s therefore treats the brain as if it were the LLM —
`livekit.plugins.openai.LLM(base_url="http://qaa-agent:8000/v1")` — which is why
there is no glue code between them, and why token streaming reaches the
synthesiser sentence by sentence.

```jsonc
// s2s → qaa-agent, per turn. Standard OpenAI, plus optional context.
{ "model": "sofia-qaa", "stream": true,
  "messages": [{"role": "user", "content": "che ore sono?"}],
  "sofia_room": "lobby", "sofia_language": "it" }   // extra_body, ignored upstream
```

## Dependencies

**You provide**

| | Why | Configured by |
|---|---|---|
| A LiveKit server or LiveKit Cloud | where the conversation happens | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| An OpenAI-compatible LLM endpoint | the actual reasoning | `SOFIA_QAA_LLM_BASE_URL`, `SOFIA_QAA_LLM_MODEL` |

**Host**

- Docker with Compose v2
- NVIDIA: [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- AMD: ROCm kernel driver, `/dev/kfd` + `/dev/dri`
- Neither, for the CPU profile — every model here runs on CPU
- [go-task](https://taskfile.dev) for the `task` commands below
- Python 3.11+ and [uv](https://docs.astral.sh/uv/), for local development only

**Python** — one package, four entrypoints, installed per image via extras:

| Extra | Used by | Key packages |
|---|---|---|
| `s2s` | s2s | `livekit-agents[openai,silero,turn-detector]`, `openai` |
| `qaa` | qaa-agent | `fastapi`, `uvicorn`, `openai` |
| `audio` | stt, tts | `transformers>=5.13`, `sherpa-onnx`, `kokoro`, `librosa`, `soundfile` |

`torch` is deliberately *not* in `pyproject.toml`: the Dockerfile installs the
`cu126`, `rocm7.2` or `cpu` wheel first, which is the only difference between
GPU targets. System packages `ffmpeg`, `espeak-ng` and `libsndfile1` come from
the audio image.

**Models** — fetched on first boot into the `hf-cache` / `models` volumes, so it
happens once:

| Role | Default | Size |
|---|---|---|
| Batch ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` | ~1.3 GB |
| Streaming ASR | sherpa-onnx streaming zipformer (en) | ~310 MB |
| TTS | Kokoro-82M | ~330 MB |
| VAD + turn detection | Silero + LiveKit multilingual | baked into the s2s image |

## Quick start

Tasks are run with [go-task](https://taskfile.dev) (`task`), not make.

```bash
cp .env.example .env      # fill in LIVEKIT_* and your LLM endpoint
task up:rocm              # or: task up:nvidia   (or: task up:cpu)
```

Then connect a human to a room — the LiveKit Agents playground, or your own
client. The worker joins every room by default.

No Docker, no LiveKit, no room:

```bash
task setup
task console                # full voice loop in your terminal
task ask Q="who are you?"   # just the brain, over HTTP
task say T="ciao"           # just the voice, to out.wav
```

## Configuration

All environment variables; `.env.example` documents the full set. The ones that
change behaviour most:

| Variable | Default | What it does |
|---|---|---|
| `SOFIA_LANGUAGE` | `en` | Mapped to an ASR locale and given to the brain as context |
| `SOFIA_STT_STREAMING` | `false` | `true` = websocket streaming ASR; `false` = batch ([trade-off](docs/details.md#streaming-vs-batch-asr)) |
| `SOFIA_STT_ENDPOINT_SILENCE` | `0.8` | Trailing silence (s) that ends a turn — the main latency dial |
| `SOFIA_QAA_MAX_TOKENS` | `320` | Ceiling on reply length; callers may ask for less, never more |
| `SOFIA_TTS_VOICE` | `af_heart` | Kokoro voice; first letter picks the language |
| `SOFIA_AUDIO_DEVICE` | `auto` | `auto` / `cuda` / `cpu` (`cuda` also means the AMD GPU) |

The system prompt is `src/sofia_galileo/qaa/prompts.py`, overridable at runtime
with `SOFIA_QAA_SYSTEM_PROMPT_FILE`.

## Layout

```
src/sofia_galileo/
├── core/logging.py      structlog setup shared by all services
├── s2s/agent.py         the LiveKit worker, in one file
├── qaa/
│   ├── app.py           FastAPI + SSE, the OpenAI-compatible surface
│   ├── engine.py        the reasoning + tool loop
│   ├── tools.py         server-side tool registry
│   ├── prompts.py       the system prompt
│   └── schemas.py       OpenAI wire types
└── audio/               one image, two commands (sofia-stt / sofia-tts)
    ├── stt_app.py       HTTP + websocket routes
    ├── batch.py         Nemotron / Whisper engines, locale mapping, decoding
    ├── realtime.py      OpenAI realtime-transcription websocket protocol
    ├── streaming.py     sherpa-onnx recogniser + model fetch
    └── tts_app.py       Kokoro
```

## Development

```bash
task setup   # venv with every extra
task check   # lint + test, what CI would run
task fmt     # ruff --fix + format
task config  # validate every compose file combination
```

Bare `task` lists everything. Arguments after `--` pass through:
`task test -- -k realtime -v`, `task logs -- s2s`.

What the tests deliberately cover, and why, is in
[docs/details.md](docs/details.md#what-the-tests-actually-protect).
