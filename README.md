# sofia-galileo

A vocal agent that joins a LiveKit room, listens to people, and talks back.

Four containers, two of which are the ones that matter:

- **`s2s.service`** — the LiveKit worker. Owns audio in and audio out: voice
  activity detection, end-of-turn detection, speech-to-text, speech synthesis.
  Contains no LLM logic and no model weights.
- **`qaa-agent.service`** — the brain. Owns the system prompt, the conversation
  policy, and the tools. Exposes an **OpenAI-compatible** API and calls your own
  OpenAI-compatible LLM upstream.
- `stt` / `tts` — OpenAI-compatible shims over ASR and Kokoro. `stt` serves two
  backends: **streaming** sherpa-onnx over a websocket, and **batch** Whisper
  over HTTP. See [Streaming vs batch ASR](#streaming-vs-batch-asr).

## How the pieces talk

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
  │ sherpa/ │   │  the brain   │   │ Kokoro  │
  │ Whisper │   └──────┬───────┘   │  (GPU)  │
  └─────────┘          │           └─────────┘
                       │ POST /v1/chat/completions
                       ▼
              ┌──────────────────┐
              │  your LLM        │  vLLM / llama.cpp / Ollama /
              │ (OpenAI-compat)  │  TGI / a hosted API
              └──────────────────┘
```

### Why qaa-agent speaks OpenAI, not a custom protocol

The obvious design is a bespoke `POST /turn {transcript} → {reply}` API between
the two services. Speaking the OpenAI Chat Completions protocol instead buys
several things for free:

- **No glue code in s2s.** It points `livekit.plugins.openai.LLM` at
  `http://qaa-agent:8000/v1` and is done.
- **Token streaming end to end.** s2s starts synthesising speech on the first
  sentence rather than waiting for the whole answer. In a spoken conversation
  that is the single largest latency win available.
- **A/B debugging.** Point s2s straight at the raw LLM by changing one env var.
  If the conversation is still bad, the problem is not qaa-agent.
- **Testable without LiveKit.** `curl`, the OpenAI SDK, or any OpenAI client
  drives the brain directly. `make ask Q="what time is it?"`.
- **Tool calls work in both directions.** Tools defined *inside* qaa-agent run
  server-side and are invisible to s2s. LiveKit `function_tool`s declared in s2s
  are passed through and handed back for s2s to execute — exactly as a real
  OpenAI endpoint behaves.

## Streaming vs batch ASR

The `stt` service serves both, and s2s picks one with a single flag. This is the
most consequential setting in the project.

```bash
SOFIA_STT_STREAMING=true    # websocket, sherpa-onnx  — decodes while they talk
SOFIA_STT_STREAMING=false   # HTTP, Whisper           — decodes after they stop
```

|  | streaming (sherpa-onnx) | batch (Whisper) |
|---|---|---|
| Transcript ready | while they are still speaking | 200–500 ms *after* they stop |
| Punctuation / casing | none — `hello this is sofia` | full — `Hello, this is Sofia.` |
| Accuracy | lower (heard "to day" for "today") | high |
| Languages | en, fr, de, es, ru, zh, ko… **no Italian** | ~100, including Italian |
| Hardware | CPU, ~0.06 RTF (≈16 concurrent per core) | GPU strongly preferred |
| Interim transcripts | yes — the turn detector can use them | no |

Measured on the streaming path, pacing 24 kHz audio through the websocket in
50 ms frames exactly as LiveKit does:

```
t=1.13s  delta 'Hello'          ← first text, 1.1s into a 3.4s utterance
t=3.34s  delta ' to day'        ← transcript complete before the audio ends
t=4.35s  completed              ← 922 ms after speech ended
         └─ 800 ms of that is the endpoint silence rule, not compute
```

So the transcription itself costs ~120 ms after end of speech instead of
200–500 ms, and the remaining wait is a **policy** you choose:
`SOFIA_STT_ENDPOINT_SILENCE` (default 0.8 s) is how long someone must be quiet
before their turn is considered over. Lower feels snappier and cuts people off;
higher feels polite and sluggish. **With streaming enabled this rule, not
Whisper, is what sets your turn latency** — it is the first thing to tune, and
LiveKit's own endpointing runs on top of it.

Notes on the trade-off:

- **CPU is the right answer here**, not a compromise. At 0.06 RTF a single core
  handles ~16 concurrent conversations, and it sidesteps ROCm entirely — the
  sherpa-onnx wheels ship CUDA support only.
- **No streaming Italian model exists.** If you need Italian, use batch Whisper.
  Alternatives that do exist are listed in `.env.example`.
- **"Sofia" is heard as "sophia".** Point `SOFIA_STT_SHERPA_HOTWORDS_FILE` at a
  file of product and person names, one per line, to bias decoding towards them.
- The model (~310 MB) downloads on first boot into the `models` volume. Switch
  languages by changing `SOFIA_STT_SHERPA_MODEL_URL` and deleting that volume.
- Run `SOFIA_STT_BACKEND=both` (the default) to keep both loaded, so flipping
  the flag is a restart rather than a redeploy. Set it to one backend to halve
  the memory.

## Quick start

```bash
cp .env.example .env      # fill in LIVEKIT_* and your LLM endpoint
make up-nvidia            # or: make up-rocm   (or: make up-cpu)
```

Then connect a human to a room, e.g. with the LiveKit Agents playground or your
own client. The worker joins every room by default.

To iterate on the agent without any of Docker or LiveKit:

```bash
make setup
make console              # full voice loop in your terminal
make ask Q="who are you?" # just the brain, over HTTP
```

## GPU targets

Application code is identical on both platforms — PyTorch's ROCm build exposes
the same `torch.cuda` API as the CUDA build, so `stt` and `tts` never branch on
vendor. The images differ only in which PyTorch wheel index they install from:

| Target | Overlay | Wheel index | Host requirement |
|---|---|---|---|
| NVIDIA | `compose.nvidia.yaml` | `…/whl/cu124` | NVIDIA Container Toolkit |
| AMD | `compose.rocm.yaml` | `…/whl/rocm6.2` | ROCm kernel driver, `/dev/kfd` + `/dev/dri` |
| CPU | *(none)* | `…/whl/cpu` | nothing |

`s2s` and `qaa-agent` build once and run anywhere; they hold no weights.

**AMD notes.** `group_add` in the overlay uses group *names*, which only resolve
if `video`/`render` exist inside the container — if the GPU comes back invisible,
run `getent group render video` on the host and substitute the numeric GIDs. For
consumer cards ROCm does not officially list, set `HSA_OVERRIDE_GFX_VERSION`
(`10.3.0` for RDNA2, `11.0.0` for RDNA3).

## Configuration

Everything is environment variables; see `.env.example` for the full set.

| Variable | What it does |
|---|---|
| `LIVEKIT_URL` / `_API_KEY` / `_API_SECRET` | Where the worker connects |
| `SOFIA_QAA_LLM_BASE_URL` | Your OpenAI-compatible LLM (must end in `/v1`) |
| `SOFIA_QAA_LLM_MODEL` | Model name to request upstream |
| `SOFIA_QAA_MAX_TOKENS` | Hard ceiling on reply length; a caller can ask for less, never more |
| `SOFIA_LANGUAGE` | Passed to Whisper and given to the brain as context |
| `SOFIA_STT_STREAMING` | `true` for websocket streaming ASR, `false` for batch Whisper |
| `SOFIA_STT_BACKEND` | Which backends `stt` loads: `batch`, `streaming` or `both` |
| `SOFIA_STT_ENDPOINT_SILENCE` | Trailing silence (s) that ends a turn — your main latency dial |
| `SOFIA_STT_MODEL_ID` | Whisper checkpoint (batch backend) |
| `SOFIA_STT_SHERPA_MODEL_URL` | Streaming model release asset (streaming backend) |
| `SOFIA_TTS_VOICE` | Kokoro voice; its first letter picks the language pipeline |
| `SOFIA_AUDIO_DEVICE` | `auto` / `cuda` / `cpu` (`cuda` also means the AMD GPU) |

The system prompt lives in `src/sofia_galileo/qaa/prompts.py`, and can be
overridden at runtime with `SOFIA_QAA_SYSTEM_PROMPT_FILE` without a rebuild.

## Adding a tool

Server-side tools live in `src/sofia_galileo/qaa/tools.py`. A tool is an async
function that returns a string and never raises:

```python
@registry.register(
    name="lookup_booking",
    description="Look up a booking by its reference number.",
    parameters={
        "type": "object",
        "properties": {"reference": {"type": "string"}},
        "required": ["reference"],
    },
)
async def lookup_booking(reference: str) -> str:
    ...
    return "Booked for two people on Friday at eight."
```

Keep the tool surface small. Every tool round is an extra round trip to the LLM
that the person on the call spends listening to silence — `SOFIA_QAA_MAX_TOOL_ROUNDS`
caps it at 3 for that reason.

## Layout

```
src/sofia_galileo/
├── core/logging.py      structlog setup shared by all services
├── s2s/                 LiveKit worker (agent.py is the whole thing)
├── qaa/
│   ├── app.py           FastAPI + SSE, the OpenAI-compatible surface
│   ├── engine.py        the reasoning + tool loop
│   ├── tools.py         server-side tool registry
│   ├── prompts.py       the system prompt
│   └── schemas.py       OpenAI wire types
└── audio/               one image, two commands (sofia-stt / sofia-tts)
    ├── stt_app.py       batch Whisper + the realtime route
    ├── realtime.py      OpenAI realtime-transcription websocket protocol
    ├── streaming.py     sherpa-onnx recogniser + model fetch
    └── tts_app.py       Kokoro
```

## Development

```bash
make setup   # venv with every extra
make test    # pytest
make lint    # ruff
make fmt     # ruff --fix + format
```

`tests/test_engine.py` drives the tool loop against a stub upstream served over
real HTTP, because that loop — tool-call fragments arriving split across SSE
chunks, ours-versus-theirs dispatch, round limits — is where the bodies are
buried.

`tests/test_realtime_ws.py` checks the realtime websocket against a fake
recogniser: the wire format is what actually breaks in integration (deltas must
be *incremental*, one `item_id` per utterance), and testing it needs no model.

`tests/test_streaming_asr.py` exercises the real sherpa-onnx model and is
skipped unless you point it at one:

```bash
SOFIA_STT_SHERPA_MODEL_DIR=/path/to/sherpa-onnx-streaming-zipformer-en-... \
    uv run pytest tests/test_streaming_asr.py -v
```

## Things to know before production

- **Latency budget.** The interesting numbers are per-stage: STT, LLM
  time-to-first-token, TTS first byte. `s2s` logs all three via the
  livekit-agents metrics collector. With streaming ASR on, STT effectively
  disappears from the budget and LLM TTFT dominates — tune that next.
- **This is a cascade, not a speech-to-speech model.** Despite the service name,
  audio is transcribed, reasoned over as text, and re-synthesised. A true S2S
  model (OpenAI Realtime, Moshi, Qwen-Omni) reaches ~300 ms and full duplex, but
  the model *is* the LLM — incompatible with bringing your own.
- **First boot is slow.** `stt` downloads several gigabytes of Whisper weights.
  The healthcheck allows ten minutes for it; the `hf-cache` volume means it
  happens once.
- **No conversation persistence.** qaa-agent is stateless — LiveKit resends the
  full history each turn. Long calls will eventually need history trimming or
  summarisation.
- **Non-English voices.** Kokoro phonemises English through misaki + spaCy
  (`en_core_web_sm`, installed in the image) and everything else through
  espeak-ng (also installed). Switching `SOFIA_TTS_VOICE` to e.g. `if_sara`
  should just work, but listen to it before you ship it — quality varies by
  language far more than the model card suggests.
- **`stt` and `tts` are unauthenticated** on the compose network. Do not expose
  their ports beyond it.
