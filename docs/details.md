# sofia-galileo — details

The long version: why things are built the way they are, what was measured, and
what to know before production. For setup and the service contracts, see the
[README](../README.md).

Four containers, two of which are the ones that matter:

- **`s2s.service`** — the LiveKit worker. Owns audio in and audio out: voice
  activity detection, end-of-turn detection, speech-to-text, speech synthesis.
  Contains no LLM logic and no model weights.
- **`qaa-agent.service`** — the brain. Owns the system prompt, the conversation
  policy, and the tools. Exposes an **OpenAI-compatible** API and calls your own
  OpenAI-compatible LLM upstream.
- `stt` / `tts` — OpenAI-compatible shims over ASR and Kokoro. `stt` serves
  **batch** Nemotron 3.5 (or Whisper) over HTTP, and **streaming** over a
  websocket — sherpa-onnx by default, or Nemotron again via parakeet.cpp/ggml
  for deployments that need Italian while streaming. See
  [Streaming vs batch ASR](#streaming-vs-batch-asr). `tts` streams speech back
  as Kokoro produces it (`response_format=pcm`, `s2s`'s default) rather than
  synthesising the whole reply first — see [docs/benchmark.md](benchmark.md).

## Why qaa-agent speaks OpenAI, not a custom protocol

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
  drives the brain directly. `task ask Q="what time is it?"`.
- **Tool calls work in both directions.** Tools defined *inside* qaa-agent run
  server-side and are invisible to s2s. LiveKit `function_tool`s declared in s2s
  are passed through and handed back for s2s to execute — exactly as a real
  OpenAI endpoint behaves.

## Streaming vs batch ASR

The `stt` service serves both, and s2s picks one with a single flag. This is the
most consequential setting in the project. Streaming itself has two engines —
`SOFIA_STT_STREAMING_ENGINE` — because no single one covers every language with
punctuation on a CPU.

```bash
SOFIA_STT_STREAMING=true    # websocket — sherpa-onnx or parakeet, see below
SOFIA_STT_STREAMING=false   # HTTP, Nemotron 3.5      — decodes after they stop
```

|  | streaming (sherpa-onnx) | streaming (parakeet.cpp) | batch (Nemotron 3.5) |
|---|---|---|---|
| Transcript ready | while they are still speaking | while they are still speaking | ~0.1× audio length after they stop |
| Punctuation / casing | none — `hello this is sofia` | full — `Hello, this is Sofia.` | full — `Hello, this is Sofia.` |
| Accuracy | lower (heard "to day" for "today") | same model as batch (below) | high (4.25% WER Italian, FLEURS) |
| Languages | en, fr, de, es, ru, zh, ko… **no Italian** | same 40 locales as batch, **including `it-IT`** | 40 locales **including `it-IT`** |
| Hardware | CPU, ~0.06 RTF | **GPU** (Vulkan or CUDA), RTF ~0.10 measured on an RX 9060 XT | CPU, ~0.08 RTF — GPU optional |
| Turn boundary | model's own endpoint detection | its own trailing-silence rule (no model signal — see below) | n/a |
| Interim transcripts | yes — the turn detector can use them | yes | no |

parakeet.cpp runs the same Nemotron weights sherpa can't stream and batch
already uses, just through a different (ggml) runtime — so it inherits batch's
language coverage and accuracy, at the cost of needing a GPU-capable
`docker/audio.Dockerfile` build. See `openspec/changes/add-parakeet-streaming-asr/`
(design.md, especially D6) for the full reasoning, including a finding that
mattered enough to change the design: Nemotron has **no end-of-utterance signal
of its own** — that capability belongs only to the separate, English-only
`nvidia/parakeet_realtime_eou_120m-v1` model — so the parakeet engine
implements its own trailing-silence rule (`SOFIA_STT_PARAKEET_ENDPOINT_SILENCE`,
same 0.8 s default as sherpa's rule2) rather than relying on the model for it.

Because Nemotron is so fast, batch is no longer the slow option it was with
Whisper. Measured on one CPU core, same 6.6 s Italian clip, same container:

```
Nemotron 3.5 ASR  (638M)   737 ms     ← the default
whisper-small     (244M)  2017 ms     ← 2.7× slower at 2.6× fewer params
```

Both transcribed it correctly. Streaming still wins on *interim* transcripts,
which the semantic turn detector can use — but if you only care about latency,
batch Nemotron is now within a couple of hundred milliseconds of it.

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

- **CPU is the right answer for sherpa and batch**, not a compromise — neither
  needs a GPU, which sidesteps ROCm entirely (the sherpa-onnx wheels ship CUDA
  only, and Nemotron is small enough not to care on a CPU in batch mode).
  Streaming Nemotron via parakeet.cpp is the exception: it needs a GPU to be
  worth running over batch — see [GPU targets](#gpu-targets).
- **For Italian while streaming, use `SOFIA_STT_STREAMING_ENGINE=parakeet`**
  on a GPU-capable deployment, or batch on a CPU-only one. No streaming Italian
  model exists in sherpa-onnx — it covers en/fr/de/es/ru/zh/ko — and the
  multilingual models that do include Italian were offline-only until
  parakeet.cpp made Nemotron's own streaming mode usable outside NeMo.
- **Nemotron takes locales, not language codes.** `SOFIA_LANGUAGE=it` is mapped
  to `it-IT` for you; unknown codes fall back to `auto` rather than to a guessed
  locale, since a wrong locale quietly wrecks accuracy. Auto-detection measures
  within ~0.1% WER of naming the language, so the fallback is cheap.
- **"Sofia" is heard as "sophia".** Point `SOFIA_STT_SHERPA_HOTWORDS_FILE` at a
  file of product and person names, one per line, to bias decoding towards them.
- The model (~310 MB) downloads on first boot into the `models` volume. Switch
  languages by changing `SOFIA_STT_SHERPA_MODEL_URL` and deleting that volume.
- Run `SOFIA_STT_BACKEND=both` (the default) to keep both loaded, so flipping
  the flag is a restart rather than a redeploy. Set it to one backend to halve
  the memory.

## GPU targets

Application code is identical on both platforms — PyTorch's ROCm build exposes
the same `torch.cuda` API as the CUDA build, so `stt` and `tts` never branch on
vendor for torch. The images differ only in which PyTorch wheel index they
install from:

| Target | Overlay | Wheel index | Host requirement |
|---|---|---|---|
| NVIDIA | `compose.nvidia.yaml` | `…/whl/cu126` | NVIDIA Container Toolkit |
| AMD | `compose.rocm.yaml` | `…/whl/rocm7.2` | ROCm kernel driver, `/dev/kfd` + `/dev/dri` |
| CPU | *(none)* | `…/whl/cpu` | nothing |

`s2s` and `qaa-agent` build once and run anywhere; they hold no weights.

**The parakeet streaming engine picks its own backend independently**, via the
`PARAKEET_BACKEND` build arg (`vulkan` default, or `hip`/`cuda`/`cpu`) — Vulkan
because it runs the identical build on AMD and NVIDIA, which torch's own
wheel-per-vendor split does not buy you. `hip` and `cuda` select the right
`cmake` flag but need a different builder base image than this Dockerfile
ships (ggml needs `hipcc`/`nvcc` at build time, not just a runtime library) —
they fail the build with a clear error rather than silently doing something
else. Measured on the target card for this integration (RX 9060 XT, gfx1200,
RDNA4): the GPU enumerates correctly under Vulkan/RADV and returns to idle
power within seconds of the process exiting.

**AMD notes.** `group_add` in the overlay uses group *names*, which only resolve
if `video`/`render` exist inside the container — if the GPU comes back invisible,
run `getent group render video` on the host and substitute the numeric GIDs. For
consumer cards ROCm does not officially list, set `HSA_OVERRIDE_GFX_VERSION`
(`10.3.0` for RDNA2, `11.0.0` for RDNA3).

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

## What the tests actually protect

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

`tests/test_parakeet_streaming.py` covers engine selection and the resampler
unconditionally (no model needed — the resampler bug it caught, a silently
dropped final ~35 ms of trailing audio on every `finalize()`, only shows up in
a test that checks exact output length, not just a matching prefix). The real
parakeet.cpp engine is exercised the same opt-in way as sherpa above:

```bash
SOFIA_STT_PARAKEET_LIBRARY_PATH=/path/to/libparakeet.so \
SOFIA_STT_PARAKEET_MODEL_PATH=/path/to/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf \
SOFIA_STT_PARAKEET_TEST_WAV=/path/to/16kHz-mono-italian.wav \
    uv run pytest tests/test_parakeet_streaming.py -v
```

`tests/test_tts.py` covers the streaming TTS path with a fake Kokoro engine,
not the real model — Kokoro's own inference turned out to be non-deterministic
between independent calls (confirmed empirically: two unmodified calls on the
same input differ, max abs diff ~0.08), so bit-comparing against the real
model would test Kokoro's own randomness, not the streaming refactor. It also
had to route around `TestClient`'s ASGI transport, which coalesces separate
response chunks into one blob — chunk-boundary assertions test the streaming
generator directly instead; real incremental delivery over an actual socket
was confirmed separately (`openspec/changes/add-streaming-tts/tasks.md`).

## Things to know before production

- **Latency budget.** The interesting numbers are per-stage: STT, LLM
  time-to-first-token, TTS first byte. `s2s` logs all three via the
  livekit-agents metrics collector. With streaming ASR on, STT effectively
  disappears from the budget and LLM TTFT dominates — tune that next.
- **This is a cascade, not a speech-to-speech model.** Despite the service name,
  audio is transcribed, reasoned over as text, and re-synthesised. A true S2S
  model (OpenAI Realtime, Moshi, Qwen-Omni) reaches ~300 ms and full duplex, but
  the model *is* the LLM — incompatible with bringing your own.
- **First boot downloads models.** ~1.3 GB for batch Nemotron, ~310 MB for the
  sherpa streaming model, and (only if `SOFIA_STT_STREAMING_ENGINE=parakeet`)
  ~939 MB for the parakeet GGUF. The healthcheck allows ten minutes; the
  `hf-cache` and `models` volumes mean it happens once.
- **Nemotron's licence is OpenMDW-1.1**, not Apache or MIT. It is a permissive
  open-model licence, but read it before shipping commercially. Whisper
  (MIT) and the sherpa-onnx models remain available as alternatives.
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
