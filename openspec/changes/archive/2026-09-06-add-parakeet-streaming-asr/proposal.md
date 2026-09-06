## Why

The streaming ASR path cannot transcribe Italian. `sherpa-onnx` serves the only
streaming recogniser we have, and no streaming zipformer release covers Italian —
so an Italian deployment is forced onto the batch backend and gives up interim
transcripts, which is what the semantic turn detector uses to decide someone has
finished speaking.

The model needed to fix this is already in the repo. `SOFIA_STT_MODEL_ID` pins
`nvidia/nemotron-3.5-asr-streaming-0.6b` — a cache-aware *streaming* FastConformer
that punctuates, cases, and reaches 4.25% WER on Italian — and `batch.py` runs it
through `.generate()` in whole-utterance mode, discarding the streaming capability
entirely. What is missing is a runtime that can drive it chunk by chunk.
[parakeet.cpp](https://github.com/mudler/parakeet.cpp) is that runtime: a ggml
implementation with cache-aware streaming, in-model end-of-utterance detection, a
flat C-API, and pre-converted GGUF weights published at WER 0 against NeMo.

## What Changes

- Add a second `StreamingRecognizer` implementation backed by parakeet.cpp,
  selected by a new `SOFIA_STT_STREAMING_ENGINE` setting (`sherpa` | `parakeet`).
  This mirrors the existing `SOFIA_STT_BATCH_ENGINE` (`nemotron` | `whisper`)
  rather than introducing a new selection idiom.
- Bind `libparakeet.so` from Python with `ctypes`. The C-API is flat and
  exception-free by design, so this needs no build-time dependency and no
  compiled extension of ours.
- Build parakeet.cpp in the audio image with **Vulkan** (`PARAKEET_GGML_VULKAN=ON`)
  as the default GPU backend, with the backend selectable at build time by a
  Docker build arg. Vulkan keeps one image working on AMD and NVIDIA alike; HIP
  and CUDA remain available for anyone who measures better with them.
- Fetch GGUF weights on first boot into the existing models volume, the same way
  `ensure_model()` already fetches sherpa releases. No weights in the image.
- Resample the incoming 24 kHz LiveKit audio to the model's 16 kHz inside the new
  recogniser. `StreamingSession.push()` already carries the sample rate; today
  `_SherpaSession` ignores it because sherpa resamples internally.
- Close turns with a trailing-silence rule implemented inside the new
  recogniser. Nemotron's own end-of-utterance signal is not available: that
  capability belongs only to `nvidia/parakeet_realtime_eou_120m-v1`, a
  separate, English-only model (see `design.md` D6). `SOFIA_STT_ENDPOINT_SILENCE`
  or an equivalent parakeet-specific setting remains the dominant contributor
  to turn latency on this path, the same as it is for sherpa.
- Carry the session's language to the recogniser, so Nemotron's per-chunk locale
  prompt — the thing that makes it multilingual at all — receives a real value.
- Keep `sherpa-onnx` as-is. It stays the default and remains the answer for
  deployments that want a smaller model or a language parakeet.cpp does not cover.

Not breaking: the default engine stays `sherpa`, and the OpenAI realtime wire
format served on `WS /v1/realtime` does not change.

## Capabilities

`openspec/specs/` is currently empty, so this change introduces the first
capability spec in the project. It is scoped to the streaming ASR path only; it
does not attempt to retroactively specify the batch backend, s2s, or qaa-agent.

### New Capabilities

- `streaming-asr`: How the `stt` service transcribes audio while the person is
  still speaking — recogniser selection and configuration, the session lifecycle
  behind `WS /v1/realtime`, how a turn is opened and closed, how interim deltas
  are emitted, how the session's language reaches the recogniser, and what the
  service does when a recogniser is unavailable or fails.

### Modified Capabilities

None. No spec exists yet for any capability this change touches.

## Impact

**Code**

- `src/sofia_galileo/audio/streaming.py` — new `ParakeetRecognizer` and session
  alongside `SherpaRecognizer`; a `build_recognizer()` selector mirroring
  `build_transcriber()`.
- `src/sofia_galileo/audio/config.py` — `SOFIA_STT_STREAMING_ENGINE` plus the
  parakeet settings (GGUF URL, model dir, backend, threads, chunk size).
- `src/sofia_galileo/audio/stt_app.py` — construct the selected recogniser.
- `src/sofia_galileo/audio/realtime.py` — language must reach the recogniser.
  Today `session.update` is deliberately ignored ("Config is ours, not the
  caller's"); whether to keep that and read the locale from settings, or start
  honouring the handshake, is settled in `design.md`.

**Build and deployment**

- `docker/audio.Dockerfile` — a build stage that compiles parakeet.cpp, plus the
  Vulkan loader in the runtime layer. The image gains a compiler toolchain it
  did not have.
- `compose.yaml` / `compose.rocm.yaml` / `compose.nvidia.yaml` — the new engine
  setting and, on AMD, whatever device access the Vulkan ICD needs beyond the
  `/dev/dri` already granted.
- Target hardware for the first deployment is a Radeon RX 9060 XT (gfx1200,
  RDNA4). ROCm 7.0.2+ supports it natively, so `HSA_OVERRIDE_GFX_VERSION` stays
  unused.

**Tests**

- `tests/test_realtime_ws.py` already drives the websocket against a fake
  recogniser and needs no change — that seam is what makes this swap cheap.
- New coverage for engine selection and for the resampling step.

**Docs**

- `docs/details.md` — the streaming-vs-batch table states "no Italian, no
  punctuation" for the streaming column; both cease to be true on this path.
- `CLAUDE.md` — "turn latency is set by policy, not compute" and the claim that
  `SOFIA_STT_ENDPOINT_SILENCE` dominates continue to hold on the parakeet path
  too; only the streaming-vs-batch language-coverage table changes.
- `.env.example` — the new settings.

**Dependencies**

- parakeet.cpp (vendored or fetched at build time) and its ggml submodule.
- The Vulkan loader and headers at build time; `libvulkan1` plus a Mesa ICD at
  runtime. No new Python dependency: `ctypes` is stdlib.

**Explicit non-goals**

- Splitting `stt` and `tts` into separate images. A pure-ggml `stt` would shed
  torch, transformers, the rocrand-header build stage and the MIOpen cache
  volume, but `tts` (Kokoro) still needs torch and the two share one image.
  That is a separate change.
- Replacing the batch backend. parakeet.cpp can also serve offline
  transcription; whether it should is out of scope here.
- Removing `sherpa-onnx`.
