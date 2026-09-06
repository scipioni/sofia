## Context

See [proposal.md](proposal.md) for the motivation and the client/Kokoro research
that shapes this design. Two facts from that research drive every decision
below and are not re-derived here:

1. `livekit.plugins.openai.TTS` already consumes a chunked HTTP response
   (`with_streaming_response.create` → `async for chunk in stream.iter_bytes()`)
   and `AudioEmitter` has a **dedicated fast path for raw PCM**
   (`self._is_raw_pcm = mt.startswith("audio/pcm")`), bypassing the general
   `codecs.AudioStreamDecoder` entirely.
2. `KPipeline.__call__` is `Generator[KPipeline.Result, None, None]` — Kokoro
   already produces audio one internal text-segment at a time.

`S2SSettings.tts_response_format` defaults to `"wav"` today, despite
`tts_app.py`'s own docstring claiming PCM is what the client wants — a stale
comment, not a decision anyone made deliberately.

## Goals / Non-Goals

**Goals:**

- First audio byte reaches `s2s` as soon as Kokoro's first segment is ready,
  not after the whole reply is synthesised.
- `wav`/`flac` callers (`task say`, direct `curl`, anything saving a playable
  file) get exactly the same bytes they get today — bit-identical, not just
  "still works."
- One code path for both cases, not a fork that only the streaming path gets
  tested and maintained.

**Non-Goals:**

- True incremental streaming for `wav` or `flac`. A WAV header declares the
  total data length before the data; there is no way to patch that header
  after the fact on an HTTP response already in flight, and `flac` has the
  same shape of problem. Only `pcm` streams incrementally; `wav`/`flac`
  continue to be fully synthesised before any bytes go out — correct, just
  not faster.
- Changing what format `curl`/`task say` get by default. Their behaviour is
  unchanged; only what `s2s` itself requests changes (see D3).

## Decisions

### D1. `KokoroEngine` exposes a generator; the eager, whole-array method goes away

```python
def synthesize_chunks(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
    pipeline = self._pipeline(voice[0])
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            yield np.asarray(audio, dtype=np.float32)
```

This is `synthesize()` today minus the `chunks = [...]; concatenate` step —
Kokoro was always producing chunks; the engine was just hiding that from its
caller. `_warm_up()` and any other caller that wants one array still can:
`np.concatenate(list(engine.synthesize_chunks(...)))` is exactly today's
`synthesize()`. No caller needs both forms to coexist as separate methods.

### D2. The endpoint always streams the HTTP response; format decides whether that response is chunked in substance or in name only

`POST /v1/audio/speech` becomes a `StreamingResponse` unconditionally. What
differs by format is what the generator feeding it does:

- **`pcm`**: yield each chunk's `int16` bytes as soon as Kokoro produces it —
  genuinely incremental, first bytes before synthesis finishes.
- **`wav` / `flac`**: drain the same `synthesize_chunks()` generator to
  completion, concatenate, encode exactly as `_encode()` does today, and yield
  that once. One `StreamingResponse` code path; `wav`/`flac` just never get a
  second chunk. Bit-identical output to today, because it *is* today's code,
  called from inside a generator instead of before building the `Response`.

*Alternative considered:* keep `Response` for `wav`/`flac` and only use
`StreamingResponse` for `pcm`. Rejected — two response-construction code
paths for what is conceptually one behaviour (serve what's ready, as it's
ready) is the kind of fork that only the tested path stays correct.

*Alternative considered:* a placeholder/oversized WAV header (`0xFFFFFFFF`
data size), relying on the client tolerating a size that doesn't match the
stream — some ffmpeg-backed decoders accept this. Rejected: swapping a
robust, already-existing PCM fast path for a fragile header trick, to stream
a format nothing requires streaming, has no upside.

### D3. `S2SSettings.tts_response_format` default changes from `"wav"` to `"pcm"`

This is the one-line change that actually turns streaming on for the
conversation path. `curl`/`task say` are unaffected — they pass their own
`response_format` explicitly and get `wav` either way; only what `s2s` asks
*for by default* changes, matching what its own docstring already claimed.

*Consequence:* `AudioEmitter`'s dedicated PCM path uses `sample_rate`/
`num_channels` from the OpenAI plugin's own constants (`SAMPLE_RATE = 24000`,
`NUM_CHANNELS = 1`), not from anything the response carries — raw PCM has no
in-band format metadata by construction, so this only works because the
plugin hardcodes the same 24kHz mono `tts_app.py` already produces
(`TtsSettings.sample_rate = 24000`, "Kokoro's native rate; do not change"). If
that ever drifts between the two sides, PCM streaming decodes as noise with
no error — worth a comment at both ends, not new code.

### D4. Chunk granularity is exactly Kokoro's own segment boundary

No re-buffering, re-chunking, or minimum-chunk-size logic. Each
`synthesize_chunks()` yield is one complete Kokoro segment, converted to
`int16` bytes as one unit and pushed as one HTTP chunk. This is already
sample-aligned (a segment is a whole number of samples), so there is no
partial-sample-across-chunk-boundary case to handle.

*Alternative considered:* splitting further (e.g. fixed-size byte chunks) to
smooth delivery. Rejected for this change — Kokoro's own segmentation is
presumably already sentence/clause-sized, which is a reasonable unit; revisit
only if measurement shows segments themselves are too coarse.

**Confirmed during implementation, not assumed:** Kokoro's own segmentation
is language-dependent, and this changes where this feature actually pays off.
For English (`KPipeline`'s `lang_code in 'ab'` path), a short multi-sentence
input produced exactly one segment in testing — English tokenization only
sub-splits past an internal phoneme-count threshold, not on every sentence
boundary. For non-English languages including Italian (the deployment this
change was motivated by), Kokoro chunks by sentence boundary at ~400
characters internally, and a long paragraph produced multiple segments
arriving progressively, confirmed against the real service. In short: this
change helps most exactly where the production data motivating it came from
(long Italian replies) and may do nothing for a short English reply that fits
in one segment — which is not a bug, just what "stream what Kokoro already
segments" (D1) actually means once a real language's chunking behavior is
in the loop.

## Risks / Trade-offs

- **PCM streaming decodes silently wrong if the two sides' sample rate or
  channel count ever disagree.** No format metadata travels with raw PCM to
  catch this. → Both are already-existing constants (`TtsSettings.sample_rate`,
  the OpenAI plugin's `SAMPLE_RATE`/`NUM_CHANNELS`); this change doesn't
  introduce the coupling, only makes silently-wrong-if-violated something that
  now matters in the default path rather than an edge case.
- **`wav`/`flac` requests get no latency benefit** despite hitting the same
  "streaming" endpoint. This is inherent to the formats (Non-Goals), not a gap
  in the implementation — worth being explicit in the response/behaviour so
  nobody mistakes it for a bug later.
- **A slow first Kokoro segment still delays first audio.** Streaming moves
  the cost from "wait for the whole reply" to "wait for the first segment" —
  a reply whose first sentence is unusually long still has a slow start. Not
  addressed here; segment-level chunking is the unit this change works with.
