## Why

`tts` synthesises a whole reply before answering: `KokoroEngine.synthesize()`
drains Kokoro's own generator into one array and `POST /v1/audio/speech`
returns one complete `Response`. Real conversation logs from the
Italian/parakeet deployment (2026-09-06, see
`openspec/changes/add-parakeet-streaming-asr/`) show this cost scaling
directly with reply length: 0.57–0.63s time-to-first-byte for short replies,
climbing to 1.04–1.54s for replies with 7.6–12.8s of audio. On the turns that
took over 2s end-to-end, TTS time-to-first-byte was the dominant cost, not
endpointing or the LLM — both of those stayed flat regardless of reply length.

The fix turns out to need no client-side work. `livekit.plugins.openai.TTS`
(the client s2s already uses) drives its request through
`client.audio.speech.with_streaming_response.create(...)` and consumes
whatever the server sends as a genuine chunked HTTP body —
`async for chunk in stream.iter_bytes(): output_emitter.push(chunk)` — and
`AudioEmitter` decodes what arrives through a real streaming decoder
(`codecs.AudioStreamDecoder`), unconditionally, regardless of the plugin's own
`TTSCapabilities(streaming=False)` flag (that flag governs incremental *text*
input, a different thing). And Kokoro's own `KPipeline.__call__` is already
`Generator[KPipeline.Result, None, None]` — it yields one result per internal
text segment as that segment's audio becomes ready. Nothing changes on the
client; nothing changes in Kokoro. The only place eagerly draining a stream
that was already incremental on both sides is `tts_app.py` itself.

## What Changes

- `KokoroEngine.synthesize()` (or a new sibling method) becomes a generator,
  yielding each audio chunk as Kokoro produces it, instead of collecting the
  whole reply into one array before returning.
- `POST /v1/audio/speech` responds with a `StreamingResponse`: the first audio
  bytes go out as soon as the first chunk is ready, not after the whole reply
  is synthesised.
- Response format for streaming: `wav`'s header declares a total data length
  that is not known until synthesis finishes, which is fundamentally at odds
  with incrementally streaming an unknown-length body. `pcm` (raw samples, no
  header, no declared length) has no such conflict — and `tts_app.py`'s own
  docstring already says *"The LiveKit OpenAI TTS plugin asks for raw PCM,
  which is also the cheapest path here"*, even though `S2SSettings
  .tts_response_format` currently defaults to `wav`. Resolving that mismatch
  is in scope; the concrete mechanism (force `pcm` for streaming responses,
  or change the default, or something else) is a `design.md` decision, not
  settled here.
- Non-streaming callers (anything hitting `/v1/audio/speech` expecting one
  complete response — `task say`, direct `curl`, any client that ignores
  chunked transfer) must keep working. Whether that means always streaming
  the response (with a well-behaved client just reading it all before
  returning) or keeping both a streaming and non-streaming path is a
  `design.md` decision.

Not breaking: the wire contract stays OpenAI-compatible
(`POST /v1/audio/speech`); no change to `s2s`, to the LiveKit plugin config,
or to Kokoro itself.

## Capabilities

### New Capabilities

- `streaming-tts`: How the `tts` service synthesises and delivers speech —
  chunked delivery as audio becomes available, format constraints that follow
  from streaming an unknown-total-length body, and what callers observe
  whether or not they consume the response incrementally.

### Modified Capabilities

None. No spec exists yet for `tts` behaviour of any kind — `openspec/specs/`
is still empty (the sibling `add-parakeet-streaming-asr` change has not been
archived yet).

## Impact

**Code**

- `src/sofia_galileo/audio/tts_app.py` — `KokoroEngine.synthesize()` becomes a
  generator; `POST /v1/audio/speech` returns a `StreamingResponse`; `_encode()`
  needs a chunk-at-a-time form for whichever format(s) end up streamable.
- `src/sofia_galileo/s2s/config.py` — `tts_response_format` default, if the
  design decision changes it to `pcm`.

**Tests**

- New coverage for chunked delivery: first bytes arriving before synthesis of
  the full reply completes, and for whatever non-streaming fallback the
  design settles on.
- `docs/details.md`'s existing test-rationale section
  (`#what-the-tests-actually-protect`) gets a note once these land.

**Docs**

- `docs/benchmark.md` — the TTS section and its "Improvement levers" list
  both cite lack of streaming TTS as a known cost; both get updated once this
  ships, with fresh numbers from the same kind of real-conversation
  measurement that motivated this change.
- `docs/details.md` — TTS is currently documented as
  *"non-streaming: the service synthesises the whole text before answering"*;
  that becomes wrong.

**Explicit non-goals**

- Changing how `s2s` splits a reply into sentences before calling TTS. That
  splitting already happens upstream of this service; this change is about
  what happens to *one* call once it arrives.
- Streaming the LLM-to-TTS path in `qaa-agent` or `s2s`. Token streaming from
  the brain already reaches `s2s` sentence by sentence today; this change is
  scoped to what `tts` does with one sentence, not the pipeline around it.
- Tuning `SOFIA_QAA_MAX_TOKENS` or reply length. That is a separate, cheaper
  lever for the same symptom (see the parakeet change's benchmark notes) and
  can be pulled independently of this one.
