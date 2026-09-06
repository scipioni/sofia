Ordered so the riskiest unknown is settled first: whether a lazy generator fed
into FastAPI's `StreamingResponse` actually delivers chunks to a real client as
they're produced (not buffered by some layer in between), and whether stopping
consumption actually stops further synthesis. If that doesn't hold, D1/D2
reopen before the real endpoint is touched.

## 1. Prove the mechanism before building the real endpoint

- [x] 1.1 Write a throwaway FastAPI route that yields deliberately-delayed
      chunks (e.g. `asyncio.sleep` between them) via `StreamingResponse`, and
      verify with a real streaming HTTP client (not `TestClient`, which can
      buffer) that chunks arrive as they're yielded, not all at once at the
      end. Confirmed with byte-level timing: chunks arrived ~0.5s apart,
      matching the server's yield cadence exactly
- [x] 1.2 Confirm the same holds when the generator does real synchronous CPU
      work between yields (matching Kokoro's own blocking `pipeline()` call)
      dispatched via `anyio.to_thread.run_sync` per chunk, not the whole
      generator at once. Confirmed via a sync-generator-to-async-generator
      adapter (one `next()` per `run_sync` call): chunks still arrived
      incrementally, and 6 concurrent `/ping` requests all succeeded on
      schedule throughout, proving the blocking work never stalled the event
      loop. **Note for 3.2:** initial attempt used `readline()` on the client
      and looked like everything arrived in one lump at the end — that was a
      client-side artifact of reading non-newline-delimited bytes with
      `readline()`, not a server problem; switched to raw incremental reads
      and it resolved. Real PCM has no delimiters either, so the real
      implementation's tests must avoid the same mistake
- [x] 1.3 Confirm that stopping consumption (client disconnects or closes the
      connection early) actually stops the generator from producing further
      chunks, not just from having them delivered — this is what the spec's
      disconnect requirement depends on. Confirmed directly via server-side
      log capture: after an early disconnect following chunk 0, the log
      shows the generator's `finally` block running immediately, with no
      further "yielding chunk" lines — chunks 1-4 were never produced, not
      just undelivered
- [x] 1.4 Confirm `livekit.plugins.openai.TTS`'s raw-PCM fast path
      (`AudioEmitter._is_raw_pcm`) actually plays audio starting from the
      first chunk against a real streaming server, not just that it doesn't
      error — a quick script using the same client the way `s2s` does is
      enough, full LiveKit room not required. Confirmed by driving the real
      `livekit.plugins.openai.TTS` class against a spike server streaming PCM
      in delayed segments: first audio frame arrived at 0.547s into a 2.15s
      total stream, with frames continuing to arrive progressively
      throughout — not buffered until the end

## 2. KokoroEngine: generator instead of eager concatenation

- [x] 2.1 Add a generator method yielding one `np.ndarray` per Kokoro segment
      (design.md D1), and verify it satisfies
      `test_resample_framed_matches_whole`-style equivalence: concatenating
      every yielded chunk equals today's `synthesize()` output exactly, for
      the same input. **Scope note:** bit-comparison against the *real* model
      turned out to be meaningless — confirmed empirically that two
      unmodified, independent `synthesize()` calls on the same input already
      differ (Kokoro's own inference isn't deterministic between calls, max
      abs diff ~0.08 observed). Verified the actual refactor equivalence
      instead with a deterministic fake pipeline (`tests/test_tts.py`),
      which is what's actually testable here
- [x] 2.2 Replace `synthesize()`'s body with a thin wrapper over the new
      generator (`np.concatenate(list(...))`) so `_warm_up()` and any other
      whole-array caller need no changes, and verify `_warm_up()` still
      passes. Done; `_warm_up()` untouched and still calls `synthesize()`
      directly
- [x] 2.3 Verify chunk boundaries are sample-aligned (no partial sample split
      across two chunks) — true by construction (design.md D4) if each
      Kokoro segment is converted to bytes as one unit, but verify it rather
      than assume it. Confirmed via the fake-pipeline tests: each yielded
      chunk is exactly one segment's full array, never partial

## 3. Streaming endpoint

- [x] 3.1 Change `POST /v1/audio/speech` to a `StreamingResponse` for every
      format (design.md D2), and verify existing non-streaming behaviour for
      `wav`/`flac` is bit-identical to before this change on the same input
      (spec: "Output is unchanged by this capability"). **Bit-identical
      verification note:** Kokoro's own inference is non-deterministic
      between independent calls (see 2.1), so this was verified structurally
      instead — `_encode()`'s wav/flac branch is byte-for-byte the same code
      it was before, only now invoked from inside a generator that yields
      once rather than a function that returns a `Response` directly.
      Confirmed against the real service: a `wav` request produces one
      correctly-formed RIFF file (`ffprobe`-verified, correct declared size,
      correct duration) with all bytes arriving in a single flush
- [x] 3.2 For `pcm`, stream each chunk's `int16` bytes as soon as its Kokoro
      segment is ready; for `wav`/`flac`, drain to completion, encode via the
      existing `_encode()` path, and yield once — verify with the real
      streaming client from 1.1 that `pcm` delivers multiple chunks over time
      while `wav`/`flac` deliver one. **Finding along the way:** a short
      multi-sentence test input produced only one Kokoro segment regardless —
      Kokoro's English tokenizer only sub-splits past a phoneme-count
      threshold, not on every sentence. Retested with a long paragraph
      matching the actual slow production replies (Italian, `if_sara`):
      Kokoro's non-English path chunks by sentence boundary at ~400 chars,
      and 3 distinct segments arrived progressively over the stream,
      confirming multi-chunk streaming genuinely happens for the inputs that
      motivated this change. `wav` confirmed single-flush (3.1)
- [x] 3.3 Verify unsupported-format rejection and empty-input handling are
      unchanged (spec: both preserved-behaviour scenarios) — these currently
      return before any synthesis starts; verify that is still true now that
      the success path is a generator. Confirmed against the real service:
      unsupported format still returns HTTP 400 with the same error message,
      empty input still returns an empty 200 body, neither triggers synthesis
- [x] 3.4 Verify a client disconnecting mid-`pcm`-stream stops further
      synthesis for that request and does not affect other concurrent
      requests (spec: "A disconnected client does not affect the service"),
      using the mechanism proven in 1.3. Confirmed against the real service:
      disconnected after the first segment of a long request; no
      `tts.synthesized` completion log ever appeared for that request (waited
      well past its normal completion time), and the server served the next
      request normally immediately after

## 4. Wire the default so `s2s` actually uses it

- [x] 4.1 Change `S2SSettings.tts_response_format` default from `"wav"` to
      `"pcm"` (design.md D3), and verify `curl`/`task say` are unaffected —
      they pass their own `response_format` and get `wav` either way. `task
      say` writes `out.wav` via an explicit `"response_format":"wav"` in its
      own curl body, confirmed unaffected by the default; `task check` passes
- [x] 4.2 Verify the sample-rate/channel-count assumption PCM streaming
      depends on (design.md D3's silent-noise risk): confirm
      `TtsSettings.sample_rate` and the OpenAI plugin's hardcoded
      `SAMPLE_RATE`/`NUM_CHANNELS` agree, and leave a comment at both ends
      pointing at each other so a future change to one is caught by a reader,
      not by noisy audio in production

## 5. Tests

- [x] 5.1 Add a test driving `POST /v1/audio/speech` with `response_format=pcm`
      against multi-segment input and asserting more than one chunk is
      observed before the response completes (spec: "Multi-segment input
      streamed as PCM"). **Scope note:** `TestClient`'s ASGI transport
      coalesces separate yields into one blob — confirmed empirically,
      exactly the buffering risk 1.1 flagged — so this asserts chunk
      boundaries against `_stream_speech()` directly
      (`test_stream_speech_yields_one_chunk_per_segment`), with a separate
      `TestClient`-based test for final-content correctness over real HTTP.
      Chunk-boundary behaviour over a real socket was already confirmed in
      3.2 against the real service
- [x] 5.2 Add a test asserting concatenated PCM chunks equal a direct
      `synthesize()` call's output for the same input (spec: "Concatenated
      stream equals the full synthesis"). Covered in the same test as 5.1
- [x] 5.3 Add a test asserting a `wav` request's body is byte-identical
      before/after this change for a fixed input (spec: "WAV request is not
      chunked early", "Output is unchanged by this capability").
      `test_wav_delivers_exactly_one_http_chunk` asserts the endpoint's
      output equals `_encode()` called directly on the same audio — proving
      the streaming wrapper introduces no discrepancy in the unchanged
      encoding path
- [x] 5.4 Verify existing tests for unsupported-format and empty-input
      handling still pass unchanged. Added explicit fake-engine-backed tests
      for both (`test_unsupported_format_rejected_before_synthesis`,
      `test_empty_input_returns_empty_body_without_synthesis`)
- [x] 5.5 Verify `task check` passes (ruff + full pytest). 73 passed (up from
      68), 8 skipped, unchanged

## 6. Real measurement, against the live deployment

Mirrors `add-parakeet-streaming-asr`'s own approach: measure against the real
running stack, not just unit tests, using the same isolation discipline (a
distinct image tag or standalone container first, never touching the live
`sofia-galileo-*` project directly until the change is verified).

- [x] 6.1 Build and run the changed `tts` service standalone, and repeat the
      real-conversation-log measurement method from
      `openspec/changes/add-parakeet-streaming-asr/` (or a live test call, if
      available) to get a fresh TTS time-to-first-byte figure for a long
      reply, for direct comparison against the 1.04–1.54s figures that
      motivated this change. Built under a distinct tag
      (`sofia-galileo/audio:streaming-tts-test`), ran standalone on an
      isolated port with real ROCm GPU access, sharing the read-mostly
      `hf-cache`/`miopen-cache` volumes — the live `sofia-galileo-*` project
      was not touched (confirmed stopped throughout, exactly as the user had
      just asked). For the same long Italian reply (45.65s of audio): `wav`
      (old-equivalent, single-flush) first-byte 4.29–4.66s ≈ its own total
      time; `pcm` (new streaming) first-byte 1.83–1.86s — a ~60% cut in
      time-to-first-audio, with total synthesis time unchanged either way
      (~4.3s), exactly as expected: streaming doesn't make synthesis faster,
      it delivers the first part sooner
- [x] 6.2 Once verified, recreate the live `tts` (and `s2s`, for the
      `tts_response_format` default) containers with the change, and confirm
      `docker ps` shows both healthy afterward. Held until explicit
      confirmation (the user had just asked to stop the stack); once given,
      rebuilt `stt`/`tts`/`qaa-agent` under the production `rocm`/`local`
      tags and brought the full stack back up — all four containers healthy,
      `s2s` re-registered against the real LiveKit server. **Bug found via a
      real call afterward:** `s2s` has its own separate Dockerfile
      (`docker/s2s.Dockerfile`) and was never included in that rebuild —
      `sofia-galileo/s2s:local` was still the image from before this whole
      change (`s2s/config.py`'s `tts_response_format` default lives in that
      image, not the audio one). Every real call was still getting `wav`, the
      old whole-reply-first behaviour — streaming was never actually active
      despite this task's own completion note above. Rebuilt and recreated
      `s2s` specifically, then confirmed directly inside the running
      container (`S2SSettings().tts_response_format`) that it resolves to
      `pcm` before trusting it again

## 7. Documentation

- [x] 7.1 Update `docs/details.md`'s TTS description — "non-streaming: the
      service synthesises the whole text before answering" is no longer true
      for the `pcm` path. **Correction:** that exact claim actually lives in
      `docs/benchmark.md`, not `details.md` (fixed in 7.2). `details.md` had
      no false claim to correct; added a one-line pointer at its `tts` intro
      bullet instead, for discoverability
- [x] 7.2 Update `docs/benchmark.md`'s TTS section and its "Improvement
      levers" list (which currently names streaming TTS as a future lever)
      with the measurements from group 6. Both updated with the real
      wav-vs-pcm comparison (4.29–4.66s vs 1.83–1.86s first-byte, same total
      synthesis time); "Improvement levers" #2 struck through as done
- [x] 7.3 Update `docs/details.md`'s "What the tests actually protect" section
      to mention what the new tests guard, following its existing pattern of
      explaining *why* each test exists, not just what it does
