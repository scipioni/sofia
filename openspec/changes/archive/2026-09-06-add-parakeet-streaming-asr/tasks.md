Ordered so the riskiest unknown is settled first. Group 1 is a throwaway spike
on the host: if Vulkan on gfx1200 does not work there, it will not work in a
container either, and the backend decision (design.md D1) reopens before any
application code has been written.

## 1. Prove the runtime before building anything around it

- [x] 1.1 Build parakeet.cpp from a pinned commit on the host with
      `-DPARAKEET_GGML_VULKAN=ON -DGGML_NATIVE=OFF`, and verify the build
      completes and `libparakeet.so` and `parakeet-cli` are produced
- [x] 1.2 Run `parakeet-cli` against an Italian sample and verify it enumerates
      the RX 9060 XT as a Vulkan device (not falling back to CPU) and returns a
      correct punctuated, cased transcript
- [x] 1.3 Drive the flat C-API (`parakeet_capi_stream_begin_lang` /
      `_feed` / `_finalize`) directly against the Nemotron GGUF with real
      50 ms/16 kHz frames, and verify streaming mode produces incremental
      output. **Finding:** `eou` was 0 on every chunk — Nemotron has no
      end-of-utterance signal; only the separate, English-only
      `parakeet_realtime_eou_120m-v1` emits one. design.md D6 rewritten:
      `ParakeetSession` implements its own trailing-silence rule instead. The
      CLI's own `--stream` flag segfaults in its internal
      `run_stream_over_pcm`/`MelFrontend` path on this build — irrelevant to
      us since we bind the C-API directly, not the CLI's duplicate path, but
      worth a heads-up if anyone else reaches for `parakeet-cli --stream`
- [x] 1.4 Record wall-clock and real-time factor for both modes, and compare
      against the 737 ms batch Nemotron figure in docs/benchmark.md. Streaming
      C-API on Vulkan: model load ~1.8s (one-time, at container startup), decode
      RTF ~0.18 (5.4x realtime) on a 13.43s clip fed in real 50ms frames —
      comfortable headroom under the 1.0 realtime threshold
- [x] 1.5 Confirm the pinned commit's C-API exposes
      `parakeet_capi_stream_begin_lang` / `_feed` / `_finalize` / `_free` with
      the signatures design.md D2 assumes, by reading the installed header and
      linking against them directly
- [x] 1.6 Verify the GPU returns to idle clocks after the process exits
      (design.md D1 cites this as the reason to prefer Vulkan over HIP; confirm
      it actually holds on this card). Confirmed via
      `/sys/class/drm/card1/device/pp_dpm_sclk` and `power1_average`: power rose
      25W -> 33W under a sustained streaming workload and settled back to
      ~21-26W within 5s of process exit, on gfx1200 (the RDNA4 idle-clock
      report design.md cites was for gfx1201)

## 2. Image build

- [x] 2.1 Add a builder stage to `docker/audio.Dockerfile` that compiles
      parakeet.cpp at the pinned commit, and verify the stage caches across an
      application source change. Verified as part of the full build (2.4) —
      the stage is ordered before `COPY src ./src` for the same reason the
      torch layer is, so it does not need reworking to cache correctly
- [x] 2.2 Add a `PARAKEET_BACKEND` build arg (`vulkan|hip|cuda|cpu`, default
      `vulkan`) selecting the cmake flag, and verify each value builds.
      **Scope note:** `vulkan` and `cpu` both build and are wired against the
      same `python:3.12-slim` builder base. `hip` and `cuda` select the right
      cmake flag but are NOT verified by an actual build here — ggml's
      HIP/CUDA backends need `hipcc`/`nvcc` at build time, which means a
      different builder base image (a ROCm or CUDA devel image), not just an
      apt package. Wiring that in is out of scope for this change (see
      design.md D1); passing `hip`/`cuda` today fails the cmake configure step
      with a clear "no such compiler" error rather than silently doing
      something else
- [x] 2.3 Copy `libparakeet.so` and its ggml libraries into the runtime layer and
      install the Vulkan loader and Mesa ICD, and verify the toolchain does not
      appear in the final image. Confirmed: `which cmake gcc g++ ninja git`
      is empty in the built runtime image
- [x] 2.4 Verify `docker compose -f compose.yaml -f compose.rocm.yaml config -q`
      still passes and the image builds end to end (`task config`). All three
      overlays (cpu/rocm/nvidia) validate with placeholder LiveKit/LLM secrets
      (no real credentials available in this environment); the full
      `docker build` of `docker/audio.Dockerfile` with
      `TORCH_INDEX_URL=rocm7.2 WITH_ROCRAND_HEADERS=with-rocrand` and the
      default `PARAKEET_BACKEND=vulkan` succeeds end to end (~13 min, mostly
      the ROCm torch wheel download)
- [x] 2.5 Run the container and verify it enumerates the Vulkan device through
      the `/dev/dri` access `compose.rocm.yaml` already grants. Ran the actual
      built image with `--device=/dev/kfd --device=/dev/dri --group-add 989
      --group-add 985` (rocm.yaml's exact grants) as the non-root `sofia` user:
      GPU enumerated (`RADV GFX1200`), streaming C-API decoded the Italian
      sample correctly through `libparakeet.so` loaded from
      `/usr/local/lib`

## 3. ctypes binding

- [x] 3.1 Write the `ctypes` binding for the streaming C-API, validating at load
      time that every required symbol is present, and verify a missing symbol
      raises at import rather than mid-call (design.md D9). Implemented in
      `audio/parakeet_capi.py` (`load_library`, `ParakeetContext`,
      `ParakeetStream`); `_REQUIRED_SYMBOLS` checked via `hasattr` before
      `_configure()` sets `argtypes`/`restype`
- [x] 3.2 Verify the binding releases the GIL during `_feed`, by confirming a
      second thread makes progress while a long feed is in flight — the
      `anyio.to_thread` offload in `realtime.py` is pointless otherwise.
      `test_stream_feed_releases_the_gil` (real-engine, opt-in): a background
      thread's counter crosses 1000 while a ~13s clip is fed to the real
      library in one call — confirmed against the actual `libparakeet.so`
- [x] 3.3 Verify the library path is resolved from configuration with a sensible
      default, and that a missing library produces a clear startup error.
      `SttSettings.parakeet_library_path` defaults to `"libparakeet.so"`
      (resolved via the platform's normal shared-library search, matching
      where `docker/audio.Dockerfile` installs it); `load_library()` raises
      `ParakeetLibraryError` with the path and the underlying `OSError` on a
      missing library

## 4. Streaming recogniser

- [x] 4.1 Implement the stateful 24 kHz → 16 kHz resampler and verify, per
      design.md D5, that resampling a signal in 50 ms frames yields the same
      output as resampling it whole — a stateless implementation will not.
      Implemented with `soxr.ResampleStream` in `_ParakeetSession._resample`.
      **Found and fixed a real bug while writing the equivalence test:**
      `resample_chunk()` buffers an algorithmic delay and never emits the last
      ~tens of ms unless the final call passes `last=True` — without an
      explicit flush, `finalize()` was silently dropping exactly the trailing
      audio the model needs most to commit its last tokens. Fixed by flushing
      (and `.clear()`-ing, since a flushed `ResampleStream` raises on reuse) in
      `finalize()`. Covered by
      `test_resample_framed_matches_whole_once_finalized`,
      `test_without_finalize_trailing_audio_is_not_yet_fed`, and
      `test_finalize_flushes_resampler_and_feeds_the_tail`
- [x] 4.2 Implement `ParakeetRecognizer` / session against the existing
      `StreamingRecognizer` / `StreamingSession` protocols, deriving
      `Transcript.is_final` from a trailing-silence rule inside the session
      (design.md D6 — the C-API's `eou` output is not populated by Nemotron
      and is not used), and verify it satisfies the protocols. Verified against
      both stubs (`test_silence_with_nothing_decoded_stays_non_final`,
      `test_endpoint_fires_after_trailing_silence_once_text_exists`) and the
      real engine (`test_real_engine_transcribes_and_endpoints`)
- [x] 4.3 Implement GGUF fetch-on-first-boot following `ensure_model()`'s
      staged-move contract, and verify an interrupted download is not reused
      (spec: "Streaming model weights are acquired at runtime"). Implemented
      as `ensure_file()` in `streaming.py`: downloads to `<target>.part`,
      renames into place only on completion; a target that already exists is
      never re-downloaded
- [x] 4.4 Pass the configured locale through `batch.to_locale()` on every chunk
      and verify the same code selects the same language on both paths
      (spec: "Transcription language is applied to the streaming path").
      `ParakeetRecognizer.__init__` calls `to_locale(settings.default_language)`
      once and reuses it for every stream (D4: config-driven, not per-session)
- [x] 4.5 Add `build_recognizer(settings)` mirroring `build_transcriber`, and
      verify an unknown engine value raises naming the setting and the value.
      `test_build_recognizer_rejects_unknown_engine` covers this

## 5. Configuration and wiring

- [x] 5.1 Add the new settings to `SttSettings` (`streaming_engine`, parakeet
      library path, GGUF URL and model dir) and verify each maps to its
      `SOFIA_STT_` environment variable via the `SttSettings` env prefix.
      **No separate chunk-size or quantisation fields:** chunk size is not a
      runtime parameter at all (see design.md's Open Questions —
      `parakeet_capi_stream_begin`/`_feed` take no such argument, it is baked
      into the published GGUF at conversion time); quantisation is already
      fully covered by `parakeet_model_url` — pointing it at a different
      quantisation's URL is the entire change, no second field needed
- [x] 5.2 Construct the selected recogniser in `stt_app.py` and verify `/healthz`
      reports "loading" until it is ready and never reports ready after a failed
      load (spec: "Selected recogniser cannot be loaded"). `stt_app.py` now
      calls `build_recognizer` the same way it called `SherpaRecognizer`
      before — same `anyio.to_thread.run_sync` call, same propagation.
      Verified directly: `SttSettings(streaming_engine="bogus")` raises
      `ValueError` out of `TestClient`'s lifespan startup rather than the app
      coming up in a bad state
- [x] 5.3 Log the selected engine, the resolved compute backend and the
      enumerated devices at startup, and verify a silent fall back to CPU is
      visible in the logs (design.md Risks). `engine=` added to both
      recognisers' `stt.streaming.loading`/`ready` log lines. Device
      enumeration itself comes from parakeet.cpp's own stdout
      (`ggml_vulkan: Found N Vulkan devices` / `pk::Backend using device:
      VulkanN`), which the real-engine test run showed interleaved between
      those two log lines — visible in container logs without extra code
- [x] 5.4 Add the settings to `compose.yaml` and `.env.example` with `sherpa` as
      the default, and verify an existing deployment's behaviour is unchanged.
      Added `SOFIA_STT_STREAMING_ENGINE` (default `sherpa`) and
      `SOFIA_STT_PARAKEET_MODEL_URL`; `SOFIA_STT_ENDPOINT_SILENCE` now also
      drives `SOFIA_STT_PARAKEET_ENDPOINT_SILENCE`, the same env var already
      driving sherpa's rule2. All three overlays re-validated with `docker
      compose ... config -q` after the edit
- [x] 5.5 Add the Vulkan ICD to `NVIDIA_DRIVER_CAPABILITIES` in
      `compose.nvidia.yaml` and verify the NVIDIA overlay still validates.
      Set to `graphics,compute,utility`; also defaulted `PARAKEET_BACKEND` to
      `vulkan` on the NVIDIA overlay (not `cuda`) per design.md D1 — one
      built artifact for both vendors, `cuda` stays available as an override.
      Confirmed via rendered `docker compose config` that both resolve
      correctly

## 6. Tests

- [x] 6.1 Verify `tests/test_realtime_ws.py` passes unchanged — the fake
      recogniser is what makes this swap cheap, and it should need no edit.
      Confirmed: file untouched, still passes
- [x] 6.2 Add engine-selection tests covering default, both valid values and an
      unknown value, and verify they pass without a model on disk. Added to
      `tests/test_parakeet_streaming.py`
- [x] 6.3 Add resampler tests covering framed-vs-whole equivalence and the
      exact 2:3 ratio, and verify they pass. Also added
      `test_stream_feed_releases_the_gil` (opt-in, needs the real engine —
      covers 3.2) since it needed a genuinely slow C call to be meaningful
- [x] 6.4 Add a parakeet recogniser test that self-skips unless the library and
      weights are present, following `tests/test_streaming_asr.py`'s precedent.
      Ran it for real against the built library and downloaded GGUF from
      group 1 — passes
- [x] 6.5 Verify `task check` passes (ruff + full pytest). 63 passed, 8
      skipped (the two real-engine tests correctly skip without the opt-in
      env vars); separately confirmed both pass when pointed at the real
      library and model

## 7. Measure, then choose the defaults

Resolves the remaining open question in design.md (quantisation). Chunk size
turned out not to be a runtime knob at all (see design.md's Open Questions) —
task 7.1 below is a single measurement against the one chunk size the
published GGUF has, not a sweep.

- [x] 7.1 Measure Italian WER and time-to-first-interim for the published GGUF
      (fixed chunk size — see design.md), recorded as the baseline for future
      comparison rather than a parameter to tune. On the real UDHR Article 1
      Italian recording (13.43s, via the flat C-API in real 50ms frames):
      transcript correct against the source text, first interim delta at
      chunk 19 (~0.90s in), steady-state decode RTF ~0.104-0.108 (9.3-9.7x
      realtime) — see 1.4 for the batch comparison. **Limitation:** one clip,
      not a WER corpus — no held-out Italian test set was available in this
      environment
- [x] 7.2 Compare f16 against q8_0 on WER, download size and memory, and set the
      quantisation default from the result. q8_0: 939 MB vs f16's 1.4 GB
      (33% smaller); steady-state RTF 0.104 vs f16's 0.108 (marginally
      faster, not slower); transcript identical except q8_0 got "in dignità"
      right where f16 ran the words together as "indignità" — no accuracy
      regression on this clip. Default flipped to q8_0 in both `config.py`
      and `.env.example`, with the single-clip caveat recorded in both
- [ ] 7.3 Compare `ParakeetSession`'s trailing-silence endpointing against
      sherpa's tuned 0.8 s rule on the same audio, and tune the parakeet-side
      silence thresholds from the result (design.md D6, Risks). **Mechanism
      verified against the real deployed container, tuning still not done.**
      Built a two-utterance clip (the same real recording split around a
      genuine 1.5s silence gap) and drove it through the actual `WS
      /v1/realtime` endpoint on a real running `stt` container: two distinct
      turns, two distinct item ids, each closed by the trailing-silence rule
      mid-stream, not just on the final explicit commit — the spec's turn-
      lifecycle contract holds end-to-end, not just against a stub. What's
      still missing is real conversational audio with natural mid-thought
      pauses to *tune* the threshold against; this only confirms the 0.8s
      default (inherited from sherpa's rule2, see 5.4) works, not that it's
      right for Italian conversation. See docs/benchmark.md's "Round 2" for
      the full transcript of the two-turn run
- [ ] 7.4 Run the full loop with `SOFIA_STT_STREAMING_ENGINE=parakeet`,
      `SOFIA_STT_DEFAULT_LANGUAGE=it` and `SOFIA_S2S_STT_USE_REALTIME=true`, and
      verify an Italian conversation works end to end via `task console`.
      **Deliberately not done, and not for lack of infrastructure this time:**
      a real LiveKit server, a real upstream LLM, and a live production
      `sofia-galileo` stack (`s2s`, `qaa-agent`, `stt`, `tts`, 7+ hours up) all
      turned out to be available. `sofia-s2s console` is interactive
      mic/speaker only (`livekit-agents` CLI, not scriptable with a wav file),
      and the production `s2s` worker runs with no `SOFIA_S2S_AGENT_NAME` set
      — LiveKit round-robins an unnamed worker across *any* undispatched room,
      so a second `s2s` instance against the same account risks silently
      picking up real production calls. Rather than risk that, testing was
      narrowed to the `stt` service directly: a real container built from
      `docker/audio.Dockerfile`, run standalone on an isolated port sharing
      the model-cache volume, hit over real HTTP (`POST
      /v1/audio/transcriptions`) and real WS (`WS /v1/realtime`, OpenAI
      realtime protocol, 24kHz frames paced at real time) — see
      docs/benchmark.md's "Round 2". The production stack was confirmed
      healthy and untouched before and after. What's still unverified is
      specifically the LiveKit-dispatch-to-qaa-agent round trip with this
      engine in the loop; testing that safely needs a dedicated test agent
      name or a separate LiveKit project, neither set up here

## 8. Documentation

- [x] 8.1 Update the streaming-vs-batch table in `docs/details.md` — the
      streaming column's "no Italian, no punctuation" is no longer true on this
      path — and add the group 7 measurements. Table now has three columns
      (sherpa/parakeet/batch); "For Italian, use batch" bullet and the GPU
      targets section both corrected to describe the new streaming option
- [x] 8.2 Update `CLAUDE.md`'s streaming backend description to mention the
      parakeet engine; the "turn latency is set by policy, not compute" claim
      is unchanged and needs no edit (design.md D6). Also clarified the
      "never branches on GPU vendor" invariant: true at the application-code
      level; `PARAKEET_BACKEND` is a second, independent build-time choice
      that mirrors `TORCH_INDEX_URL`'s existing pattern rather than breaking it
- [x] 8.3 Record in `docs/benchmark.md` the Vulkan-vs-CPU numbers from group 1
      and the chunk-size sweep from 7.1. No sweep to record — chunk size isn't
      a runtime parameter (7.1/design.md); added a "Streaming Nemotron via
      parakeet.cpp" subsection instead with the real measured numbers: device
      enumeration, load time, RTF for both quantisations, idle power, and the
      n=1 caveat stated plainly
