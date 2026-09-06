## Context

See [proposal.md](proposal.md) — Why. Three properties of the existing code shape
this design more than anything else:

1. **The seam already exists.** `audio/streaming.py` defines `StreamingRecognizer`
   and `StreamingSession` as `Protocol`s, and `audio/realtime.py` depends only on
   those. `tests/test_realtime_ws.py` already drives the websocket against a fake
   recogniser. Adding a second recogniser is an additive change to one module.
2. **The batch path already picks between engines.** `build_transcriber()` selects
   `nemotron` or `whisper` from `SOFIA_STT_BATCH_ENGINE`. The streaming path gets
   the same shape rather than a new one.
3. **The audio image deliberately does not branch on GPU vendor.** Per CLAUDE.md,
   the only NVIDIA/AMD/CPU difference is the torch wheel index. That invariant is
   already leaking on exactly the path this change touches: `sherpa_provider` is
   pinned to `cpu` because the sherpa-onnx wheels ship CUDA only.

Target hardware for the first deployment is a Radeon RX 9060 XT — Navi 44,
**gfx1200**, RDNA4 — with ROCm 7.x on the host. The host CPU is a Xeon E5-2630 v4
(2016, AVX2, no AVX-512), which is why CPU-only ggml is not a satisfying answer
here even though ggml would run.

## Goals / Non-Goals

**Goals:**

- Streaming Italian transcription with punctuation and casing, using the model
  already pinned in `SOFIA_STT_MODEL_ID`.
- One audio image that accelerates on both AMD and NVIDIA, rather than a
  vendor-specific image per accelerator.
- Additive change: `sherpa` remains the default and the fallback, and rollback is
  an environment variable.
- No change to the OpenAI realtime wire format served on `WS /v1/realtime`.

**Non-Goals:**

- Tuning for maximum throughput. The workload is one conversation per session at
  real-time speed; a backend that is 20% slower but correct wins.
- Multi-session batching inside the recogniser. Sessions stay independent, as
  they are with sherpa.
- Replacing sherpa's own endpointing rules on the sherpa path. They stay exactly
  as they are.

## Decisions

### D1. Vulkan is the default GPU backend; the backend is a build arg

`PARAKEET_GGML_VULKAN=ON` by default, with a Docker build arg selecting
`vulkan | hip | cuda | cpu`. This mirrors how `TORCH_INDEX_URL` already
parameterises the same Dockerfile per accelerator.

*Why Vulkan over HIP,* despite ROCm being the host platform:

- **One image covers AMD and NVIDIA.** HIP would force the vendor-specific image
  the project has so far avoided. Vulkan actually repairs the invariant on the
  path where it currently leaks.
- **Lighter build.** HIP needs a full ROCm dev toolchain (`hipcc`, a
  `rocm/dev-ubuntu` builder stage); Vulkan needs headers, the loader, and
  `glslc`. The runtime needs `libvulkan1` plus a Mesa ICD instead of a second
  copy of the ROCm libraries — or a bet that ggml can link against the ROCm
  libraries bundled inside the torch wheel, which is fragile.
- **Idle behaviour.** RDNA4 under the HIP backend is
  [reported to hold the GPU at elevated clocks until the process exits](https://github.com/ROCm/ROCm/issues/5706);
  the Vulkan backend idles correctly. The `stt` container is
  `restart: unless-stopped` and sits idle between calls, so this is a standing
  power cost, not a benchmark footnote.
- **Speed is not a reason to prefer HIP.** On gfx1201, Vulkan measured *faster*
  than ROCm for token generation in llama.cpp (220 vs 179 t/s), with ROCm more
  stable on prompt processing. Neither result is about this model, but nothing
  suggests HIP is the performance choice.

*Alternatives considered:* **HIP** — rejected as the default for the reasons
above, retained as a build arg value so it can be measured. **CPU** — rejected
on this host; the Xeon is too old to carry a 0.6B FastConformer at real time
alongside Kokoro. **CUDA** — retained as a build arg for NVIDIA hosts that
measure better with it than with Vulkan.

*Consequence:* the AMD path needs `/dev/dri` for the Mesa RADV ICD, which
`compose.rocm.yaml` already grants. The NVIDIA path needs
`NVIDIA_DRIVER_CAPABILITIES` to include the Vulkan ICD, which
`compose.nvidia.yaml` does not currently request.

*Build portability:* `GGML_NATIVE=OFF`. The image must not bake in the build
host's ISA extensions, and the GPU is doing the work regardless.

### D2. `ctypes` binds `libparakeet.so`; no compiled extension of ours

parakeet.cpp publishes a flat, exception-free C-API precisely so it can be
embedded without a C++ ABI dependency. `ctypes` is stdlib, adds no build-time
dependency to `pyproject.toml`, and releases the GIL for the duration of each
foreign call — which matters because `realtime.py` already dispatches
`session.push` through `anyio.to_thread.run_sync`, and that offload is only
useful if the call actually releases the GIL.

*Alternatives considered:* **cffi** — equivalent capability, but a new build and
runtime dependency for no gain against a C-API this small. **pybind11 / a
compiled extension of our own** — would put a C++ toolchain into the Python
package build, contradicting the "no compiled extension" posture the project has
today. **`parakeet-server` as a sidecar** — does not work: it is
OpenAI-compatible but batch-only, single-request-at-a-time, with no websocket or
streaming endpoint. It cannot serve the streaming path at all.

### D3. Engine selection mirrors the batch path

`SOFIA_STT_STREAMING_ENGINE` (`sherpa` | `parakeet`, default `sherpa`) selects
between recognisers via a `build_recognizer(settings)` function shaped like the
existing `build_transcriber(settings)`, including raising `ValueError` on an
unknown value.

Default stays `sherpa` because parakeet.cpp's GPU backends are validated upstream
on NVIDIA GB10 and Apple M4 — not on RDNA4 under Mesa. The default flips once
this is measured on the target host, as a follow-up, not as part of this change.

### D4. Locale comes from `SOFIA_STT_DEFAULT_LANGUAGE`; `session.update` stays ignored

Nemotron's multilinguality *is* a per-chunk locale prompt, so the recogniser must
be told a language. It is read from the service's own settings and reused for
every chunk of every session, reusing `batch.to_locale()` for the mapping so the
batch and streaming paths agree on what `it` means.

This preserves the decision already documented in `realtime.py`: *"Config is
ours, not the caller's."* The client's `session.update` continues to be
acknowledged and its contents discarded.

*Alternative considered:* honouring `session.update`, which would let the room's
language reach the recogniser — s2s already passes `language=` to
`lk_openai.STT`, so the value is present in the handshake we discard. Deferred
deliberately: it reverses a documented decision and turns a service-wide setting
into per-session state, which is a larger change than this one. The cost of
deferring is that a single `stt` deployment serves one streaming language at a
time. That is acceptable now because the deployment driving this change is
Italian-only.

### D5. Resampling 24 kHz → 16 kHz is stateful, inside the recogniser

`parakeet_capi_stream_feed()` takes no sample rate, so it assumes the model's
16 kHz. LiveKit sends 24 kHz (`realtime.py:46`), and `_SherpaSession` gets away
with ignoring the rate only because sherpa-onnx resamples internally.
`ParakeetSession` must do it.

The resampler **must carry state across frames**. 24 kHz → 16 kHz is exactly 2:3,
so a polyphase resampler is exact — but resampling each 50 ms frame independently
leaves a filter discontinuity at every frame boundary, twenty times a second,
which is precisely the kind of artefact an ASR encoder notices. A streaming
resampler that retains its filter delay line between calls (e.g. `soxr`'s
streaming interface) is required; a stateless per-call `resample_poly` is not
acceptable.

*Consequence:* `StreamingSession.push(samples, sample_rate)` stops having a
decorative parameter, and a session becomes tied to the sample rate of its first
frame.

### D6. Turns close on a trailing-silence rule inside `ParakeetSession`, not on model EOU

Superseded during implementation. `parakeet_capi_stream_feed()` does return an
`eou` flag, but it is populated only by
`nvidia/parakeet_realtime_eou_120m-v1` — parakeet.cpp's own header is explicit
that this is *that* model's behaviour
(`src/streaming.hpp:20`: *"The model nvidia/parakeet_realtime_eou_120m-v1 emits
`<EOU>`..."*). A ctypes smoke test against the actual Nemotron GGUF, feeding
real 50 ms/16 kHz frames through `parakeet_capi_stream_feed`, confirmed this
directly: `eou` was 0 on every one of 269 chunks across a clip with two clear
sentence boundaries, while the accumulated transcript was correct. Nemotron
streams text; it does not detect turn ends.

Running the EOU model alongside Nemotron purely for turn detection was
considered and rejected: `parakeet_realtime_eou_120m-v1`'s HuggingFace model
card declares `language: [en]` — English only. Since Italian is the reason for
this change, a second model that cannot hear Italian buys nothing.

`ParakeetSession` therefore implements its own trailing-silence rule,
independent of the C-API's `eou` output, the same job
`enable_endpoint_detection`'s rule1/rule2/rule3 already do for sherpa. This
does not change `spec.md`'s "Turn lifecycle" requirement — turn boundaries are
still decided by the recogniser, not by a timer in the protocol layer;
`realtime.py` still just closes a turn on `result.is_final`. It changes only
which mechanism inside `ParakeetSession` decides `is_final`.

*Consequence:* `SOFIA_STT_ENDPOINT_SILENCE` (or an equivalent parakeet-specific
setting) has an effect on this path after all, and the "turn latency is set by
policy, not compute" claim in CLAUDE.md is not sherpa-specific — it holds for
both recognisers. The latency argument for parakeet.cpp over sherpa evaporates:
this change's value is Italian coverage, punctuation, and casing, not a faster
turn boundary.

### D7. Weights are fetched at boot, not baked into the image

Pre-converted GGUFs are published in a single collection repo
(`mudler/parakeet-cpp-gguf`), validated at WER 0 against NeMo, in f16 / q8_0 /
q6_k / q5_k / q4_k. A settings-driven URL is downloaded into the existing models
volume on first boot, following `ensure_model()`'s existing contract: download to
a temporary path, move into place only when complete, so a killed download cannot
leave a half-model that looks valid on the next boot.

Baking weights into the image would hard-code the language and add over a
gigabyte to it — the same reasoning already recorded for the sherpa model.
`scripts/convert_parakeet_to_gguf.py` exists upstream but is not needed: the
model this change targets is already published.

### D8. Failure to load is fatal for the streaming path, not a silent fallback

If the parakeet engine is selected and the library or weights cannot be loaded,
the service reports itself unhealthy rather than quietly falling back to sherpa.
Silently serving a different recogniser than the one configured would make a
misconfigured deployment look like a model quality problem. This matches how
`stt_app.py` already behaves when a backend is unavailable.

### D9. parakeet.cpp is pinned to an exact commit

The project is young and the C-API is new. The Dockerfile pins a specific commit
or tag, and the ctypes binding validates the symbols it needs at load time so an
ABI drift fails loudly at startup rather than mid-call.

## Risks / Trade-offs

- **Vulkan on RDNA4 under Mesa is not a validated upstream configuration.**
  parakeet.cpp's documented GPU testing is NVIDIA GB10 and Apple M4 Metal. →
  The backend is a build arg (D1), so falling back to HIP or CPU is a rebuild,
  not a redesign. Benchmark on the target host before flipping the default.
- **The container may not see a Vulkan device even when the host does.** ICD
  discovery inside a container is a common failure, and it fails as "no device"
  rather than as an error. → Log the enumerated Vulkan devices and the selected
  backend at startup, so a silent fall back to CPU is visible in the logs rather
  than showing up as mysterious latency.
- **A hand-rolled silence rule may endpoint worse than sherpa's tuned one**
  (D6). It is new code solving a problem sherpa's `enable_endpoint_detection`
  already solves, so it inherits none of sherpa's tuning. → Default stays
  `sherpa` (D3); the engines are switchable per deployment, so this is
  measurable side by side. Start from sherpa's rule1/rule2/rule3 values
  rather than retuning from scratch.
- **The image grows a compiler toolchain.** A multi-stage build keeps it out of
  the runtime layer, but build time increases materially. → Build parakeet.cpp in
  its own stage, ordered before the application copy so it caches across source
  changes, the same way the torch layer already is.
- **ctypes gives no type safety across the FFI boundary.** A signature change
  upstream becomes a crash, not a `TypeError`. → Pin the commit and validate
  symbols at load (D9).
- **`stt` accretes a fourth model-shaped dependency** (Nemotron via transformers,
  Whisper, sherpa-onnx, now parakeet.cpp) while still sharing an image with
  Kokoro. → Not addressed here; noted as a non-goal in the proposal. This change
  makes the case for splitting `stt` and `tts` stronger, not weaker.

## Migration Plan

1. Build and deploy with `SOFIA_STT_STREAMING_ENGINE` unset. Default `sherpa`
   means behaviour is unchanged and the new code path is dormant.
2. Verify the image reports its Vulkan device and selected backend at startup.
3. Flip one deployment to `parakeet`, with `SOFIA_STT_DEFAULT_LANGUAGE=it` and
   `SOFIA_S2S_STT_USE_REALTIME=true`. Compare against the batch path on the same
   Italian audio for WER, and against sherpa for endpointing behaviour.
4. Promote the default only after that measurement.

**Rollback:** unset `SOFIA_STT_STREAMING_ENGINE` and restart the `stt` container.
No data migration, no schema change, no change to any other service. The batch
backend is untouched throughout, so batch transcription remains available as a
fallback via `SOFIA_S2S_STT_USE_REALTIME=false`.

## Open Questions

- **Which quantisation?** A 0.6B model at f16 is roughly 1.2 GB, well within the
  card's VRAM, so size is not the constraint and f16 is the safe default. Whether
  q8_0 is indistinguishable in WER is worth measuring, since it halves both the
  download and the memory footprint. This is a setting with a conservative
  default (`parakeet_model_url` points at the f16 GGUF; pointing it at a q8_0
  URL is the whole change), so it does not touch the specs, the approach, or
  the task breakdown.

**Resolved during implementation, not left open:** chunk size is not a runtime
knob on this integration, despite Nemotron's transformers config exposing
80/160/320/560/1120 ms. `parakeet_capi_stream_begin`/`_feed`/`_finalize` take no
chunk-size parameter at all — `parakeet-cli info` on the published GGUF reports
a single fixed `chunk_size: [25,32]` baked in at conversion time, and
`mudler/parakeet-cpp-gguf` publishes one Nemotron GGUF per quantisation, not one
per chunk size. So there is nothing to measure or expose here: whichever chunk
size the published GGUF was converted with is what this integration gets.
Task 7.1 (measure across chunk sizes) is void for that reason and was replaced
with a single latency/WER measurement against the one chunk size available.
