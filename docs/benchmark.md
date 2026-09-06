# sofia-galileo — benchmark report

Measured on the reference deployment: AMD RX 9060 XT (RDNA4 / gfx1200),
PyTorch 2.14.0+rocm7.2, livekit-agents 1.8.0, brain = `qwen3-vl:30b-a3b-instruct`
on a remote OpenAI-compatible endpoint, batch ASR (Nemotron 3.5, fp16, GPU),
Kokoro-82M TTS (GPU). 2026-09-06. Every number below comes from a live stack —
no mocks, no synthetic timers.

## How it was measured

- **Per-service**: repeated HTTP/WS calls against the deployed containers
  (medians over 5 runs, one warmup discarded).
- **End-to-end**: a scripted WebRTC participant joins a real room on
  `wss://livekit.csgalileo.org`, publishes a microphone track
  (`source=MICROPHONE` — the agent's room input ignores tracks with any other
  source), speaks a TTS-generated question, and timestamps the first audio
  frames of the agent's reply at frame granularity. Numbers are then
  cross-checked against the worker's own `EOU/LLM/TTS metrics` log lines.

## Headline numbers

| Metric | Value |
|---|---|
| **Turn latency** (user stops speaking → agent audio starts) | **1.9 – 2.5 s** (3 instrumented runs) |
| Agent dispatch (room join → agent audio track subscribed) | 2.4 – 3.8 s |
| Greeting audible after join | ~8 s |

Where the turn latency goes (worker-side metrics, one run):

```
user stops speaking                                     0.00 s
  └─ batch STT transcription                    ~1.21 s   ← dominant cost
  └─ semantic end-of-turn + 0.4 s floor         +0.11 s   (EOU total 1.32 s)
  └─ LLM first token (qaa → qwen3-vl)           +0.73 s   (0.22–0.89 s across runs)
  └─ TTS first byte (Kokoro, whole-sentence)    +0.88 s   (0.60–1.03 s across runs)
agent audio reaches the participant              ≈ 1.9–2.5 s
```

The single biggest lever is `SOFIA_STT_STREAMING`: the batch transcriber cannot
return a transcript before the person has stopped, so it alone costs ~1.2 s per
turn. Streaming sherpa-onnx removes almost all of it (see trade-offs).

## Per-component results

### Brain — `qaa-agent` (via upstream `qwen3-vl:30b-a3b-instruct`)

| Probe | Result (median, 5 runs) |
|---|---|
| Streaming TTFT (SSE) | **135 ms** (min 123, max 247) |
| Streaming total (short answer) | 245 ms |
| Chunk rate after first token | ~64/s |
| Non-streaming total | 209 ms |

The brain is not the bottleneck: well under 1 s of the ~2 s turn budget.

### Speech-to-text

| Probe | Result |
|---|---|
| Batch Nemotron 0.6B, 2.15 s clip (en) | 178 ms → **RTF 0.083** |
| Batch Nemotron 0.6B, 5.25 s clip (it) | 334 ms → **RTF 0.064** |
| Streaming zipformer, first interim delta | 1.28 s after speech onset (decode-as-you-talk) |
| Streaming zipformer, final after trailing silence | ~0.5 s decode cost |
| LiveKit plugin one-shot `recognize` | correct transcript, RTF ≈ 0.1 |

Batch accuracy is excellent (round-trip "Ciao, sono Sofia e parlo con la tua
voce." → perfect transcript, punctuated); the cost is purely the wait-until-silence.

#### Streaming Nemotron via parakeet.cpp (Vulkan), the Italian-capable streaming path

Two rounds of measurement: first against the bare `libparakeet.so` on the host
(bypassing the deployed service, the websocket, and the resampler — a spike, not
a deployment test), then against the actual built `stt` container over its real
HTTP/WS endpoints, matching this document's own "Per-service" methodology. Same
hardware throughout, `PARAKEET_BACKEND=vulkan`, model
`nemotron-3.5-asr-streaming-0.6b`. Clip: the UDHR Article 1 Italian recitation,
13.43 s (real recorded human speech, not TTS — synthetic speech's prosody
turned out to be a bad proxy for real accuracy testing; see
`openspec/changes/add-parakeet-streaming-asr/` for the full write-up).

**Round 1 — bare library, host process, 16 kHz direct (no resampler in the loop):**

| Probe | Result |
|---|---|
| Vulkan device enumerated | `AMD Radeon RX 9060 XT (RADV GFX1200)`, not CPU fallback |
| Model load (one-time, at container startup) | ~1.2–1.8 s |
| First interim delta | chunk 19 of 269, ~0.90 s into the clip |
| Steady-state decode, f16 GGUF (1.4 GB) | RTF **0.104–0.108** (9.3–9.6× realtime) |
| Steady-state decode, q8_0 GGUF (939 MB) | RTF **0.104** (9.6–9.7× realtime) |
| Idle power after process exit | 25 W baseline → 33 W under load → 21–26 W within 5 s of exit |

Accuracy: transcript matched the source text via the streaming C-API exactly
as well as the batch path above, punctuated and cased — q8_0 got one word
boundary right ("in dignità") where f16 ran it together ("indignità"); no
regression anywhere else on this one clip. **This is n=1, not a WER corpus** —
treat the q8_0-over-f16 default as measured-and-reasonable, not exhaustively
validated.

**Round 2 — the real deployed container**, built from `docker/audio.Dockerfile`
exactly as `compose.rocm.yaml` would (ROCm torch + `PARAKEET_BACKEND=vulkan`),
run standalone alongside the live production stack on an isolated port and a
shared, read-mostly model-cache volume (no interference — the production
`sofia-galileo-*` containers stayed healthy throughout, confirmed before and
after). Real `curl` against `POST /v1/audio/transcriptions`, and a real
`websockets` client speaking OpenAI's realtime protocol against
`WS /v1/realtime`, feeding 24 kHz frames paced at real time in 50 ms steps —
the resampler is genuinely in the loop this time, not bypassed.

| Probe | Result |
|---|---|
| Batch, same clip, steady-state (4 runs, 1st discarded as warmup) | 0.54–0.78 s → RTF 0.040–0.058 |
| Streaming (parakeet), first interim delta over the real WS | 0.96–1.14 s into the clip (median 0.96 s) — consistent with Round 1's 0.90 s: the resampler adds nothing measurable |
| Streaming (parakeet), full-clip completion | median 13.70 s (13.43 s audio + real-time send pacing + final decode) |
| Container startup, cold | batch model ready ~12 s after launch; parakeet GGUF (939 MB, not yet cached) downloaded in ~23 s; parakeet engine ready ~26 s after launch |

Transcript over the real WS matched Round 1 exactly, `<it-IT>` boundary tokens
and all.

**Turn boundaries verified mid-stream, not just on explicit commit.** The clip
above has no pause long enough to trigger endpointing on its own, so a second
test clip was built — the same real recording split in two around a real
1.5 s silence gap — and fed through the same real WS:

```
t= 0.96s  speech_started    item=...4885 (turn 1)
t= 7.64s  speech_stopped  →  completed  "Articolo uno ... e diritti"
t= 8.94s  speech_started    item=...4885 (turn 2, new item id)
t=15.33s  speech_stopped  →  completed  "Esi sono ... fratellanza."
```

Two distinct turns, two distinct item ids, each closed by
`ParakeetSession`'s own trailing-silence rule mid-stream — exactly the spec's
turn-lifecycle contract, confirmed against the real service rather than a
stub. One real, small artefact: the first word after a stream reset ("Esi"
instead of "Essi") is slightly less accurate — cache-aware streaming has no
left-context to warm up on immediately after a reset. Still unresolved from
Round 1: no real conversational audio with natural mid-thought pauses was
available to *tune* the 0.8 s threshold against — this confirms the mechanism
works, not that 0.8 s is the right number for Italian conversation.

One number this integration does **not** improve: turn latency. Unlike sherpa's
own endpoint detection, Nemotron has no end-of-utterance signal — that
capability belongs only to the separate, English-only
`nvidia/parakeet_realtime_eou_120m-v1` — so `ParakeetSession` uses the same
kind of trailing-silence rule sherpa already does (0.8 s default). The value
this path buys is Italian (and 39 other locales) with punctuation while
streaming, not a faster turn boundary.

### Text-to-speech (Kokoro-82M, GPU)

| Utterance | Wall | RTF |
|---|---|---|
| ~2 s audio (short) | 0.49 s | 0.28 |
| ~5 s audio (medium) | 0.78 s | 0.15 |
| ~8 s audio (long) | 1.0 s | 0.13 |

**Streaming as of `add-streaming-tts`** for `response_format=pcm` (`s2s`'s own
default): the service sends audio as Kokoro produces each internal segment,
rather than synthesising the whole reply before answering. `wav`/`flac`
responses are unchanged — still one complete body, because a WAV header must
declare a total length that isn't known until synthesis finishes.

Real measurement, same Italian voice, same hardware, same 45.65 s-audio reply
that motivated the change — `wav` (the old behaviour, still what you get if
you request it) versus `pcm` (the new default):

```
wav (old behaviour)   first byte 4.29–4.66 s  ≈ total time (one flush)
pcm (streaming)       first byte 1.83–1.86 s    total time unchanged, ~4.3 s
```

A ~60% cut in time-to-first-audio for a long reply. Streaming doesn't make
synthesis faster — total time is the same either way — it just stops making
the person wait for all of it before hearing any of it. Kokoro's own
segmentation is language-dependent: non-English languages (including Italian)
chunk by sentence boundary at ~400 characters internally, which is why long
replies see multiple segments and a real win; a short English reply that fits
in one Kokoro segment sees none, because there's nothing to stream ahead of.

The LiveKit plugin still requests per sentence upstream, which is a separate,
unaffected latency floor — this change is about what happens to *one* call
once it arrives, not how `s2s` splits replies before sending them.

**Deployment pitfall, found the hard way:** `s2s` has its own Dockerfile
(`docker/s2s.Dockerfile`), separate from the `stt`/`tts` image — and the
`tts_response_format` default (the one-line change that actually turns
streaming on) lives in `s2s/config.py`. Rebuilding `stt`/`tts`/`qaa-agent`
after this change and recreating the stack is *not* enough; `s2s` needs
rebuilding too, or every real call keeps getting `wav`, silently, with no
error — just the old latency. Confirmed live: a full conversation right after
the "completed" deploy still showed `fmt: wav` on every turn, because `s2s`
was still running the pre-change image. Fixed by rebuilding `s2s` specifically
and confirming inside the running container
(`S2SSettings().tts_response_format`) that it actually resolved to `pcm`
before trusting it again.

**First live confirmation**, a real Italian conversation over `wss://livekit.
csgalileo.org` after that fix, `fmt: pcm` confirmed on every turn:

| Turn | Speech | EOU wait | LLM ttft | TTS ttfb | Reply audio | Total |
|---|---|---|---|---|---|---|
| 1 | 1.6 s | 1.46 s | 0.20 s | 0.53 s | 2.1 s | 0.94 s |
| 2 | 2.45 s | 0.82 s | 0.19 s | 0.94 s | 6.4 s | 1.78 s |
| 3 | 2.05 s | 0.95 s | 0.23 s | 0.66 s | 3.5 s | 1.25 s |
| 4 | 4.0 s | 0.94 s | 0.21 s | 0.84 s | 5.5 s | 1.67 s |
| 5 | 0.0 s | 1.50 s | 0.40 s | 0.55 s | 2.1 s | 2.27 s |
| 6 | 1.65 s | 0.93 s | 0.19 s | 0.66 s | 3.3 s | 1.25 s |

All comfortably under ~2.3 s — consistent with, not worse than, the
pre-streaming-TTS baseline. Worth being honest about what this run does and
doesn't show: no reply in it ran long enough (max 7.2 s of audio) to surface
the dramatic gain the isolated 45.65 s-reply measurement above shows — `pcm`
vs `wav` barely differs on a short-to-medium reply, since there's little to
stream ahead of. The mechanism is confirmed genuinely active in production;
how much it saves on any given call still depends on how long the reply runs.

### GPU memory footprint

`rocm-smi --showpids` attributes VRAM per process directly — no estimation,
this is what the driver reports right now for the live `stt` and `tts`
containers on the reference deployment (RX 9060 XT, 16 GiB):

| Process | VRAM |
|---|---|
| `sofia-stt` | 1.85 GiB |
| `sofia-tts` | 2.12 GiB |
| **stt + tts** | **3.97 GiB** |
| GPU total used | 9.35 GiB / 15.92 GiB |

The ~5.4 GiB gap between "stt + tts" and "GPU total used" isn't these
services — this card is the host's own display GPU, not a headless
compute-only card, so part of that gap is the desktop compositor and whatever
else has a window open. Don't read "GPU total used" as this project's
footprint.

`sofia-stt`'s 1.85 GiB is two engines sharing one process, not separable via
`rocm-smi` alone: batch Nemotron (transformers/torch) and the parakeet.cpp/
ggml streaming engine both live in the same `stt` container when
`SOFIA_STT_BACKEND=both` (the default). Splitting that figure between the two
would need measuring with only one engine loaded at a time — not done here,
since the combined figure is what actually matters for capacity planning: it's
what one `stt` container costs on the card, regardless of which engine a given
request happens to use.

`sofia-tts` (Kokoro-82M) at 2.12 GiB is a single engine, so that number is
already the real per-engine cost.

Both comfortably fit alongside each other and the rest of this deployment on
a 16 GiB card, with room to spare — VRAM headroom is not the constraint here;
see [Turn latency](#headline-numbers) and the streaming-TTS section above for
what actually is.

### CPU busy-spin at idle (ROCm HSA runtime bug)

`sofia-tts` was found pinning one CPU core at ~100% continuously, including
with zero calls in progress — reproducible on every startup, not a
stuck-request artifact (confirmed: a plain container restart didn't clear
it, and CPU returned to ~100% within seconds even before any request was
served). `sofia-stt`, on the same ROCm-bundled torch, does not show this.

Root cause: a known, currently-open ROCm bug where the HSA runtime's
`AsyncEventsLoop` background thread busy-spins indefinitely after any GPU
operation, specific to the pip-bundled ROCm runtime (the wheel torch installs
its own copy of `libhsa-runtime64.so`, distinct from any system package).
`py-spy` can't see it at all — it's a thread the HSA runtime spawns directly,
never registered with CPython, invisible even with `--native`.

Fix: set `GPU_MAX_HW_QUEUES=1` on the `tts` service (`compose.yaml`).
Confirmed in an isolated standalone container first: CPU dropped from ~100%
to ~0.1–0.13% at idle with this set, synthesis output and duration unchanged,
GPU device still selected (`"device": "cuda"` in the ready log — ROCm's HIP
compat layer). Verified again after applying to the live `sofia-tts-1`
container: idle CPU stayed at ~0.12–0.13% across multiple samples, and a
real synthesis request over the production port produced correct audio with
no CPU regression afterward.

An earlier, more invasive workaround (`LD_PRELOAD`-ing the host's older
system `libhsa-runtime64.so` into the container) was abandoned partway
through: the host is Arch and the container is Debian-based, and the two
runtimes' own transitive dependencies (`librocprofiler-register.so.0`, then
`libfmt.so.12`) don't line up across distros — a real rabbit hole for no
benefit once the single env var was confirmed to work instead.

## The MIOpen story (why the first TTS benchmark looked terrible)

Before the fix, every *new* utterance length triggered a fresh MIOpen JIT
compile of Kokoro's LSTM kernels — measured 9–30 s, once per distinct sentence
length. Three layered fixes, all in place:

1. `torch.backends.cudnn.enabled = False` on ROCm builds (`tts_app.py`): the
   plain PyTorch LSTM fallback matches MIOpen's warm throughput here (Kokoro's
   LSTMs are tiny) and eliminates the per-shape cliffs entirely. CUDA builds
   keep cuDNN, whose kernels are precompiled.
2. `miopen-cache` volume: whatever MIOpen does compile (Nemotron's convs) is
   compiled once per deployment, not once per boot.
3. Multi-length boot warmups on both services so the first real user never pays
   for kernel compilation.

Measured after the fix: worst case for a previously-unseen sentence length is
**1.0 s total**, median RTF 0.13.

## Notes for reproduction

- Scripts used: `bench_components.py` (per-service HTTP/WS probes),
  `bench_e2e2.py` (WebRTC participant with frame-level logging),
  `twoclient.py` (two-client track-flow sanity check). They live outside the
  repo; the ops-level equivalent is `task ask` / `task say` plus the worker's
  own metrics log.
- Turn-latency runs measure the *agent's* audio frames, not local playback
  buffers; the greeting is waited out and its frames drained before the
  question is spoken.
- Numbers for the brain depend on the upstream endpoint's load — the TTFT range
  above (123–247 ms warm) was stable across the session, but treat it as
  endpoint-specific.

## Improvement levers, in order of expected payoff

1. **`SOFIA_STT_STREAMING=true`** — removes ~1.2 s/turn of transcription delay.
   Trade-off: English-only, no punctuation, lower accuracy — or use
   `SOFIA_STT_STREAMING_ENGINE=parakeet` for streaming with neither trade-off,
   at the cost of needing a GPU (see `add-parakeet-streaming-asr`).
2. ~~Streaming TTS~~ — done (`add-streaming-tts`): ~60% cut in time-to-first-
   audio for long replies, see above.
3. **Local LLM** — the brain is a remote 30B-A3B model; TTFT is already small
   relative to the turn budget, but a local deployment removes a network hop
   and the tail (247 ms).
4. `SOFIA_STT_ENDPOINT_SILENCE` / `min_endpointing_delay` are already at sane
   floors (0.8 s ASR-side, 0.4 s agent-side); lowering them further trades
   interruption robustness for ~100 ms.
