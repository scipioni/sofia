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

### Text-to-speech (Kokoro-82M, GPU)

| Utterance | Wall | RTF |
|---|---|---|
| ~2 s audio (short) | 0.49 s | 0.28 |
| ~5 s audio (medium) | 0.78 s | 0.15 |
| ~8 s audio (long) | 1.0 s | 0.13 |

Response is **non-streaming**: the service synthesises the whole text before
answering, so TTFB ≈ full synthesis of the sentence. The LiveKit plugin requests
per sentence, which caps the damage at ~1 s per sentence — this is the
explanation for the one 7.6 s outlier observed with a 6.2 s single-sentence
reply during early (pre-fix) runs.

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
   Trade-off: English-only, no punctuation, lower accuracy.
2. **Streaming TTS** — chunked synthesis would cut the per-sentence tail
   (~1 s) and make long single sentences start speaking sooner.
3. **Local LLM** — the brain is a remote 30B-A3B model; TTFT is already small
   relative to the turn budget, but a local deployment removes a network hop
   and the tail (247 ms).
4. `SOFIA_STT_ENDPOINT_SILENCE` / `min_endpointing_delay` are already at sane
   floors (0.8 s ASR-side, 0.4 s agent-side); lowering them further trades
   interruption robustness for ~100 ms.
