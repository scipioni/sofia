# sofia-galileo — quick start

Fastest path from a clean checkout to a human talking to Sofia. For the
architecture and trade-offs, see [README.md](../README.md) and
[docs/details.md](details.md).

## 1. Prerequisites

- Docker with Compose v2, and [go-task](https://taskfile.dev)
- A LiveKit server — [LiveKit Cloud](https://cloud.livekit.io) (free tier is
  fine) or your own, either works identically
- An OpenAI-compatible LLM endpoint reachable from the containers (a hosted
  API, or something like Ollama/vLLM on the host — use
  `http://host.docker.internal:PORT/v1` for the latter)
- One of: NVIDIA Container Toolkit, a ROCm host (`/dev/kfd` + `/dev/dri`), or
  nothing at all (CPU profile — slower, but every model here runs on CPU)

## 2. Configure

```bash
cp .env.example .env
```

Fill in `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` and
`SOFIA_QAA_LLM_BASE_URL` / `SOFIA_QAA_LLM_MODEL`. Everything else has a
working default.

## 3. Bring the stack up

```bash
task up:rocm      # or: task up:nvidia | task up:cpu
```

First boot downloads model weights (~1.3 GB batch ASR, ~310 MB for sherpa or ~640 MB for parakeet streaming ASR,
~330 MB TTS) into `./data/` — the healthcheck allows ten minutes for this, it
only happens once. `task up:*` runs detached (`up -d`) and returns once
everything is scheduled; use `task logs -- <service>` to follow a service's
logs, and `docker compose ps` / `docker ps` to check status.

Confirm everything is actually healthy:

```bash
docker compose ps
task ask Q="who are you?"   # talks to the brain directly, skips STT/TTS
```

## 4. Make the first call with a human

The `s2s` worker joins **every** room on your LiveKit project by default (no
`SOFIA_S2S_AGENT_NAME` set) — you don't dispatch it, you just need a human in
a room.

**LiveKit Cloud** — open the
[Agents Playground](https://agents-playground.livekit.io/), connect it to
your Cloud project (same `LIVEKIT_URL`), grant microphone access, and speak.
The worker joins automatically within a couple of seconds.

**Self-hosted LiveKit** — mint a token with the
[LiveKit CLI](https://github.com/livekit/livekit-cli) and hand it to the
playground manually:

```bash
lk token create \
  --api-key "$LIVEKIT_API_KEY" --api-secret "$LIVEKIT_API_SECRET" \
  --join --room lobby --identity human --valid-for 24h
```

Open the [Agents Playground](https://agents-playground.livekit.io/), pick its
manual/custom connect option, paste your `LIVEKIT_URL` and the token above.

Either way: there's no push-to-talk. VAD and semantic turn detection decide
when you've finished speaking — just talk, then stop, and wait for the reply.

## Troubleshooting

| Symptom | Look at |
|---|---|
| `stt`/`tts` stuck "starting" | First-boot download in progress — `task logs -- stt` |
| Worker joins, never replies | Brain reachable? `task ask Q="hi"`. If that hangs, `SOFIA_QAA_LLM_BASE_URL` isn't reachable from inside the container |
| Reply text is right, no audio | `task say T="ciao"` in isolation — check `tts` logs, and that `SOFIA_TTS_VOICE`'s language matches what's being said |
| Wrong language transcribed/spoken | `SOFIA_LANGUAGE` (context + STT locale) and `SOFIA_TTS_VOICE` (first letter picks the voice's language) |
| Long pause before the agent responds | `SOFIA_STT_ENDPOINT_SILENCE` (default 0.8 s) is the main latency dial — see [Streaming vs batch ASR](details.md#streaming-vs-batch-asr) |

Sources: [LiveKit Agents Playground](https://agents-playground.livekit.io/),
[LiveKit CLI](https://github.com/livekit/livekit-cli).
