"""The LiveKit worker: joins a room, listens, and speaks.

This module contains no LLM logic on purpose. It wires four interchangeable
boxes together — VAD, STT, "LLM", TTS — where the "LLM" happens to be
qaa-agent.service behind an OpenAI-compatible URL. Everything smart lives there;
everything real-time lives here.

That split is what keeps this file hardware-agnostic: it holds no model weights
and needs no GPU, so the same image runs unchanged on CUDA and ROCm hosts.
"""

from __future__ import annotations

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    metrics,
)
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from sofia_galileo.core.logging import get_logger
from sofia_galileo.s2s.config import S2SSettings

log = get_logger(__name__)


class SofiaAgent(Agent):
    """A thin Agent shell. The persona and the speech rules live in qaa-agent.

    The instructions here are only room-local context; qaa-agent stacks its own
    system prompt in front of them, so the two never compete.
    """

    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)


def prewarm(proc: JobProcess) -> None:
    """Load Silero VAD once per worker process, not once per call.

    Without this the first participant of every job pays the model load time
    before the agent can hear anything.
    """
    proc.userdata["vad"] = silero.VAD.load()


def _build_session(settings: S2SSettings, vad: silero.VAD, room_name: str) -> AgentSession:
    return AgentSession(
        vad=vad,
        stt=lk_openai.STT(
            model=settings.stt_model,
            language=settings.language,
            base_url=settings.stt_base_url,
            api_key=settings.stt_api_key,
            # Streaming ASR gives interim transcripts, which the semantic turn
            # detector uses to decide the person is finished — so this changes
            # how the agent listens, not just how fast it transcribes.
            use_realtime=settings.stt_use_realtime,
        ),
        llm=lk_openai.LLM(
            model=settings.qaa_model,
            base_url=settings.qaa_base_url,
            api_key=settings.qaa_api_key,
            # Tags every turn with its room, so one call's turns are greppable in
            # qaa-agent's logs; the body fields become session context there.
            extra_headers={"X-Sofia-Room": room_name},
            extra_body={"sofia_room": room_name, "sofia_language": settings.language},
        ),
        tts=lk_openai.TTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            base_url=settings.tts_base_url,
            api_key=settings.tts_api_key,
            # Must be set explicitly: the plugin defaults to mp3, which our TTS
            # service does not encode. wav carries its own sample rate, so a
            # mismatch is impossible.
            response_format=settings.tts_response_format,
        ),
        # Semantic end-of-turn detection: decides the person is finished from what
        # they said, not just from how long they have been quiet. This is what
        # stops the agent interrupting someone who is mid-thought.
        turn_detection=MultilingualModel(),
        min_endpointing_delay=settings.min_endpointing_delay,
        max_endpointing_delay=settings.max_endpointing_delay,
    )


def _room_context(settings: S2SSettings, room_name: str) -> str:
    return f"{settings.instructions}\nRoom: {room_name}. Preferred language: {settings.language}."


async def entrypoint(ctx: JobContext) -> None:
    settings = S2SSettings()
    room_name = ctx.job.room.name if ctx.job and ctx.job.room else "unknown"
    log.info("job.accepted", room=room_name)

    session = _build_session(settings, ctx.proc.userdata["vad"], room_name)

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        # Per-stage latency (STT, LLM time-to-first-token, TTS) — the numbers you
        # actually need when a conversation feels sluggish.
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def _log_usage() -> None:
        log.info("job.finished", room=room_name, usage=usage.get_summary())

    ctx.add_shutdown_callback(_log_usage)

    await session.start(
        agent=SofiaAgent(_room_context(settings, room_name)),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )
    await ctx.connect()

    if settings.greeting:
        await session.generate_reply(instructions=settings.greeting)


def worker_options() -> WorkerOptions:
    settings = S2SSettings()
    options = WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    if settings.agent_name:
        options.agent_name = settings.agent_name
    return options
