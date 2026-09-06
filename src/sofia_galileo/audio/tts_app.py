"""OpenAI-compatible text-to-speech: POST /v1/audio/speech.

Backed by Kokoro-82M — small enough to be fast on modest hardware, good enough
to sound like a person. The LiveKit OpenAI TTS plugin asks for raw PCM, which is
also the cheapest path here: no container, no re-encode, straight into the room.

Kokoro derives its language from the voice name: the first letter of a voice id
selects the pipeline ('a' American English, 'b' British, 'i' Italian, 'f' French,
'e' Spanish, 'p' Portuguese, 'h' Hindi, 'j' Japanese, 'z' Mandarin).
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from sofia_galileo.audio.config import TtsSettings, resolve_device
from sofia_galileo.core.logging import configure_logging, get_logger

log = get_logger(__name__)

# Formats we can actually produce. mp3/opus/aac are deliberately absent: encoding
# them depends on how libsndfile was built, and a voice agent that silently ships
# an undecodable body is worse than one that says no. s2s asks for wav.
MEDIA_TYPES = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "flac": "audio/flac",
}


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str
    model: str = "kokoro"
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


class KokoroEngine:
    """Holds one Kokoro pipeline per language, built lazily on first use."""

    def __init__(self, settings: TtsSettings) -> None:
        import torch

        if torch.version.hip:
            # MIOpen JIT-compiles its RNN kernels per sequence length, so every
            # new sentence length stalls synthesis for 10-30 s while clang
            # builds a shape-specific kernel — fatal for a voice agent, where
            # no two replies have the same length. Kokoro's LSTMs are tiny and
            # the plain PyTorch fallback matches MIOpen's warm throughput with
            # no cliffs, so on ROCm we bypass it entirely. CUDA builds keep
            # cuDNN: its kernels are precompiled and fast.
            torch.backends.cudnn.enabled = False

        self._settings = settings
        self._device = resolve_device(settings.device)
        self._pipelines: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._device

    def _pipeline(self, lang_code: str) -> Any:
        from kokoro import KPipeline

        if lang_code not in self._pipelines:
            log.info("tts.loading", lang_code=lang_code, device=self._device)
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device=self._device)
        return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        pipeline = self._pipeline(voice[0])
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in pipeline(text, voice=voice, speed=speed)
            if audio is not None
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def _encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")

    if fmt == "pcm":
        # Raw signed 16-bit little-endian mono, no header — what LiveKit wants.
        return pcm16.tobytes()

    import soundfile as sf

    buffer = io.BytesIO()
    subtype = {"wav": "PCM_16", "flac": "PCM_16"}.get(fmt)
    sf.write(buffer, pcm16, sample_rate, format=fmt.upper(), subtype=subtype)
    return buffer.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: TtsSettings = app.state.settings
    engine = KokoroEngine(settings)
    app.state.engine = engine
    # Warm the default voice at several utterance lengths: MIOpen (AMD) JIT-
    # compiles LSTM kernels per sequence-length shape, and without this the
    # first real reply of each new length stalls for 10+ seconds mid-call.
    # Repeats and cached shapes cost milliseconds.
    await anyio.to_thread.run_sync(_warm_up, engine, settings)
    log.info("tts.ready", port=settings.port, device=engine.device)
    yield
    app.state.engine = None


def _warm_up(engine: KokoroEngine, settings: TtsSettings) -> None:
    for sentence in (
        "Hello.",
        "This is a warm-up sentence of medium length for the synthesiser.",
        "And this is a longer warm-up paragraph; it exists so that the GPU has "
        "already compiled its kernels for the longer sentences a person will "
        "hear during a real conversation, not just for short greetings.",
    ):
        engine.synthesize(sentence, settings.default_voice, 1.0)


def create_app(settings: TtsSettings | None = None) -> FastAPI:
    settings = settings or TtsSettings()
    configure_logging("tts", settings.log_level, json_logs=settings.json_logs)

    app = FastAPI(title="sofia-galileo tts", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok" if app.state.engine is not None else "loading"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):  # type: ignore[no-untyped-def]
        fmt = req.response_format.lower()
        if fmt not in MEDIA_TYPES:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"unsupported response_format {fmt!r}; "
                        f"expected one of {sorted(MEDIA_TYPES)}"
                    }
                },
            )

        text = req.input.strip()
        if not text:
            return Response(content=b"", media_type=MEDIA_TYPES[fmt])

        voice = req.voice or settings.default_voice
        engine: KokoroEngine = app.state.engine
        audio = await anyio.to_thread.run_sync(lambda: engine.synthesize(text, voice, req.speed))
        payload = _encode(audio, settings.sample_rate, fmt)

        log.info(
            "tts.synthesized",
            chars=len(text),
            voice=voice,
            fmt=fmt,
            seconds=round(len(audio) / settings.sample_rate, 2),
        )
        return Response(content=payload, media_type=MEDIA_TYPES[fmt])

    return app
