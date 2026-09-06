"""OpenAI-compatible speech-to-text.

Two endpoints, two backends:

  POST /v1/audio/transcriptions   batch — Nemotron 3.5 ASR, or Whisper
  WS   /v1/realtime               streaming — sherpa-onnx, OpenAI realtime format

Which one s2s uses is its choice; serving both means switching is a restart.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import anyio.to_thread
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse

from sofia_galileo.audio.batch import NemotronTranscriber, build_transcriber
from sofia_galileo.audio.config import SttSettings
from sofia_galileo.audio.realtime import serve_realtime
from sofia_galileo.core.logging import configure_logging, get_logger

log = get_logger(__name__)


def _tone_wav(seconds: float, sample_rate: int = 16000) -> bytes:
    """A quiet 220 Hz tone as wav bytes — realistic input for kernel warm-up."""
    import io

    import numpy as np
    import soundfile as sf

    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: SttSettings = app.state.settings

    if settings.backend in ("batch", "both"):
        # Model load can block on a multi-gigabyte download; keep the loop free
        # so the healthcheck answers "loading" rather than timing out.
        transcriber = await anyio.to_thread.run_sync(build_transcriber, settings)
        app.state.transcriber = transcriber
        if isinstance(transcriber, NemotronTranscriber):
            # Warm the conv kernels on the shapes of a typical short turn: MIOpen
            # (AMD) JIT-compiles per input length, which would otherwise stall
            # the first real transcription. Tone, not silence — silences take
            # the padded fast path and compile nothing.
            await anyio.to_thread.run_sync(transcriber.transcribe, _tone_wav(2.0), None)
    if settings.backend in ("streaming", "both"):
        from sofia_galileo.audio.streaming import build_recognizer

        # Loading blocks on a download the first time; keep the loop free so the
        # healthcheck can still answer "loading" instead of timing out. A
        # failure here (missing library, bad weights, unknown engine) propagates
        # out of the lifespan context and the service does not come up — no
        # silent fallback to a different recogniser than the one configured
        # (design.md D8 in add-parakeet-streaming-asr).
        app.state.recognizer = await anyio.to_thread.run_sync(build_recognizer, settings)

    log.info("stt.ready", port=settings.port, backend=settings.backend)
    yield
    app.state.transcriber = None
    app.state.recognizer = None


def create_app(settings: SttSettings | None = None) -> FastAPI:
    settings = settings or SttSettings()
    configure_logging("stt", settings.log_level, json_logs=settings.json_logs)

    if settings.backend not in ("batch", "streaming", "both"):
        raise ValueError(
            f"SOFIA_STT_BACKEND must be batch, streaming or both; got {settings.backend!r}"
        )

    app = FastAPI(title="sofia-galileo stt", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.transcriber = None
    app.state.recognizer = None

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        wants_batch = settings.backend in ("batch", "both")
        wants_streaming = settings.backend in ("streaming", "both")
        ready = (app.state.transcriber is not None or not wants_batch) and (
            app.state.recognizer is not None or not wants_streaming
        )
        return {"status": "ok" if ready else "loading"}

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        """Streaming ASR, in OpenAI's realtime-transcription wire format.

        s2s reaches this by setting SOFIA_S2S_STT_USE_REALTIME=true; the plugin
        derives the ws URL from the same base_url the batch endpoint uses.
        """
        if app.state.recognizer is None:
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": "streaming backend is not enabled; "
                        "set SOFIA_STT_BACKEND=streaming or both"
                    },
                }
            )
            await websocket.close(code=1011)
            return
        await serve_realtime(websocket, app.state.recognizer)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(  # type: ignore[no-untyped-def]
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "whisper-1",
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
    ):
        if app.state.transcriber is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "batch backend is not enabled; "
                        "set SOFIA_STT_BACKEND=batch or both"
                    }
                },
            )

        audio = await file.read()
        if not audio:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "empty audio upload"}},
            )

        # `prompt` and `temperature` are accepted for API compatibility and
        # ignored: neither engine takes free-text prompting the way the OpenAI
        # endpoint does, and passing them through would raise.
        del prompt, temperature

        # Inference is synchronous and compute-bound; hand it to a worker thread
        # so one long utterance cannot stall the loop for other requests.
        try:
            text = await anyio.to_thread.run_sync(
                app.state.transcriber.transcribe, audio, language or settings.default_language
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})

        log.info("stt.transcribed", chars=len(text), bytes=len(audio), model=model)

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text}

    return app
