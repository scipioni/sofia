"""FastAPI surface of qaa-agent.service.

Speaks the OpenAI Chat Completions protocol so that s2s.service can point
`livekit.plugins.openai.LLM` straight at it. Swapping this service out for the
raw upstream LLM is a one-line env change, which makes it easy to tell whether a
bad conversation is the model's fault or ours.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sofia_galileo.core.logging import configure_logging, get_logger
from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.engine import Done, QaaEngine, TextDelta, ToolCallsDelta
from sofia_galileo.qaa.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ModelCard,
    ModelList,
    ToolCall,
)
from sofia_galileo.qaa.tools import build_default_toolset

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: QaaSettings = app.state.settings
    toolset = build_default_toolset()
    app.state.engine = QaaEngine(settings, toolset)
    log.info(
        "qaa.started",
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        served_model_name=settings.served_model_name,
        tools=len(toolset.tools) if settings.tools_enabled else 0,
    )
    try:
        yield
    finally:
        await app.state.engine.aclose()
        log.info("qaa.stopped")


def create_app(settings: QaaSettings | None = None) -> FastAPI:
    settings = settings or QaaSettings()
    configure_logging("qaa-agent", settings.log_level, json_logs=settings.json_logs)

    app = FastAPI(
        title="sofia-galileo qaa-agent",
        version="0.1.0",
        summary="OpenAI-compatible reasoning brain for the Sofia voice agent",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def bind_request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        structlog.contextvars.bind_contextvars(
            request_id=request.headers.get("x-request-id", uuid.uuid4().hex[:12]),
            room=request.headers.get("x-sofia-room"),
        )
        started = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            if request.url.path.endswith("/chat/completions"):
                log.info(
                    "turn.handled",
                    elapsed_ms=round((time.perf_counter() - started) * 1000),
                )
            structlog.contextvars.clear_contextvars()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=settings.served_model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):  # type: ignore[no-untyped-def]
        engine: QaaEngine = request.app.state.engine
        model = settings.served_model_name

        if not req.messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "messages must not be empty",
                        "type": "invalid_request_error",
                    }
                },
            )

        if req.stream:
            return StreamingResponse(
                _sse(engine, req, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        text, tool_calls, finish_reason, usage = await engine.complete(req)
        message = ChatMessage(role="assistant", content=text or None)
        if tool_calls:
            message.tool_calls = [ToolCall.model_validate(call) for call in tool_calls]
        return ChatCompletionResponse(
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    return app


async def _sse(engine: QaaEngine, req: ChatCompletionRequest, model: str) -> AsyncIterator[str]:
    """Render engine events as an OpenAI-compatible SSE stream."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def chunk(delta: ChoiceDelta, finish_reason: str | None = None) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            model=model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

    # The role-only opening chunk is what OpenAI sends; some clients rely on it.
    yield chunk(ChoiceDelta(role="assistant"))

    try:
        async for event in engine.stream(req):
            match event:
                case TextDelta(text=text):
                    yield chunk(ChoiceDelta(content=text))
                case ToolCallsDelta(tool_calls=calls):
                    yield chunk(
                        ChoiceDelta(
                            tool_calls=[
                                ToolCall.model_validate({**call, "index": index})
                                for index, call in enumerate(calls)
                            ]
                        )
                    )
                case Done(finish_reason=reason):
                    yield chunk(ChoiceDelta(), finish_reason=reason)
    except Exception:
        log.exception("stream.failed")
        yield chunk(ChoiceDelta(), finish_reason="stop")

    yield "data: [DONE]\n\n"


__all__ = ["create_app"]
