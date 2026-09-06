"""The reasoning loop behind qaa-agent.

Responsibilities:
  * own the system prompt (the caller's own system message is kept as extra context)
  * call the upstream OpenAI-compatible LLM
  * run *server-side* tools (ours) inline, transparently to the caller
  * forward *client-side* tools (LiveKit `function_tool`s declared in s2s) back to
    the caller to execute, exactly as a real OpenAI endpoint would

The loop itself is owned by pydantic-ai (major-pinned): model requests, tool
dispatch, streamed fragment accumulation, retry semantics and the request
budget are all framework territory now. What remains hand-written here is the
protocol glue — OpenAI wire messages in, framework messages out, and our three
stream events back — plus the voice-agent policies the framework cannot know:
the round cap, the never-silent failure path, and the token ceiling.

Streaming is the point of this module. Text deltas are emitted the instant they
arrive so s2s can start speaking on the first sentence instead of waiting for
the whole answer — in a voice agent that is the difference between a natural
reply and an awkward pause.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    ExternalToolset,
    FunctionToolset,
    ModelRequest,
    ModelRequestNode,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ImageUrl, ModelResponsePart, UserContent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits

from sofia_galileo.core.logging import get_logger
from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.schemas import ChatCompletionRequest, Usage
from sofia_galileo.qaa.tools import build_default_toolset

log = get_logger(__name__)

# Spoken to the user when the upstream LLM is unreachable. Silence is the worst
# possible failure mode for a voice agent, so we always say *something*.
UPSTREAM_ERROR_REPLY = "Sorry, I'm having trouble thinking right now. Could you try again?"

TOOL_ROUNDS_FALLBACK = "Sorry, I couldn't work that one out."

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class TextDelta:
    """A piece of assistant text, ready to be spoken."""

    text: str


@dataclass
class ToolCallsDelta:
    """Tool calls the *caller* must execute (LiveKit-side function tools)."""

    tool_calls: list[dict[str, Any]]


@dataclass
class Done:
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


StreamEvent = TextDelta | ToolCallsDelta | Done


# -- wire -> framework adapters --------------------------------------------


def _text_content(content: str | list[dict[str, Any]] | None) -> str:
    """Flatten any message content to plain text (system/assistant direction)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _user_content(content: str | list[dict[str, Any]] | None) -> UserContent:
    if not isinstance(content, list):
        return content or ""
    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append(TextPart(content=str(part.get("text", ""))))
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                parts.append(ImageUrl(url=str(url)))
            else:
                log.warning("content.part.unmapped", part_type=kind)
        else:
            # Unknown part types have no framework equivalent; degrade to JSON
            # text rather than silently dropping information the model may need.
            log.warning("content.part.unmapped", part_type=kind)
            parts.append(TextPart(content=json.dumps(part)))
    return parts or ""


def _call_args(raw: str | None) -> str | dict[str, Any]:
    """Framework tool-call args: a parsed object, or the raw string when it isn't one."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, dict) else raw


def build_message_history(req: ChatCompletionRequest) -> list[ModelRequest | ModelResponse]:
    """Translate the caller's OpenAI message list into framework messages.

    Sofia's session context goes in first (as its own system message); the
    caller's own system messages are kept, not replaced. Tool results arrive
    over the wire without the tool name, so names are recovered from the
    assistant tool_calls that preceded them — upstream generated those ids.
    """
    history: list[ModelRequest | ModelResponse] = []
    context = session_context(req)
    if context:
        history.append(ModelRequest(parts=[SystemPromptPart(content=context)]))

    tool_names: dict[str, str] = {}
    for message in req.messages:
        if message.role in ("system", "developer"):
            history.append(
                ModelRequest(parts=[SystemPromptPart(content=_text_content(message.content))])
            )
        elif message.role == "user":
            history.append(
                ModelRequest(parts=[UserPromptPart(content=_user_content(message.content))])
            )
        elif message.role == "assistant":
            parts: list[ModelResponsePart] = []
            if isinstance(message.content, str) and message.content:
                parts.append(TextPart(content=message.content))
            elif isinstance(message.content, list):
                text = _text_content(message.content)
                if text:
                    parts.append(TextPart(content=text))
            for call in message.tool_calls or []:
                name = call.function.name or ""
                call_id = call.id or f"call_{len(tool_names)}"
                tool_names[call_id] = name
                parts.append(
                    ToolCallPart(
                        tool_name=name,
                        args=_call_args(call.function.arguments),
                        tool_call_id=call_id,
                    )
                )
            history.append(ModelResponse(parts=parts))
        elif message.role == "tool":
            content = (
                message.content if isinstance(message.content, str) else json.dumps(message.content)
            )
            history.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=tool_names.get(message.tool_call_id or "", "unknown_tool"),
                            content=content or "",
                            tool_call_id=message.tool_call_id or "",
                        )
                    ]
                )
            )
    return history


def session_context(req: ChatCompletionRequest) -> str | None:
    parts = []
    if req.sofia_language:
        parts.append(f"The person is speaking {req.sofia_language}.")
    if req.sofia_participant:
        parts.append(f"You are speaking with participant '{req.sofia_participant}'.")
    return " ".join(parts) if parts else None


def external_toolset_from_wire(tools: list[dict[str, Any]] | None) -> ExternalToolset | None:
    """The caller's declared tools as schema-only entries we never execute.

    When the model calls one, the run ends with the call surfaced back to the
    caller — the framework's deferred-tool flow is a 1:1 match for how a real
    OpenAI endpoint hands function calls back. `strict=False` keeps upstream
    strict-mode inference from rewriting the caller's schema behind our back.
    """
    defs = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            log.warning("tools.unsupported_entry")
            continue
        fn = tool.get("function")
        if not fn or not fn.get("name"):
            log.warning("tools.unsupported_entry")
            continue
        defs.append(
            ToolDefinition(
                name=fn["name"],
                description=fn.get("description"),
                parameters_json_schema=fn.get("parameters") or _EMPTY_SCHEMA,
                strict=False,
            )
        )
    return ExternalToolset(defs) if defs else None


def deferred_calls_to_openai(calls: list[ToolCallPart]) -> list[dict[str, Any]]:
    """Deferred tool calls back into OpenAI-format dicts for the caller.

    Args come back from the framework validated (a dict), so they are
    re-serialized: the arguments string may differ cosmetically from what the
    model emitted, but it is always valid JSON for the declared schema.
    """
    return [
        {
            "id": call.tool_call_id,
            "type": "function",
            "function": {
                "name": call.tool_name,
                "arguments": call.args
                if isinstance(call.args, str)
                else json.dumps(call.args or {}),
            },
        }
        for call in calls
    ]


def _as_usage(run_usage: RunUsage) -> Usage:
    return Usage(
        prompt_tokens=run_usage.input_tokens,
        completion_tokens=run_usage.output_tokens,
        total_tokens=run_usage.total_tokens,
    )


def upstream_model(settings: QaaSettings, client: AsyncOpenAI) -> OpenAIChatModel:
    """The upstream model, built to keep the wire identical to pre-framework days.

    By default pydantic-ai renders the token ceiling as `max_completion_tokens`
    (OpenAI's modern field); Sofia's actual upstreams — vLLM, llama.cpp,
    Ollama, TGI — are lowest-common-denominator OpenAI servers that only
    promise `max_tokens`, so the profile pins the legacy mapping.
    """
    return OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(openai_client=client),
        profile={"openai_chat_supports_max_completion_tokens": False},
    )


class QaaEngine:
    def __init__(
        self,
        settings: QaaSettings,
        toolset: FunctionToolset | None = None,
        *,
        model: OpenAIChatModel | None = None,
    ) -> None:
        self._settings = settings
        self._toolset = toolset if toolset is not None else build_default_toolset()
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        self._model = model or upstream_model(settings, self._client)
        # end_strategy is the framework default today; pinned explicitly because
        # a text-then-tool-call response must still run its tools — the text was
        # already streamed (and is being spoken) by the time the calls arrive.
        self._agent = Agent(
            self._model,
            instructions=settings.system_prompt,
            toolsets=[self._toolset] if settings.tools_enabled else None,
            output_type=[str, DeferredToolRequests],
            end_strategy="graceful",
        )

    async def aclose(self) -> None:
        await self._client.close()

    # -- request shaping ---------------------------------------------------

    def _model_settings(self, req: ChatCompletionRequest) -> OpenAIChatModelSettings:
        settings = OpenAIChatModelSettings(
            temperature=req.temperature
            if req.temperature is not None
            else self._settings.temperature,
            top_p=req.top_p if req.top_p is not None else self._settings.top_p,
            # The caller may ask for fewer tokens, never more: long replies are a
            # bug in a spoken conversation, not a feature.
            max_tokens=min(req.max_tokens or self._settings.max_tokens, self._settings.max_tokens),
        )
        if req.stop:
            settings["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else list(req.stop)
        if req.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = req.parallel_tool_calls
        if req.tool_choice is not None:
            if req.tool_choice in ("auto", "none"):
                settings["tool_choice"] = req.tool_choice
            else:
                # 'required' and structured forms collide with the framework's
                # deferred-tool output mode (a forced tool call can't coexist
                # with a run that may end in a handback) — dropping it beats
                # failing the whole upstream request.
                log.warning("tool_choice.dropped", tool_choice=req.tool_choice)
        return settings

    # -- streaming ---------------------------------------------------------

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        external = external_toolset_from_wire(req.tools)
        try:
            async with self._agent.iter(
                message_history=build_message_history(req),
                toolsets=[external] if external else None,
                # One upstream request per round; the +1 is the initial call.
                usage_limits=UsageLimits(request_limit=self._settings.max_tool_rounds + 1),
                model_settings=self._model_settings(req),
            ) as run:
                # Only text from the *current* upstream request counts for the
                # failure guard: an apology must never be stitched onto a
                # half-finished sentence from an earlier round.
                streamed_this_request = False
                async for node in run:
                    if isinstance(node, ModelRequestNode):
                        streamed_this_request = False
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                # The first text fragment arrives as a started
                                # part, its continuations as deltas; both must
                                # reach the caller the instant they arrive.
                                if isinstance(event, PartStartEvent) and isinstance(
                                    event.part, TextPart
                                ):
                                    if event.part.content:
                                        streamed_this_request = True
                                        yield TextDelta(event.part.content)
                                elif isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    if event.delta.content_delta:
                                        streamed_this_request = True
                                        yield TextDelta(event.delta.content_delta)

                result = run.result
                usage = _as_usage(result.usage)
                if isinstance(result.output, DeferredToolRequests):
                    # Anything the caller declared is theirs to run: hand it
                    # back and stop, exactly as a real OpenAI endpoint would.
                    yield ToolCallsDelta(deferred_calls_to_openai(result.output.calls))
                    yield Done(finish_reason="tool_calls", usage=usage)
                    return
                yield Done(finish_reason="stop", usage=usage)
                return
        except UsageLimitExceeded:
            log.warning("tool.rounds_exhausted", limit=self._settings.max_tool_rounds)
            yield TextDelta(TOOL_ROUNDS_FALLBACK)
            yield Done(finish_reason="stop")
        except (OpenAIError, ModelHTTPError) as exc:
            log.error("upstream.failed", error=str(exc))
            if not streamed_this_request:
                yield TextDelta(UPSTREAM_ERROR_REPLY)
            yield Done(finish_reason="stop")

    # -- non-streaming -----------------------------------------------------

    async def complete(
        self, req: ChatCompletionRequest
    ) -> tuple[str, list[dict[str, Any]], str, Usage]:
        """Collect the stream into one response: (text, tool_calls, finish_reason, usage)."""
        text: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        finish_reason = "stop"
        usage = Usage()

        async for event in self.stream(req):
            match event:
                case TextDelta(text=delta):
                    text.append(delta)
                case ToolCallsDelta(tool_calls=calls):
                    tool_calls = calls
                case Done(finish_reason=reason, usage=collected):
                    finish_reason = reason
                    usage = collected

        return "".join(text), tool_calls, finish_reason, usage
