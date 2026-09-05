"""The reasoning loop behind qaa-agent.

Responsibilities:
  * own the system prompt (the caller's own system message is kept as extra context)
  * call the upstream OpenAI-compatible LLM
  * run *server-side* tools (ours) inline, transparently to the caller
  * forward *client-side* tools (LiveKit `function_tool`s declared in s2s) back to
    the caller to execute, exactly as a real OpenAI endpoint would

Streaming is the point of this module. Text deltas are emitted the instant they
arrive so s2s can start speaking on the first sentence instead of waiting for the
whole answer — in a voice agent that is the difference between a natural reply
and an awkward pause.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from sofia_galileo.core.logging import get_logger
from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.schemas import ChatCompletionRequest, Usage
from sofia_galileo.qaa.tools import ToolRegistry

log = get_logger(__name__)

# Spoken to the user when the upstream LLM is unreachable. Silence is the worst
# possible failure mode for a voice agent, so we always say *something*.
UPSTREAM_ERROR_REPLY = "Sorry, I'm having trouble thinking right now. Could you try again?"


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


def _accumulate_tool_calls(acc: dict[int, dict[str, Any]], deltas: list[Any]) -> None:
    """Fold streamed tool-call fragments into complete calls, keyed by index."""
    for delta in deltas:
        index = delta.index if delta.index is not None else len(acc)
        slot = acc.setdefault(index, {"id": None, "type": "function", "name": "", "arguments": ""})
        if delta.id:
            slot["id"] = delta.id
        fn = getattr(delta, "function", None)
        if fn is not None:
            # Most servers send the name whole in the first fragment, but a few
            # split it; concatenating is correct either way.
            if fn.name:
                slot["name"] += fn.name
            if fn.arguments:
                slot["arguments"] += fn.arguments


def _as_openai_tool_calls(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": call["id"] or f"call_{index}",
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
        }
        for index, call in sorted(acc.items())
    ]


class QaaEngine:
    def __init__(self, settings: QaaSettings, registry: ToolRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # -- request shaping ---------------------------------------------------

    def _build_messages(self, req: ChatCompletionRequest) -> list[dict[str, Any]]:
        """Sofia's system prompt first; the caller's own system message is kept.

        s2s passes a short delivery-oriented instruction of its own. Rather than
        fighting over which one wins, we stack them: ours sets the persona and the
        speech rules, the caller's adds room-specific context.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._settings.system_prompt}
        ]
        for message in req.messages:
            messages.append(message.model_dump(exclude_none=True))

        context = self._session_context(req)
        if context:
            messages.insert(1, {"role": "system", "content": context})
        return messages

    @staticmethod
    def _session_context(req: ChatCompletionRequest) -> str | None:
        parts = []
        if req.sofia_language:
            parts.append(f"The person is speaking {req.sofia_language}.")
        if req.sofia_participant:
            parts.append(f"You are speaking with participant '{req.sofia_participant}'.")
        return " ".join(parts) if parts else None

    def _tool_specs(self, req: ChatCompletionRequest) -> list[dict[str, Any]] | None:
        """Our tools plus whatever the caller declared, in one list for upstream."""
        specs: list[dict[str, Any]] = list(req.tools or [])
        if self._settings.tools_enabled:
            specs.extend(self._registry.specs())
        return specs or None

    async def _upstream(
        self, messages: list[dict[str, Any]], req: ChatCompletionRequest, *, stream: bool
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": stream,
            "temperature": (
                req.temperature if req.temperature is not None else self._settings.temperature
            ),
            "top_p": req.top_p if req.top_p is not None else self._settings.top_p,
            # The caller may ask for fewer tokens, never more: long replies are a
            # bug in a spoken conversation, not a feature.
            "max_tokens": min(
                req.max_tokens or self._settings.max_tokens, self._settings.max_tokens
            ),
        }
        if req.stop:
            kwargs["stop"] = req.stop
        tools = self._tool_specs(req)
        if tools:
            kwargs["tools"] = tools
            if req.tool_choice is not None:
                kwargs["tool_choice"] = req.tool_choice
        return await self._client.chat.completions.create(**kwargs)

    # -- streaming ---------------------------------------------------------

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(req)
        client_tool_names = {tool.get("function", {}).get("name") for tool in (req.tools or [])}

        for round_index in range(self._settings.max_tool_rounds + 1):
            text_buffer: list[str] = []
            tool_acc: dict[int, dict[str, Any]] = {}
            usage = Usage()

            try:
                stream = await self._upstream(messages, req, stream=True)
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = Usage.model_validate(chunk.usage.model_dump())
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if delta.content:
                        text_buffer.append(delta.content)
                        yield TextDelta(delta.content)
                    if delta.tool_calls:
                        _accumulate_tool_calls(tool_acc, delta.tool_calls)
            except OpenAIError as exc:
                log.error("upstream.failed", error=str(exc), round=round_index)
                # If we already spoke part of an answer, cut it off cleanly rather
                # than stitching an apology onto a half-finished sentence.
                if not text_buffer:
                    yield TextDelta(UPSTREAM_ERROR_REPLY)
                yield Done(finish_reason="stop", usage=usage)
                return

            if not tool_acc:
                yield Done(finish_reason="stop", usage=usage)
                return

            calls = _as_openai_tool_calls(tool_acc)
            ours = [c for c in calls if c["function"]["name"] in self._registry]
            theirs = [c for c in calls if c["function"]["name"] in client_tool_names]

            # Anything the caller declared is theirs to run: hand it back and stop,
            # exactly as a real OpenAI endpoint would.
            if theirs:
                yield ToolCallsDelta(theirs)
                yield Done(finish_reason="tool_calls", usage=usage)
                return

            if not ours:
                log.warning("tool.unknown", names=[c["function"]["name"] for c in calls])
                yield Done(finish_reason="stop", usage=usage)
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text_buffer) or None,
                    "tool_calls": ours,
                }
            )
            for call in ours:
                result = await self._registry.call(
                    call["function"]["name"], call["function"]["arguments"]
                )
                log.info("tool.called", tool=call["function"]["name"], round=round_index)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        log.warning("tool.rounds_exhausted", limit=self._settings.max_tool_rounds)
        yield TextDelta("Sorry, I couldn't work that one out.")
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
