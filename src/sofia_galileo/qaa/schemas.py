"""Wire types for the OpenAI-compatible surface of qaa-agent.

We deliberately speak the OpenAI Chat Completions protocol rather than a bespoke
one: s2s.service can then drive us with `livekit.plugins.openai.LLM`, streaming
and tool-calling come for free, and the brain can be exercised with plain curl
or any OpenAI client with zero LiveKit involved.

Models are permissive (`extra="allow"`) on purpose — unknown fields from newer
clients are preserved rather than rejected.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool"]


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    index: int | None = None
    type: Literal["function"] = "function"
    function: FunctionCall = Field(default_factory=FunctionCall)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    # `content` is a list when the caller sends multimodal parts; we pass those
    # through untouched so a vision-capable upstream still works.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None

    # --- Sofia extensions (sent by s2s via `extra_body`, ignored by upstream) ---
    sofia_room: str | None = None
    sofia_participant: str | None = None
    sofia_language: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_new_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ChoiceDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sofia-galileo"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
