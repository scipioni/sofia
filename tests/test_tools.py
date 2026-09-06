"""Tool-surface tests after the registry -> framework-toolset refactor.

The error-as-text contract is enforced in code (`never_raises`) and verified
here at the level the model actually sees: the tool message content that goes
back upstream on the next request.
"""

from __future__ import annotations

import asyncio

import httpx
from openai import AsyncOpenAI
from pydantic_ai import FunctionToolset
from test_engine import StubUpstream, text_chunk, tool_chunk

from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.engine import QaaEngine, upstream_model
from sofia_galileo.qaa.schemas import ChatCompletionRequest
from sofia_galileo.qaa.tools import (
    TOOL_TIMEOUT_S,
    build_default_toolset,
    get_current_time,
    never_raises,
)


async def test_get_current_time_returns_spoken_shape() -> None:
    out = await get_current_time(timezone="Europe/Rome")
    assert "Europe/Rome" in out
    assert "Error" not in out


async def test_unknown_timezone_is_reported_not_raised() -> None:
    out = await get_current_time(timezone="Mars/Olympus")
    assert out.startswith("Error")


def test_default_toolset_holds_the_default_tools() -> None:
    toolset = build_default_toolset()
    assert "get_current_time" in toolset.tools
    assert toolset.timeout == TOOL_TIMEOUT_S


async def test_handler_exception_becomes_text_for_the_model() -> None:
    async def boom() -> str:
        raise RuntimeError("kaboom")

    out = await never_raises(boom)()
    assert out.startswith("Error")
    assert "kaboom" in out


async def test_timeout_reaches_the_model_as_a_retry_prompt() -> None:
    """A hanging tool must not hang the run: the framework bounds it and the
    model gets a retry prompt it can act on instead of endless silence."""

    async def slow() -> str:
        await asyncio.sleep(5)
        return "never"

    stub = StubUpstream(
        [[tool_chunk(0, call_id="c", name="slow", arguments="{}")], [text_chunk("giving up.")]]
    )
    settings = QaaSettings(llm_base_url="http://stub/v1", llm_model="stub", llm_max_retries=0)
    toolset = FunctionToolset([never_raises(slow)], timeout=0.05)
    client = AsyncOpenAI(
        base_url="http://stub/v1",
        api_key="x",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=stub.app)),
    )
    engine = QaaEngine(settings, toolset, model=upstream_model(settings, client))

    text, _, _, _ = await engine.complete(
        ChatCompletionRequest(model="sofia-qaa", messages=[{"role": "user", "content": "go"}])
    )

    assert text == "giving up."
    retry_message = stub.requests[1]["messages"][-1]
    assert retry_message["role"] == "tool"
    assert "timed out" in retry_message["content"].lower()


def test_default_tool_schema_is_a_valid_openai_object() -> None:
    toolset = build_default_toolset()
    tool = toolset.tools["get_current_time"]
    schema = tool.tool_def.parameters_json_schema
    assert schema["type"] == "object"
    assert "timezone" in schema["properties"]
