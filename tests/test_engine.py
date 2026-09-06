"""Engine tests against a stub upstream LLM.

The tool loop itself is pydantic-ai's now; what these tests protect is the
*contract*: fragments of a tool call arrive spread across SSE chunks, some
tools are ours to run and some belong to the caller, and getting either wrong
shows up as a voice agent that goes silent mid-sentence. So we drive the
engine with a real HTTP-shaped upstream and assert on the exact wire, so a
framework upgrade cannot shift behavior silently.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.engine import (
    Done,
    QaaEngine,
    TextDelta,
    ToolCallsDelta,
    upstream_model,
)
from sofia_galileo.qaa.schemas import ChatCompletionRequest


def text_chunk(content: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stub",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def tool_chunk(index: int, *, call_id=None, name=None, arguments=None) -> dict:
    fn: dict = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    call: dict = {"index": index, "type": "function", "function": fn}
    if call_id is not None:
        call["id"] = call_id
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stub",
        "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
    }


class StubUpstream:
    """Serves a scripted list of streamed responses, one per request."""

    def __init__(self, script: list[list[dict]]) -> None:
        self.script = script
        self.requests: list[dict] = []
        self.app = FastAPI()

        @self.app.post("/v1/chat/completions")
        async def completions(request: Request):  # type: ignore[no-untyped-def]
            body = await request.json()
            self.requests.append(body)
            chunks = self.script[len(self.requests) - 1]

            async def stream() -> AsyncIterator[bytes]:
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

    def engine(self, **overrides) -> QaaEngine:  # type: ignore[no-untyped-def]
        settings = QaaSettings(
            llm_base_url="http://stub/v1", llm_model="stub", llm_max_retries=0, **overrides
        )
        # The model is injected over an ASGI transport so every assertion sees
        # the exact wire bytes the framework sends upstream.
        client = AsyncOpenAI(
            base_url="http://stub/v1",
            api_key="x",
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app)),
        )
        return QaaEngine(settings, model=upstream_model(settings, client))


def user(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="sofia-qaa", messages=[{"role": "user", "content": content}])


async def collect(engine: QaaEngine, req: ChatCompletionRequest) -> list:
    return [event async for event in engine.stream(req)]


async def test_plain_text_streams_through_verbatim() -> None:
    stub = StubUpstream([[text_chunk("Hi "), text_chunk("there.")]])
    events = await collect(stub.engine(), user("hello"))

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi ", "there."]
    assert isinstance(events[-1], Done)
    assert events[-1].finish_reason == "stop"


async def test_sofia_system_prompt_is_prepended() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    await collect(stub.engine(), user("hello"))

    messages = stub.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "voice assistant" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hello"}


async def test_callers_system_message_is_kept_not_replaced() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    req = ChatCompletionRequest(
        model="sofia-qaa",
        messages=[
            {"role": "system", "content": "Room: lobby."},
            {"role": "user", "content": "hi"},
        ],
    )
    await collect(stub.engine(), req)

    contents = [m["content"] for m in stub.requests[0]["messages"] if m["role"] == "system"]
    assert any("voice assistant" in c for c in contents)
    assert any("Room: lobby." in c for c in contents)


async def test_server_side_tool_runs_inline_and_answer_continues() -> None:
    """A tool call split across chunks: execute ours, then stream the real answer."""
    stub = StubUpstream(
        [
            [
                tool_chunk(0, call_id="call_1", name="get_current_time", arguments='{"time'),
                tool_chunk(0, arguments='zone":"Europe/Rome"}'),
            ],
            [text_chunk("It's just gone three.")],
        ]
    )
    events = await collect(stub.engine(), user("what time is it"))

    # The caller never sees the internal tool call — only the final answer.
    assert not any(isinstance(e, ToolCallsDelta) for e in events)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["It's just gone three."]

    # Second upstream request carries the assistant tool_call and its result.
    second = stub.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert second[-1]["role"] == "tool"
    assert "Europe/Rome" in second[-1]["content"]


async def test_client_side_tool_is_handed_back_unexecuted() -> None:
    stub = StubUpstream(
        [[tool_chunk(0, call_id="c1", name="transfer_call", arguments='{"to":"sales"}')]]
    )
    req = user("put me through to sales")
    req.tools = [
        {
            "type": "function",
            "function": {
                "name": "transfer_call",
                "description": "transfer",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    events = await collect(stub.engine(), req)

    handed_back = [e for e in events if isinstance(e, ToolCallsDelta)]
    assert len(handed_back) == 1
    assert handed_back[0].tool_calls[0]["function"]["name"] == "transfer_call"
    assert events[-1].finish_reason == "tool_calls"
    # Only one upstream call: we stopped and gave the turn back to s2s.
    assert len(stub.requests) == 1


async def test_caller_tools_are_forwarded_alongside_ours() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    req = user("hi")
    req.tools = [
        {
            "type": "function",
            "function": {"name": "transfer_call", "description": "t", "parameters": {}},
        }
    ]
    await collect(stub.engine(), req)

    names = {t["function"]["name"] for t in stub.requests[0]["tools"]}
    assert names == {"transfer_call", "get_current_time"}


async def test_max_tokens_ceiling_is_not_raisable_by_the_caller() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    req = user("hi")
    req.max_tokens = 100_000
    await collect(stub.engine(max_tokens=320), req)

    assert stub.requests[0]["max_tokens"] == 320


async def test_upstream_failure_still_says_something() -> None:
    """Silence is the worst failure mode for a voice agent."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def boom():  # type: ignore[no-untyped-def]
        raise RuntimeError("upstream down")

    settings = QaaSettings(llm_base_url="http://stub/v1", llm_model="stub", llm_max_retries=0)
    client = AsyncOpenAI(
        base_url="http://stub/v1",
        api_key="x",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False)
        ),
    )
    engine = QaaEngine(settings, model=upstream_model(settings, client))

    events = await collect(engine, user("hello"))
    spoken = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert spoken.strip()
    assert isinstance(events[-1], Done)


async def test_tool_round_limit_is_enforced() -> None:
    """A model stuck calling the same tool must not loop forever."""
    forever = [tool_chunk(0, call_id="c", name="get_current_time", arguments="{}")]
    stub = StubUpstream([forever] * 10)
    engine = stub.engine(max_tool_rounds=2)

    events = await collect(engine, user("what time is it"))

    assert len(stub.requests) == 3  # initial + 2 tool rounds
    assert isinstance(events[-1], Done)


@pytest.mark.parametrize("stream", [True, False])
async def test_complete_matches_stream(stream: bool) -> None:
    stub = StubUpstream([[text_chunk("a"), text_chunk("b")]])
    text, calls, reason, _ = await stub.engine().complete(user("hi"))

    assert text == "ab"
    assert calls == []
    assert reason == "stop"


# -- sampling settings on the wire -----------------------------------------


async def test_stop_sequences_reach_upstream_as_stop() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    req = user("hi")
    req.stop = "ADIOS"
    await collect(stub.engine(), req)

    assert stub.requests[0]["stop"] == ["ADIOS"]


async def test_passthrough_tool_choice_reaches_upstream() -> None:
    stub = StubUpstream([[text_chunk("ok")]])
    req = user("hi")
    req.tool_choice = "none"
    await collect(stub.engine(), req)

    assert stub.requests[0]["tool_choice"] == "none"


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_current_time"}}],
)
async def test_forced_or_structured_tool_choice_is_dropped_not_sent(tool_choice) -> None:
    """The framework forbids forcing tool calls on a run that may end in a
    handback (deferred output) — it would raise before any request. Dropping
    the directive beats failing the whole turn; the framework then resolves
    its own default ('auto'). Pinned so this stays deliberate."""
    stub = StubUpstream([[text_chunk("ok")]])
    req = user("hi")
    req.tool_choice = tool_choice
    await collect(stub.engine(), req)

    assert stub.requests[0]["tool_choice"] == "auto"


# -- behavior pins for the refactor ----------------------------------------


async def test_text_then_tool_call_in_one_response_streams_text_and_runs_tool() -> None:
    """The text was already streamed (and is being spoken) by the time the
    tool calls arrive — they must still run. This is why the run is driven
    with an explicit end strategy rather than a first-output-wins stream."""
    stub = StubUpstream(
        [
            [
                text_chunk("Let me check. "),
                tool_chunk(0, call_id="c", name="get_current_time", arguments="{}"),
            ],
            [text_chunk("It's noon.")],
        ]
    )
    events = await collect(stub.engine(), user("what time is it"))

    spoken = [e.text for e in events if isinstance(e, TextDelta)]
    assert "Let me check. " in spoken
    assert "It's noon." in spoken
    assert len(stub.requests) == 2  # the tool actually ran


async def test_handback_arguments_are_valid_json_for_the_declared_schema() -> None:
    stub = StubUpstream(
        [[tool_chunk(0, call_id="c1", name="transfer_call", arguments='{"to" :  "sales"}')]]
    )
    req = user("put me through to sales")
    req.tools = [
        {
            "type": "function",
            "function": {
                "name": "transfer_call",
                "description": "transfer",
                "parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                },
            },
        }
    ]
    events = await collect(stub.engine(), req)

    handed = next(e for e in events if isinstance(e, ToolCallsDelta))
    # Args reach the caller exactly as the model emitted them — the framework
    # checks JSON-object-ness but passes the string through untouched.
    assert handed.tool_calls[0]["function"]["arguments"] == '{"to" :  "sales"}'
    assert events[-1].finish_reason == "tool_calls"


async def test_schema_violating_args_still_reach_the_caller_verbatim() -> None:
    """The caller's schema is enforced by s2s/LiveKit, not by us: args that
    violate it are handed back as emitted, exactly as the hand-rolled engine
    did. (The framework's external-tool validator checks JSON-object-ness
    only — pinned so a framework upgrade cannot start silently retrying.)"""
    stub = StubUpstream(
        [[tool_chunk(0, call_id="c1", name="transfer_call", arguments='{"wrong": 1}')]]
    )
    req = user("put me through to sales")
    req.tools = [
        {
            "type": "function",
            "function": {
                "name": "transfer_call",
                "description": "transfer",
                "parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                },
            },
        }
    ]
    events = await collect(stub.engine(), req)

    assert len(stub.requests) == 1  # no in-run retry; the turn is the caller's
    handed = next(e for e in events if isinstance(e, ToolCallsDelta))
    assert handed.tool_calls[0]["function"]["arguments"] == '{"wrong": 1}'
    assert events[-1].finish_reason == "tool_calls"


async def test_mixed_ours_and_theirs_calls_hand_theirs_back() -> None:
    """Precedence decision (design.md): a caller-declared tool always wins the
    turn — the run ends handing theirs back, whatever ours were already doing.
    Pinned here so a framework upgrade cannot shift it silently."""
    stub = StubUpstream(
        [
            [
                tool_chunk(0, call_id="c1", name="get_current_time", arguments="{}"),
                tool_chunk(1, call_id="c2", name="transfer_call", arguments='{"to": "sales"}'),
            ]
        ]
    )
    req = user("what time is it? and put me through to sales")
    req.tools = [
        {
            "type": "function",
            "function": {"name": "transfer_call", "description": "t", "parameters": {}},
        }
    ]
    events = await collect(stub.engine(), req)

    handed = [e for e in events if isinstance(e, ToolCallsDelta)]
    assert len(handed) == 1
    assert [c["function"]["name"] for c in handed[0].tool_calls] == ["transfer_call"]
    assert events[-1].finish_reason == "tool_calls"
    assert not any(
        "get_current_time" in c["function"]["name"] for e in handed for c in e.tool_calls
    )
