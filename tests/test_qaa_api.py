"""Contract tests for the OpenAI-compatible surface.

The engine is stubbed out here on purpose: what matters is that s2s can drive
this service with a stock OpenAI client, which is a question about wire format,
not about what the model says.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from sofia_galileo.qaa.app import create_app
from sofia_galileo.qaa.config import QaaSettings
from sofia_galileo.qaa.engine import Done, StreamEvent, TextDelta, ToolCallsDelta
from sofia_galileo.qaa.schemas import ChatCompletionRequest, Usage


class StubEngine:
    """Replays a fixed script of engine events."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event

    async def complete(self, req: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        text, calls, reason, usage = "", [], "stop", Usage()
        async for event in self.stream(req):
            match event:
                case TextDelta(text=delta):
                    text += delta
                case ToolCallsDelta(tool_calls=tc):
                    calls = tc
                case Done(finish_reason=r, usage=u):
                    reason, usage = r, u
        return text, calls, reason, usage

    async def aclose(self) -> None:
        return None


def client_with(events: list[StreamEvent]) -> TestClient:
    settings = QaaSettings(llm_base_url="http://unused/v1", llm_model="stub")
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()  # run lifespan so app.state.engine exists
    app.state.engine = StubEngine(events)  # then replace it
    return client


def sse_payloads(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line.removeprefix("data: ")))
    return out


@pytest.fixture
def hello_client() -> TestClient:
    events = [TextDelta("Hi "), TextDelta("there."), Done(finish_reason="stop")]
    client = client_with(events)
    yield client
    client.__exit__(None, None, None)


def test_healthz(hello_client: TestClient) -> None:
    assert hello_client.get("/healthz").json() == {"status": "ok"}


def test_models_advertises_the_served_name(hello_client: TestClient) -> None:
    body = hello_client.get("/v1/models").json()
    assert body["data"][0]["id"] == "sofia-qaa"


def test_non_streaming_completion_shape(hello_client: TestClient) -> None:
    resp = hello_client.post(
        "/v1/chat/completions",
        json={"model": "sofia-qaa", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "sofia-qaa"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hi there."
    assert choice["finish_reason"] == "stop"


def test_streaming_emits_openai_chunks_then_done(hello_client: TestClient) -> None:
    resp = hello_client.post(
        "/v1/chat/completions",
        json={
            "model": "sofia-qaa",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.endswith("data: [DONE]\n\n")

    chunks = sse_payloads(resp.text)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    spoken = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"][0]["delta"].get("content")
    )
    assert spoken == "Hi there."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_empty_messages_is_a_400() -> None:
    client = client_with([])
    try:
        resp = client.post("/v1/chat/completions", json={"model": "sofia-qaa", "messages": []})
        assert resp.status_code == 400
    finally:
        client.__exit__(None, None, None)


def test_client_side_tool_calls_are_handed_back() -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup_booking", "arguments": '{"id":"7"}'},
        }
    ]
    client = client_with([ToolCallsDelta(calls), Done(finish_reason="tool_calls")])
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "sofia-qaa", "messages": [{"role": "user", "content": "hi"}]},
        )
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup_booking"
    finally:
        client.__exit__(None, None, None)
