"""Table-driven tests for the wire <-> framework adapters in qaa.engine.

These adapters are the protocol seam of the refactor: every OpenAI message
shape the service accepts must land in the framework's message types with its
information intact, and deferred tool calls must come back as OpenAI dicts.
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai import ModelRequest, ModelResponse, SystemPromptPart, TextPart, ToolCallPart
from pydantic_ai.messages import ImageUrl, ToolReturnPart, UserPromptPart

from sofia_galileo.qaa.engine import (
    build_message_history,
    deferred_calls_to_openai,
    external_toolset_from_wire,
)


def req(messages: list[dict], **extra: object):
    from sofia_galileo.qaa.schemas import ChatCompletionRequest

    return ChatCompletionRequest(model="sofia-qaa", messages=messages, **extra)  # type: ignore[arg-type]


def test_system_message_becomes_system_prompt_part() -> None:
    [msg] = build_message_history(req([{"role": "system", "content": "Room: lobby."}]))
    assert isinstance(msg, ModelRequest)
    assert isinstance(msg.parts[0], SystemPromptPart)
    assert msg.parts[0].content == "Room: lobby."


def test_user_text_message_round_trips() -> None:
    [msg] = build_message_history(req([{"role": "user", "content": "hello"}]))
    assert isinstance(msg, ModelRequest)
    assert isinstance(msg.parts[0], UserPromptPart)
    assert msg.parts[0].content == "hello"


def test_user_multimodal_parts_are_preserved() -> None:
    content = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]
    [msg] = build_message_history(req([{"role": "user", "content": content}]))
    parts = msg.parts[0].content  # type: ignore[union-attr]
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "what is this?"
    assert isinstance(parts[1], ImageUrl)
    assert parts[1].url == "https://example.com/cat.png"


def test_user_unknown_part_type_degrades_to_json_not_silence() -> None:
    content = [{"type": "weird_future_part", "payload": 42}]
    [msg] = build_message_history(req([{"role": "user", "content": content}]))
    parts = msg.parts[0].content  # type: ignore[union-attr]
    assert isinstance(parts[0], TextPart)
    assert json.loads(parts[0].content)["payload"] == 42


def test_assistant_text_becomes_model_response() -> None:
    [msg] = build_message_history(req([{"role": "assistant", "content": "ciao"}]))
    assert isinstance(msg, ModelResponse)
    assert isinstance(msg.parts[0], TextPart)
    assert msg.parts[0].content == "ciao"


def test_assistant_tool_call_args_stay_structured() -> None:
    [msg] = build_message_history(
        req(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key": "value"}'},
                        }
                    ],
                }
            ]
        )
    )
    part = msg.parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.tool_name == "lookup"
    assert part.args == {"key": "value"}
    assert part.tool_call_id == "c1"


def test_assistant_tool_call_malformed_args_stay_verbatim() -> None:
    [msg] = build_message_history(
        req(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{oops"},
                        }
                    ],
                }
            ]
        )
    )
    assert msg.parts[0].args == "{oops"  # type: ignore[union-attr]


def test_tool_result_recovers_tool_name_from_preceding_call() -> None:
    history = build_message_history(
        req(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Monday"},
            ]
        )
    )
    [_, returned] = history
    part = returned.parts[0]  # type: ignore[union-attr]
    assert isinstance(part, ToolReturnPart)
    assert part.tool_name == "get_current_time"
    assert part.content == "Monday"
    assert part.tool_call_id == "c1"


def test_tool_result_without_a_matching_call_gets_a_placeholder_name() -> None:
    [msg] = build_message_history(req([{"role": "tool", "tool_call_id": "orphan", "content": "x"}]))
    assert msg.parts[0].tool_name == "unknown_tool"  # type: ignore[union-attr]


def test_developer_role_is_treated_as_system() -> None:
    [msg] = build_message_history(req([{"role": "developer", "content": "be brief"}]))
    assert isinstance(msg.parts[0], SystemPromptPart)  # type: ignore[union-attr]


def test_sofia_session_context_is_prepended_as_system() -> None:
    history = build_message_history(
        req(
            [{"role": "user", "content": "hi"}], sofia_language="Italian", sofia_participant="mario"
        )
    )
    [context, user_msg] = history
    assert isinstance(context.parts[0], SystemPromptPart)
    assert "Italian" in context.parts[0].content
    assert "mario" in context.parts[0].content
    assert isinstance(user_msg.parts[0], UserPromptPart)


def test_message_order_is_preserved() -> None:
    history = build_message_history(
        req(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ]
        )
    )
    assert isinstance(history[0].parts[0], SystemPromptPart)
    assert isinstance(history[1].parts[0], UserPromptPart)
    assert isinstance(history[2], ModelResponse)
    assert history[2].parts[0].content == "a1"
    assert history[3].parts[0].content == "u2"  # type: ignore[union-attr]


# -- external toolset from wire --------------------------------------------


def test_wire_tools_become_schema_only_definitions() -> None:
    toolset = external_toolset_from_wire(
        [
            {
                "type": "function",
                "function": {
                    "name": "transfer_call",
                    "description": "transfer",
                    "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
                },
            }
        ]
    )
    assert toolset is not None
    [definition] = toolset.tool_defs
    assert definition.name == "transfer_call"
    assert definition.description == "transfer"
    assert definition.parameters_json_schema == {
        "type": "object",
        "properties": {"to": {"type": "string"}},
    }
    # strict-mode inference must not rewrite the caller's schema upstream
    assert definition.strict is False


def test_no_tools_yields_no_toolset() -> None:
    assert external_toolset_from_wire(None) is None
    assert external_toolset_from_wire([]) is None


def test_malformed_wire_entries_are_skipped() -> None:
    assert (
        external_toolset_from_wire([{"type": "web_search"}, {"type": "function"}, "junk"]) is None
    )


# -- deferred calls -> OpenAI handback -------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"to": "sales"}, '{"to": "sales"}'),
        ('{"raw": "string"}', '{"raw": "string"}'),
        (None, "{}"),
    ],
)
def test_deferred_calls_render_as_openai_dicts(args: object, expected: str) -> None:
    [call] = deferred_calls_to_openai(
        [ToolCallPart(tool_name="transfer_call", args=args, tool_call_id="c9")]
    )
    assert call["id"] == "c9"
    assert call["type"] == "function"
    assert call["function"]["name"] == "transfer_call"
    assert json.loads(call["function"]["arguments"]) == json.loads(expected)
