from __future__ import annotations

import json

import pytest

from sofia_galileo.qaa.tools import ToolRegistry, build_default_registry


async def test_get_current_time_returns_spoken_shape() -> None:
    registry = build_default_registry()
    out = await registry.call("get_current_time", json.dumps({"timezone": "Europe/Rome"}))
    assert "Europe/Rome" in out
    assert "Error" not in out


async def test_unknown_timezone_is_reported_not_raised() -> None:
    registry = build_default_registry()
    out = await registry.call("get_current_time", json.dumps({"timezone": "Mars/Olympus"}))
    assert out.startswith("Error")


async def test_unknown_tool_is_reported_not_raised() -> None:
    registry = build_default_registry()
    assert (await registry.call("teleport", "{}")).startswith("Error")


async def test_malformed_arguments_are_reported_not_raised() -> None:
    registry = build_default_registry()
    assert (await registry.call("get_current_time", "{not json")).startswith("Error")


async def test_handler_exception_becomes_text_for_the_model() -> None:
    registry = ToolRegistry()

    @registry.register("boom", "always fails", {"type": "object", "properties": {}})
    async def boom() -> str:
        raise RuntimeError("kaboom")

    out = await registry.call("boom", "{}")
    assert "kaboom" in out


async def test_slow_handler_times_out() -> None:
    import asyncio

    registry = ToolRegistry(timeout_s=0.05)

    @registry.register("slow", "never returns", {"type": "object", "properties": {}})
    async def slow() -> str:
        await asyncio.sleep(5)
        return "never"

    out = await registry.call("slow", "{}")
    assert "timed out" in out


def test_specs_are_valid_openai_tool_entries() -> None:
    for spec in build_default_registry().specs():
        assert spec["type"] == "function"
        assert spec["function"]["name"]
        assert spec["function"]["parameters"]["type"] == "object"


@pytest.mark.parametrize("name", ["get_current_time"])
def test_membership(name: str) -> None:
    assert name in build_default_registry()
