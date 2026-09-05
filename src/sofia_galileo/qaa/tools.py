"""Tool registry for the brain.

Deliberately dependency-free: a decorator, a dict, and a JSON Schema per tool.
An agent framework would add indirection and latency without buying anything
here — the tool surface of a voice agent is small by necessity, because every
tool round is a round trip the human spends in silence.

Add a tool by writing an async function and decorating it. Handlers must return
a string (it is fed straight back to the model as the tool message content) and
must never raise — errors are caught and reported to the model as text so the
conversation can recover instead of dropping.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sofia_galileo.core.logging import get_logger

log = get_logger(__name__)

Handler = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def spec(self) -> dict[str, Any]:
        """OpenAI `tools` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    timeout_s: float = 10.0
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(
        self, name: str, description: str, parameters: dict[str, Any]
    ) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self._tools[name] = Tool(name, description, parameters, fn)
            return fn

        return decorator

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    async def call(self, name: str, raw_arguments: str | None) -> str:
        """Run a tool by name. Always returns text, never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r} is available."

        try:
            kwargs = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            return f"Error: could not parse arguments for {name}: {exc}"
        if not isinstance(kwargs, dict):
            return f"Error: arguments for {name} must be a JSON object."

        try:
            async with asyncio.timeout(self.timeout_s):
                return await tool.handler(**kwargs)
        except TimeoutError:
            log.warning("tool.timeout", tool=name, timeout_s=self.timeout_s)
            return f"Error: tool {name} timed out after {self.timeout_s:g} seconds."
        except Exception as exc:
            log.exception("tool.failed", tool=name)
            return f"Error while running {name}: {exc}"


def build_default_registry() -> ToolRegistry:
    """The tools every Sofia deployment gets. Extend or replace as needed."""
    registry = ToolRegistry()

    @registry.register(
        name="get_current_time",
        description=(
            "Get the current date and time. Use this whenever the person asks what "
            "time or what day it is, or when a reply depends on the current moment."
        ),
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone name, e.g. 'Europe/Rome'. Defaults to UTC "
                        "when the person's timezone is unknown."
                    ),
                }
            },
            "required": [],
        },
    )
    async def get_current_time(timezone: str = "UTC") -> str:
        try:
            tz = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            return f"Error: {timezone!r} is not a known timezone."
        now = datetime.now(UTC).astimezone(tz)
        return now.strftime("%A %d %B %Y, %H:%M") + f" ({timezone})"

    return registry
