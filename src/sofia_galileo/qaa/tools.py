"""Tool definitions for the brain.

The agent framework (pydantic-ai) owns dispatch, argument validation, retries
and timeouts; what lives here is the tool surface itself — typed async
functions whose signatures and docstrings become the JSON schema the model
sees. The tool surface of a voice agent stays small by necessity, because
every tool round is a round trip the human spends in silence.

The contract that survived the registry → toolset refactor unchanged: a tool
returns a string that is fed straight back to the model, and expected errors
are returned as text rather than raised, so the conversation can recover
instead of dropping. `never_raises` enforces the second half; the framework's
per-toolset timeout covers the first.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic_ai import FunctionToolset

from sofia_galileo.core.logging import get_logger

log = get_logger(__name__)

Handler = Callable[..., Awaitable[str]]

# A hanging tool must not hang the run: the framework turns a timeout into a
# retry prompt for the model (one retry by default), same budget as before.
TOOL_TIMEOUT_S = 10.0


def never_raises(fn: Handler) -> Handler:
    """Report an unexpected tool failure to the model as text instead of raising.

    A raised exception would burn the tool's retry budget and, once exhausted,
    fail the whole run — an apology instead of an answer. An error string lets
    the model decide how to proceed, exactly as the old registry did.
    """

    @functools.wraps(fn)
    async def wrapper(*args: object, **kwargs: object) -> str:
        try:
            return await fn(*args, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            log.exception("tool.failed", tool=fn.__name__)
            return f"Error while running {fn.__name__}: {exc}"

    return wrapper


async def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Use this whenever the person asks what time or what day it is, or when a
    reply depends on the current moment.

    Args:
        timezone: IANA timezone name, e.g. 'Europe/Rome'. Defaults to UTC when
            the person's timezone is unknown.
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: {timezone!r} is not a known timezone."
    now = datetime.now(UTC).astimezone(tz)
    return now.strftime("%A %d %B %Y, %H:%M") + f" ({timezone})"


def build_default_toolset() -> FunctionToolset:
    """The tools every Sofia deployment gets. Extend or replace as needed."""
    return FunctionToolset([never_raises(get_current_time)], timeout=TOOL_TIMEOUT_S)
