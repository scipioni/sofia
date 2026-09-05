"""Structured logging shared by every service.

Voice agents live or die on latency, so every log line carries the service name
and, where relevant, the room/session it belongs to. JSON output by default so
`docker compose logs` can be shipped somewhere useful without reparsing.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    service: str,
    level: str = "INFO",
    *,
    json_logs: bool = True,
    attach_handler: bool = True,
) -> None:
    """Configure stdlib logging and structlog to share one renderer.

    Pass ``attach_handler=False`` when something else already owns the root
    logger — livekit-agents' CLI installs its own handler, and adding a second
    one prints every line twice. In that mode structlog hands its events to the
    existing stdlib handler instead of rendering them itself.
    """
    if not attach_handler:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.render_to_log_kwargs,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        structlog.contextvars.bind_contextvars(service=service)
        return

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty and say nothing we don't already log ourselves.
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
