"""Entrypoint for qaa-agent.service."""

from __future__ import annotations

import uvicorn

from sofia_galileo.qaa.app import create_app
from sofia_galileo.qaa.config import QaaSettings


def main() -> None:
    settings = QaaSettings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # structlog owns logging; don't let uvicorn reset it
        access_log=False,
    )


if __name__ == "__main__":
    main()
