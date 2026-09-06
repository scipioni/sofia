"""Entrypoint for qaa-agent.service."""

from __future__ import annotations

import uvicorn

from sofia_galileo.qaa.config import QaaSettings


def main() -> None:
    settings = QaaSettings()
    if settings.reload:
        # See audio/__main__.py's main_stt for why this needs the factory-string
        # form plus an editable install (docker/qaa.Dockerfile's DEV build arg).
        uvicorn.run(
            "sofia_galileo.qaa.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            reload_dirs=["/app/src"],
            log_config=None,
            access_log=False,
        )
        return

    from sofia_galileo.qaa.app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # structlog owns logging; don't let uvicorn reset it
        access_log=False,
    )


if __name__ == "__main__":
    main()
