"""Entrypoints for the stt and tts services (one image, two commands)."""

from __future__ import annotations

import uvicorn

from sofia_galileo.audio.config import SttSettings, TtsSettings


def main_stt() -> None:
    settings = SttSettings()
    if settings.reload:
        # The factory-string form is what lets uvicorn's own file-watcher
        # re-import create_app() on every change; a pre-built app instance
        # (the else branch) can't be reloaded. Needs an editable install
        # (docker/audio.Dockerfile's DEV build arg) so the reloaded import
        # actually sees a bind-mounted /app/src, not the build-time copy.
        uvicorn.run(
            "sofia_galileo.audio.stt_app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            reload_dirs=["/app/src"],
            log_config=None,
            access_log=False,
        )
        return

    from sofia_galileo.audio.stt_app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


def main_tts() -> None:
    settings = TtsSettings()
    if settings.reload:
        uvicorn.run(
            "sofia_galileo.audio.tts_app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            reload_dirs=["/app/src"],
            log_config=None,
            access_log=False,
        )
        return

    from sofia_galileo.audio.tts_app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main_stt()
