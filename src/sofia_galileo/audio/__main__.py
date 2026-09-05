"""Entrypoints for the stt and tts services (one image, two commands)."""

from __future__ import annotations

import uvicorn

from sofia_galileo.audio.config import SttSettings, TtsSettings


def main_stt() -> None:
    from sofia_galileo.audio.stt_app import create_app

    settings = SttSettings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


def main_tts() -> None:
    from sofia_galileo.audio.tts_app import create_app

    settings = TtsSettings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main_stt()
