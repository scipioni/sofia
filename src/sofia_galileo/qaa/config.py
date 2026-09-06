"""Settings for qaa-agent.service. Every field maps to a SOFIA_QAA_* env var."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sofia_galileo.qaa.prompts import DEFAULT_SYSTEM_PROMPT


class QaaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOFIA_QAA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    json_logs: bool = True
    # Dev only: see audio/config.py's SttSettings.reload — same mechanism, same
    # requirements (an editable install, docker/qaa.Dockerfile's DEV build arg).
    reload: bool = False

    # Upstream OpenAI-compatible LLM (vLLM, llama.cpp, Ollama, TGI, hosted API…).
    llm_base_url: str = "http://host.docker.internal:8000/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 1

    # Model id we advertise to s2s. Kept distinct from `llm_model` so the brain
    # is addressable by name regardless of what runs underneath.
    served_model_name: str = "sofia-qaa"

    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 320  # spoken replies should be short; this is a hard ceiling

    # Tool-calling loop. Each round is one extra upstream request, so keep it low
    # for a voice agent — a human is waiting in silence.
    tools_enabled: bool = True
    max_tool_rounds: int = 3

    # Point at a file to override the system prompt without rebuilding the image.
    system_prompt_file: Path | None = None

    @property
    def system_prompt(self) -> str:
        if self.system_prompt_file is not None:
            return self.system_prompt_file.read_text(encoding="utf-8").strip()
        return DEFAULT_SYSTEM_PROMPT
