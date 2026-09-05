"""Settings for s2s.service. Every field maps to a SOFIA_S2S_* env var.

LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are deliberately absent:
the livekit-agents CLI reads those from the environment itself.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class S2SSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOFIA_S2S_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    json_logs: bool = True

    # Only set this if you dispatch the agent explicitly by name; leaving it
    # empty makes the worker join every room automatically.
    agent_name: str = ""

    # --- the brain (qaa-agent.service, OpenAI-compatible) ---
    qaa_base_url: str = "http://qaa-agent:8000/v1"
    qaa_api_key: str = "sofia"
    qaa_model: str = "sofia-qaa"

    # --- speech to text ---
    stt_base_url: str = "http://stt:8100/v1"
    stt_api_key: str = "sofia"
    stt_model: str = "whisper-1"
    language: str = "en"
    # True  -> websocket streaming ASR (ws://.../v1/realtime): the transcript is
    #          decoded while the person talks, so end-of-turn costs ~nothing.
    # False -> batch Whisper: more accurate and punctuated, but the whole
    #          utterance is transcribed only after they stop.
    # The stt service must be serving the matching backend (SOFIA_STT_BACKEND).
    stt_use_realtime: bool = False

    # --- text to speech ---
    tts_base_url: str = "http://tts:8200/v1"
    tts_api_key: str = "sofia"
    tts_model: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    # wav or pcm. The livekit plugin would otherwise default to mp3, which the
    # Kokoro service does not encode.
    tts_response_format: str = "wav"

    # Spoken once, when the agent joins. Empty means wait for the human to speak
    # first — which is often the better choice for inbound calls.
    greeting: str = "Greet the person warmly in one short sentence and offer to help."

    # Room-specific context handed to the brain alongside its own system prompt.
    instructions: str = "You have just joined a live voice call."

    # Silence, in seconds, before the agent assumes the person is done talking.
    # The semantic turn detector adapts around this; it is a floor, not a fixed wait.
    min_endpointing_delay: float = 0.4
    max_endpointing_delay: float = 5.0
