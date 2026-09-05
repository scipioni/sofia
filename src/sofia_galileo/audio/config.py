"""Settings for the stt and tts services."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_device(requested: str) -> str:
    """Map 'auto' to the best available device.

    `torch.cuda` is also the entrypoint for AMD GPUs under the ROCm/HIP build of
    PyTorch, so this single check covers both target platforms.
    """
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class SttSettings(BaseSettings):
    """The stt service can serve two backends at once, on different endpoints.

    ``batch``      Whisper on ``POST /v1/audio/transcriptions`` — accurate,
                   punctuated, multilingual, but it cannot start until the
                   person stops talking.
    ``streaming``  sherpa-onnx on ``WS /v1/realtime`` — transcribes *while* they
                   speak, so end-of-turn costs almost nothing. Lower accuracy,
                   no punctuation, and only the languages with a streaming model.

    Which one s2s actually uses is its choice (``SOFIA_S2S_STT_USE_REALTIME``);
    serving both means flipping that is a restart of one container, not a
    redeploy.
    """

    model_config = SettingsConfigDict(env_prefix="SOFIA_STT_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "INFO"
    json_logs: bool = True

    # batch | streaming | both
    backend: str = "both"
    default_language: str = "en"

    # --- batch backend ---
    # nemotron: NVIDIA Nemotron 3.5 ASR (638M). Punctuated, cased, ~0.08 RTF on a
    #           CPU core, and strong on European languages including Italian.
    # whisper : transformers Whisper. Slower, but ~100 languages against
    #           Nemotron's 40 locales — the fallback for anything unsupported.
    # Changing this means changing model_id to match.
    batch_engine: str = "nemotron"
    model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "float16"  # ignored on cpu, where float32 is used
    # Cap on one utterance's decoded tokens. Spoken turns are short; this exists
    # to stop a pathological input decoding forever.
    max_new_tokens: int = 448
    # Whisper only: it degrades on long single chunks, and 30s is its training window.
    chunk_length_s: int = 30
    batch_size: int = 8

    # --- streaming backend (sherpa-onnx) ---
    # Fetched on first boot into the models volume if the directory is empty.
    # Swap languages by pointing this at another release asset; see the README
    # for what streaming models exist (there is no Italian one).
    sherpa_model_url: str = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
    )
    sherpa_model_dir: Path = Path("/home/sofia/models/stt-streaming")
    # cpu is not a compromise here: a streaming zipformer runs at ~0.06 RTF on
    # one core, so a CPU core serves ~16 concurrent conversations. It is also
    # the only option on ROCm, since the sherpa-onnx wheels ship CUDA only.
    sherpa_provider: str = "cpu"  # cpu | cuda
    sherpa_num_threads: int = 2
    # int8 encoder roughly halves compute for a small accuracy cost. The decoder
    # and joiner are tiny, so they use int8 unconditionally when present.
    sherpa_encoder_int8: bool = False

    # Endpointing rules, in seconds. rule2 is the one that matters for
    # conversation: how much trailing silence ends a turn that has speech in it.
    # sherpa's 1.2 default feels sluggish out loud; 0.8 is closer to human.
    sherpa_rule1_min_trailing_silence: float = 2.4  # silence with nothing decoded
    sherpa_rule2_min_trailing_silence: float = 0.8  # silence after speech
    sherpa_rule3_min_utterance_length: float = 20.0  # hard cap on one utterance

    # One phrase per line, biases decoding towards them. Worth pointing at a file
    # with your product and person names — "Sofia" is otherwise heard as "Sophia".
    sherpa_hotwords_file: Path | None = None
    sherpa_hotwords_score: float = 1.5

    # Zipformer emits uppercase, unpunctuated text. Handing SHOUTING to an LLM
    # and to the semantic turn detector is out-of-distribution for both.
    sherpa_normalize_case: bool = True


class TtsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOFIA_TTS_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8200
    log_level: str = "INFO"
    json_logs: bool = True

    device: str = "auto"  # auto | cuda | cpu
    default_voice: str = "af_heart"
    sample_rate: int = 24000  # Kokoro's native rate; do not change
