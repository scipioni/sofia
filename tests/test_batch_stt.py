"""Batch STT: locale mapping, audio decoding, engine selection.

The model itself is exercised in the container; what is worth unit-testing is
the glue that silently degrades quality — sending Nemotron a bare "it" (which it
does not understand) instead of "it-IT", or handing it the wrong sample rate.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from sofia_galileo.audio.batch import (
    MODEL_SAMPLE_RATE,
    build_transcriber,
    decode_audio,
    match_model_dtype,
    to_locale,
)
from sofia_galileo.audio.config import SttSettings


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("it", "it-IT"),
        ("en", "en-US"),
        ("de", "de-DE"),
        ("IT", "it-IT"),  # case-insensitive
        (" it ", "it-IT"),  # whitespace from env files
        ("pt-BR", "pt-BR"),  # an explicit locale passes through
        ("en_GB", "en-GB"),  # underscore form normalised
        ("auto", "auto"),
        ("", "auto"),
        (None, "auto"),
        # Unknown languages must fall back to detection, never to an invented
        # locale — a wrong locale silently wrecks accuracy.
        ("xx", "auto"),
        ("klingon", "auto"),
    ],
)
def test_locale_mapping(given: str | None, expected: str) -> None:
    assert to_locale(given) == expected


def test_italian_is_mapped_because_it_is_the_reason_this_exists() -> None:
    assert to_locale("it") == "it-IT"


def wav_bytes(seconds: float, sample_rate: int, channels: int = 1) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_resamples_to_model_rate() -> None:
    """LiveKit sends 24 kHz; the model wants 16 kHz."""
    audio = decode_audio(wav_bytes(1.0, 24000))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - MODEL_SAMPLE_RATE) < 100


def test_decode_passes_through_at_model_rate() -> None:
    audio = decode_audio(wav_bytes(0.5, MODEL_SAMPLE_RATE))
    assert abs(len(audio) - MODEL_SAMPLE_RATE // 2) < 10


def test_decode_downmixes_stereo() -> None:
    audio = decode_audio(wav_bytes(0.5, MODEL_SAMPLE_RATE, channels=2))
    assert audio.ndim == 1
    assert abs(len(audio) - MODEL_SAMPLE_RATE // 2) < 10


def test_decode_rejects_garbage_with_a_useful_error() -> None:
    with pytest.raises(ValueError, match="could not decode audio"):
        decode_audio(b"this is not audio at all")


def test_unknown_engine_is_rejected_at_startup() -> None:
    with pytest.raises(ValueError, match="nemotron or whisper"):
        build_transcriber(SttSettings(batch_engine="parakeet"))


class _FakeTensor:
    def __init__(self, floating: bool) -> None:
        self.floating = floating
        self.casted_to: object | None = None

    def is_floating_point(self) -> bool:
        return self.floating

    def to(self, dtype: object) -> _FakeTensor:
        self.casted_to = dtype
        return self


def test_feature_dtype_matches_fp16_weights_but_ids_stay_integer() -> None:
    """GPU profiles load fp16 weights; float32 features would crash the conv."""
    inputs = {
        "input_features": _FakeTensor(floating=True),
        "prompt_ids": _FakeTensor(floating=False),
        "language": "it-IT",  # non-tensor entries pass through
    }
    result = match_model_dtype(inputs, "float16")
    assert result["input_features"].casted_to == "float16"
    assert result["prompt_ids"].casted_to is None
    assert result["language"] == "it-IT"


def test_default_engine_is_nemotron() -> None:
    settings = SttSettings()
    assert settings.batch_engine == "nemotron"
    assert "nemotron" in settings.model_id
