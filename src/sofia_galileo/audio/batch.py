"""Batch (whole-utterance) speech recognition.

Two interchangeable engines behind one interface:

``nemotron``  NVIDIA Nemotron 3.5 ASR — a 638M FastConformer-RNNT. Small, fast
              (~0.08 RTF on a CPU core), punctuated, cased, and strong on the
              European languages this project cares about. The default.
``whisper``   transformers Whisper. Kept as the fallback: it covers ~100
              languages against Nemotron's 40 locales, so it is the answer for
              anything outside Nemotron's list.

Nemotron is fast enough that batch transcription stops being the thing that
makes a turn feel slow — which is why it is worth preferring even though the
streaming backend exists.
"""

from __future__ import annotations

import io
from typing import Any, Protocol

import numpy as np

from sofia_galileo.audio.config import SttSettings, resolve_device
from sofia_galileo.core.logging import get_logger

log = get_logger(__name__)

MODEL_SAMPLE_RATE = 16000

# Nemotron takes locales, not bare language codes. Bare codes from the caller are
# mapped here; anything already containing a region ("pt-BR") passes through, and
# anything unknown falls back to "auto" rather than to a guessed locale.
# Transcription-ready tier first, then broad-coverage.
LOCALES: dict[str, str] = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-PT",
    "nl": "nl-NL",
    "de": "de-DE",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "vi": "vi-VN",
    "uk": "uk-UA",
    # broad-coverage tier — usable, less accurate than the above
    "pl": "pl-PL",
    "sv": "sv-SE",
    "cs": "cs-CZ",
    "nb": "nb-NO",
    "da": "da-DK",
    "bg": "bg-BG",
    "fi": "fi-FI",
    "hr": "hr-HR",
    "sk": "sk-SK",
    "zh": "zh-CN",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "et": "et-EE",
}


def to_locale(language: str | None) -> str:
    """Map a language code to a Nemotron locale, or 'auto' to let it detect.

    Auto-detection measures within ~0.1% WER of naming the language, so falling
    back to it is cheap. Naming the language is still preferred where we know it:
    it stops a strong accent being decoded as a neighbouring language mid-call.
    """
    if not language:
        return "auto"
    language = language.strip()
    if language.lower() == "auto":
        return "auto"
    if "-" in language or "_" in language:
        return language.replace("_", "-")
    return LOCALES.get(language.lower(), "auto")


def decode_audio(raw: bytes) -> np.ndarray:
    """Decode uploaded audio to 16 kHz mono float32.

    soundfile covers what actually arrives here — LiveKit's STT plugin uploads
    wav — plus flac and ogg, and mp3 where libsndfile was built with it.
    """
    import librosa
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError(f"could not decode audio: {exc}") from exc

    audio = audio.mean(axis=1)  # to mono
    if sample_rate != MODEL_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=MODEL_SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


class BatchTranscriber(Protocol):
    def transcribe(self, raw: bytes, language: str | None) -> str: ...


class NemotronTranscriber:
    """NVIDIA Nemotron 3.5 ASR, via plain transformers.

    Native transformers support (no NeMo, no trust_remote_code) is what keeps
    this hardware-agnostic: it is the same torch code path as everything else,
    so CUDA and ROCm builds differ only in the wheel they installed.
    """

    def __init__(self, settings: SttSettings) -> None:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        device = resolve_device(settings.device)
        dtype = getattr(torch, settings.dtype) if device != "cpu" else torch.float32

        log.info("stt.batch.loading", engine="nemotron", model=settings.model_id, device=device)
        self._processor = AutoProcessor.from_pretrained(settings.model_id)
        self._model = (
            AutoModelForRNNT.from_pretrained(settings.model_id, dtype=dtype).to(device).eval()
        )
        self._device = device
        self._max_new_tokens = settings.max_new_tokens
        self._default_locale = to_locale(settings.default_language)
        log.info("stt.batch.ready", engine="nemotron", locale=self._default_locale)

    def transcribe(self, raw: bytes, language: str | None) -> str:
        import torch

        audio = decode_audio(raw)
        locale = to_locale(language) if language else self._default_locale

        inputs = self._processor(
            audio, sampling_rate=MODEL_SAMPLE_RATE, return_tensors="pt", language=locale
        ).to(self._device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)

        text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        # The model pads between sentences; collapse it so the LLM sees clean prose.
        return " ".join(text.split())


class WhisperTranscriber:
    """transformers Whisper. Slower, but covers ~100 languages."""

    def __init__(self, settings: SttSettings) -> None:
        import torch
        from transformers import pipeline

        device = resolve_device(settings.device)
        dtype = getattr(torch, settings.dtype) if device != "cpu" else torch.float32

        log.info("stt.batch.loading", engine="whisper", model=settings.model_id, device=device)
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=settings.model_id,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=settings.chunk_length_s,
            batch_size=settings.batch_size,
        )
        self._default_language = settings.default_language
        log.info("stt.batch.ready", engine="whisper")

    def transcribe(self, raw: bytes, language: str | None) -> str:
        generate_kwargs: dict[str, Any] = {
            # Whisper wants a bare code, not a locale.
            "language": (language or self._default_language).split("-")[0],
            "task": "transcribe",
        }
        result = self._pipe(raw, generate_kwargs=generate_kwargs)
        return (result.get("text") or "").strip()


def build_transcriber(settings: SttSettings) -> BatchTranscriber:
    if settings.batch_engine == "nemotron":
        return NemotronTranscriber(settings)
    if settings.batch_engine == "whisper":
        return WhisperTranscriber(settings)
    raise ValueError(
        f"SOFIA_STT_BATCH_ENGINE must be nemotron or whisper; got {settings.batch_engine!r}"
    )
