"""Streaming ASR built on sherpa-onnx.

A streaming zipformer transducer decodes as audio arrives, so by the time the
person stops talking the transcript is already there — end-of-turn costs a few
tens of milliseconds instead of a full Whisper pass. That is the whole reason
this module exists; see the README for the accuracy trade-off it buys.

Endpointing is the model's own: sherpa-onnx decides an utterance has finished
from trailing silence plus what it has decoded, which is both cheaper and
better-timed than bolting a separate VAD on top.

The recognizer is hidden behind `StreamingSession` so the websocket protocol
layer can be tested against a fake, without a 300 MB model on disk.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sofia_galileo.audio.config import SttSettings
from sofia_galileo.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Transcript:
    """What the recogniser knows right now about the utterance in progress."""

    text: str
    is_final: bool


class StreamingSession(Protocol):
    """One conversation's decoding state. Not safe to share across connections."""

    def push(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        """Feed audio, get the current hypothesis back."""
        ...

    def finalize(self) -> Transcript:
        """Force the current utterance closed (an explicit commit from the client)."""
        ...


class StreamingRecognizer(Protocol):
    def session(self) -> StreamingSession: ...


# --------------------------------------------------------------------------
# model fetching
# --------------------------------------------------------------------------


def ensure_model(url: str, target: Path) -> Path:
    """Download and unpack a sherpa-onnx model release if it is not there yet.

    Kept out of the image on purpose: baking in 300 MB would hard-code the
    language, and the models volume already exists for exactly this.
    """
    if target.exists() and any(target.glob("*.onnx")):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("stt.model.downloading", url=url, target=str(target))

    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "model.tar.bz2"
        with urllib.request.urlopen(url) as response, archive.open("wb") as fh:
            shutil.copyfileobj(response, fh)

        with tarfile.open(archive, "r:bz2") as tar:
            # `data` refuses absolute paths and traversal outside the destination.
            tar.extractall(tmp_path, filter="data")

        roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"expected one directory in {url}, found {len(roots)}")

        # Move into place only once it is complete, so a killed download does not
        # leave a half-model that looks valid on the next boot.
        staged = target.with_name(target.name + ".incomplete")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.move(str(roots[0]), str(staged))
        staged.rename(target)

    log.info("stt.model.ready", target=str(target))
    return target


def _pick(model_dir: Path, prefix: str, *, prefer_int8: bool) -> str:
    """Find one of the three transducer parts without hard-coding release names.

    Filenames carry the training epoch and chunk size (`encoder-epoch-99-avg-1-
    chunk-16-left-128.onnx`), which differ per release, so we glob instead.
    """
    candidates = sorted(model_dir.glob(f"{prefix}-*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"no {prefix}-*.onnx in {model_dir}")
    int8 = [p for p in candidates if ".int8." in p.name]
    plain = [p for p in candidates if ".int8." not in p.name]
    chosen = (int8 or plain) if prefer_int8 else (plain or int8)
    return str(chosen[0])


class SherpaRecognizer:
    """Loads the model once; hands out a cheap decoding stream per connection."""

    def __init__(self, settings: SttSettings) -> None:
        import sherpa_onnx

        model_dir = ensure_model(settings.sherpa_model_url, settings.sherpa_model_dir)
        log.info(
            "stt.streaming.loading",
            model_dir=str(model_dir),
            provider=settings.sherpa_provider,
            encoder_int8=settings.sherpa_encoder_int8,
        )

        kwargs: dict[str, Any] = {
            "tokens": str(model_dir / "tokens.txt"),
            "encoder": _pick(model_dir, "encoder", prefer_int8=settings.sherpa_encoder_int8),
            # decoder and joiner are small; int8 there is free accuracy-wise.
            "decoder": _pick(model_dir, "decoder", prefer_int8=True),
            "joiner": _pick(model_dir, "joiner", prefer_int8=True),
            "num_threads": settings.sherpa_num_threads,
            "provider": settings.sherpa_provider,
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": settings.sherpa_rule1_min_trailing_silence,
            "rule2_min_trailing_silence": settings.sherpa_rule2_min_trailing_silence,
            "rule3_min_utterance_length": settings.sherpa_rule3_min_utterance_length,
        }
        if settings.sherpa_hotwords_file is not None:
            kwargs["hotwords_file"] = str(settings.sherpa_hotwords_file)
            kwargs["hotwords_score"] = settings.sherpa_hotwords_score
            kwargs["decoding_method"] = "modified_beam_search"  # hotwords need beam search

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self._normalize_case = settings.sherpa_normalize_case
        log.info("stt.streaming.ready")

    def session(self) -> StreamingSession:
        return _SherpaSession(self._recognizer, normalize_case=self._normalize_case)


def _normalize(text: str) -> str:
    """Zipformer emits `HELLO THIS IS SOFIA`. Make it read like speech.

    Casing matters more than it looks: the semantic turn detector and the LLM
    were both trained on ordinary prose, and all-caps input is out of
    distribution for them.
    """
    text = text.strip()
    if not text:
        return text
    if text.isupper():
        text = text.lower()
    return text[0].upper() + text[1:]


class _SherpaSession:
    def __init__(self, recognizer: Any, *, normalize_case: bool) -> None:
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()
        self._normalize_case = normalize_case

    def _current(self) -> str:
        text = self._recognizer.get_result(self._stream)
        return _normalize(text) if self._normalize_case else text.strip()

    def push(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        # sherpa-onnx resamples internally, so the 24 kHz LiveKit sends is fine.
        self._stream.accept_waveform(sample_rate, samples.tolist())
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._current()
        if self._recognizer.is_endpoint(self._stream):
            self._recognizer.reset(self._stream)
            # An endpoint with nothing decoded is just silence passing by.
            return Transcript(text=text, is_final=bool(text))
        return Transcript(text=text, is_final=False)

    def finalize(self) -> Transcript:
        # Padding with silence is how you tell a streaming model "that's the end":
        # it needs trailing context before it will commit its last tokens.
        self._stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32).tolist())
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._current()
        self._recognizer.reset(self._stream)
        return Transcript(text=text, is_final=True)
