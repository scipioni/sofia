"""Streaming ASR: two engines behind one `StreamingRecognizer` interface.

``sherpa``    A streaming zipformer transducer via sherpa-onnx. Decodes as audio
              arrives, so by the time the person stops talking the transcript
              is already there. No Italian, no punctuation. Endpointing is the
              model's own: sherpa-onnx decides an utterance has finished from
              trailing silence plus what it has decoded, which is cheaper and
              better-timed than bolting a separate VAD on top.
``parakeet``  Nemotron 3.5 ASR via parakeet.cpp/ggml (ctypes). Punctuated,
              cased, covers Italian. Has no end-of-utterance signal of its own
              — that belongs only to the separate, English-only
              nvidia/parakeet_realtime_eou_120m-v1 model — so its session
              implements its own trailing-silence rule instead (see
              design.md D6 in openspec/changes/add-parakeet-streaming-asr).

Which one a deployment runs is `SttSettings.streaming_engine`; `sherpa` is the
default. Both are hidden behind `StreamingRecognizer`/`StreamingSession` so the
websocket protocol layer (realtime.py) can be tested against a fake, without a
model on disk of either kind.
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


def ensure_file(url: str, target: Path) -> Path:
    """Download a single file into place if it is not there yet.

    Same staged-move discipline as ensure_model() above, but for one file (a
    GGUF) rather than a directory extracted from a release tarball: download
    to a temporary path next to the target and rename into place only once
    complete, so a killed download cannot leave a half-file that looks usable
    on the next boot.
    """
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("stt.model.downloading", url=url, target=str(target))

    tmp = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.rename(target)

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
            engine="sherpa",
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
        log.info("stt.streaming.ready", engine="sherpa")

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


# --------------------------------------------------------------------------
# parakeet.cpp (Nemotron 3.5 ASR) backend
# --------------------------------------------------------------------------

_PARAKEET_SAMPLE_RATE = 16000
# ~ -40 dBFS. Below this a 50ms frame counts as silence for endpointing
# purposes — the same job sherpa_rule2/rule3 do via the model's own decoder
# state, which the flat C-API does not expose.
_PARAKEET_SILENCE_RMS = 0.01


class ParakeetRecognizer:
    """Loads Nemotron once via ctypes; hands out a cheap stream per connection."""

    def __init__(self, settings: SttSettings) -> None:
        from sofia_galileo.audio import parakeet_capi
        from sofia_galileo.audio.batch import to_locale

        model_path = ensure_file(settings.parakeet_model_url, settings.parakeet_model_path)
        log.info("stt.streaming.loading", engine="parakeet", model_path=str(model_path))

        lib = parakeet_capi.load_library(settings.parakeet_library_path)
        self._ctx = parakeet_capi.ParakeetContext(lib, str(model_path))
        self._lang = to_locale(settings.default_language)
        self._endpoint_silence = settings.parakeet_endpoint_silence
        self._min_utterance_length = settings.parakeet_min_utterance_length
        log.info("stt.streaming.ready", engine="parakeet", lang=self._lang)

    def session(self) -> StreamingSession:
        return _ParakeetSession(
            self._ctx, self._lang, self._endpoint_silence, self._min_utterance_length
        )


class _ParakeetSession:
    def __init__(
        self,
        ctx: Any,
        lang: str,
        endpoint_silence: float,
        min_utterance_length: float,
    ) -> None:
        self._ctx = ctx
        self._lang = lang
        self._endpoint_silence = endpoint_silence
        self._min_utterance_length = min_utterance_length
        self._stream = ctx.stream(lang)
        self._resampler: Any = None
        self._resampler_rate: int | None = None
        self._text = ""
        self._trailing_silence_s = 0.0
        self._utterance_s = 0.0

    def _resample(self, samples: np.ndarray, sample_rate: int, *, last: bool = False) -> np.ndarray:
        if sample_rate == _PARAKEET_SAMPLE_RATE:
            return samples.astype(np.float32, copy=False)
        if self._resampler is None or self._resampler_rate != sample_rate:
            # A session is tied to the rate of its first frame: LiveKit never
            # changes rate mid-call, so this builds the resampler exactly once.
            # soxr.ResampleStream carries its filter delay line across calls —
            # a stateless per-frame resample would introduce a discontinuity at
            # every 50ms frame boundary (design.md D5).
            import soxr

            self._resampler = soxr.ResampleStream(
                sample_rate, _PARAKEET_SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )
            self._resampler_rate = sample_rate
        return self._resampler.resample_chunk(samples.astype(np.float32, copy=False), last=last)

    def _reset(self) -> None:
        self._stream.close()
        self._stream = self._ctx.stream(self._lang)
        self._text = ""
        self._trailing_silence_s = 0.0
        self._utterance_s = 0.0

    def push(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        if len(samples) == 0:
            return Transcript(text=self._text, is_final=False)

        chunk_s = len(samples) / sample_rate
        rms = float(np.sqrt(np.mean(np.square(samples))))
        pcm16k = self._resample(samples, sample_rate)

        # eou is intentionally ignored: Nemotron does not populate it (see
        # design.md D6). Turn boundaries come from the trailing-silence rule
        # below instead.
        new_text, _eou = self._stream.feed(pcm16k)
        if new_text:
            self._text += new_text

        if rms < _PARAKEET_SILENCE_RMS:
            self._trailing_silence_s += chunk_s
        else:
            self._trailing_silence_s = 0.0
        self._utterance_s += chunk_s

        endpointed = bool(self._text) and (
            self._trailing_silence_s >= self._endpoint_silence
            or self._utterance_s >= self._min_utterance_length
        )
        if endpointed:
            text = self._text
            self._reset()
            return Transcript(text=text, is_final=True)
        return Transcript(text=self._text, is_final=False)

    def finalize(self) -> Transcript:
        # soxr.ResampleStream buffers an algorithmic delay internally; without
        # this flush the last ~tens of ms of audio — exactly the trailing
        # context the model needs to commit its last tokens — is silently
        # dropped (caught by test_resample_ratio_is_exact_two_thirds).
        if self._resampler is not None:
            flushed = self._resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            if len(flushed):
                new_text, _eou = self._stream.feed(flushed)
                if new_text:
                    self._text += new_text
            # A flushed ResampleStream raises "Input after last input" on
            # further use; clear() re-arms it for the next utterance in this
            # same connection (input_audio_buffer.commit does not end the
            # session — see realtime.py).
            self._resampler.clear()

        tail = self._stream.finalize()
        if tail:
            self._text += tail
        text = self._text
        self._reset()
        return Transcript(text=text, is_final=True)


def build_recognizer(settings: SttSettings) -> StreamingRecognizer:
    if settings.streaming_engine == "sherpa":
        return SherpaRecognizer(settings)
    if settings.streaming_engine == "parakeet":
        return ParakeetRecognizer(settings)
    raise ValueError(
        f"SOFIA_STT_STREAMING_ENGINE must be sherpa or parakeet; got {settings.streaming_engine!r}"
    )
