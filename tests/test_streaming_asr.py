"""Streaming ASR against the real sherpa-onnx model.

Skipped unless a model is on disk, so the default test run stays fast and
offline. Point SOFIA_STT_SHERPA_MODEL_DIR at an unpacked release to run it:

    SOFIA_STT_SHERPA_MODEL_DIR=/path/to/sherpa-onnx-streaming-zipformer-en-... \
        uv run pytest tests/test_streaming_asr.py -v

Every sherpa-onnx release ships `test_wavs/` with reference transcripts, so the
fixture audio comes from the model itself.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from sofia_galileo.audio.config import SttSettings
from sofia_galileo.audio.streaming import SherpaRecognizer, _normalize

MODEL_DIR = os.environ.get("SOFIA_STT_SHERPA_MODEL_DIR", "")

pytestmark = pytest.mark.skipif(
    not MODEL_DIR or not Path(MODEL_DIR).is_dir(),
    reason="set SOFIA_STT_SHERPA_MODEL_DIR to an unpacked sherpa-onnx model",
)


@pytest.fixture(scope="module")
def recognizer() -> SherpaRecognizer:
    return SherpaRecognizer(
        SttSettings(
            sherpa_model_dir=Path(MODEL_DIR),
            sherpa_rule2_min_trailing_silence=0.8,
            backend="streaming",
        )
    )


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, w.getframerate()


def test_partials_grow_then_endpoint(recognizer: SherpaRecognizer) -> None:
    wav = next(iter(sorted(Path(MODEL_DIR).glob("test_wavs/*.wav"))))
    audio, sample_rate = read_wav(wav)
    session = recognizer.session()

    partials: list[str] = []
    final = ""
    chunk = sample_rate // 20  # 50ms, matching the livekit plugin
    for i in range(0, len(audio), chunk):
        result = session.push(audio[i : i + chunk], sample_rate)
        if result.text and (not partials or result.text != partials[-1]):
            partials.append(result.text)
        if result.is_final:
            final = result.text
            break

    if not final:
        final = session.finalize().text

    assert final, "expected a transcript"
    # The point of streaming: text existed well before the audio ran out.
    assert len(partials) > 3, f"expected progressive partials, got {partials}"
    assert partials[0] != final
    assert final.startswith(partials[0][: len(partials[0]) // 2])


def test_second_utterance_reuses_the_session(recognizer: SherpaRecognizer) -> None:
    """A session must reset cleanly, or turn two inherits turn one's words."""
    wav = next(iter(sorted(Path(MODEL_DIR).glob("test_wavs/*.wav"))))
    audio, sample_rate = read_wav(wav)
    session = recognizer.session()

    finals = []
    for _ in range(2):
        for i in range(0, len(audio), sample_rate // 20):
            session.push(audio[i : i + sample_rate // 20], sample_rate)
        finals.append(session.finalize().text)

    assert finals[0] and finals[0] == finals[1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HELLO THIS IS SOFIA", "Hello this is sofia"),
        ("Already cased text", "Already cased text"),
        ("  padded  ", "Padded"),
        ("", ""),
    ],
)
def test_case_normalisation(raw: str, expected: str) -> None:
    """Uppercase output is out-of-distribution for the LLM and turn detector."""
    assert _normalize(raw) == expected
