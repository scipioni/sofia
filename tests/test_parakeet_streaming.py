"""Engine selection, resampler equivalence, and (opt-in) the real parakeet.cpp
engine.

The engine-selection and resampler tests need neither the library nor a model
and always run. The real-engine test mirrors test_streaming_asr.py's
precedent: skipped unless both a built `libparakeet.so` and a downloaded GGUF
are available.

    SOFIA_STT_PARAKEET_LIBRARY_PATH=/path/to/libparakeet.so \
    SOFIA_STT_PARAKEET_MODEL_PATH=/path/to/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
    SOFIA_STT_PARAKEET_TEST_WAV=/path/to/16kHz-mono-italian.wav \
        uv run pytest tests/test_parakeet_streaming.py -v
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from sofia_galileo.audio import streaming
from sofia_galileo.audio.config import SttSettings
from sofia_galileo.audio.streaming import ParakeetRecognizer, _ParakeetSession

# --------------------------------------------------------------------------
# engine selection (no model, no library needed)
# --------------------------------------------------------------------------


def test_build_recognizer_defaults_to_parakeet(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(streaming, "ParakeetRecognizer", lambda settings: sentinel)
    # This must check the class default, not whatever a developer's local
    # .env sets SOFIA_STT_STREAMING_ENGINE to — and `task test`'s own
    # `dotenv: [".env"]` directive injects .env straight into the process
    # environment, so _env_file=None alone would not be enough here; the real
    # variable itself has to go.
    monkeypatch.delenv("SOFIA_STT_STREAMING_ENGINE", raising=False)
    assert streaming.build_recognizer(SttSettings(_env_file=None)) is sentinel


def test_build_recognizer_selects_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(streaming, "SherpaRecognizer", lambda settings: sentinel)
    settings = SttSettings(streaming_engine="sherpa")
    assert streaming.build_recognizer(settings) is sentinel


def test_build_recognizer_rejects_unknown_engine() -> None:
    settings = SttSettings(streaming_engine="bogus")
    with pytest.raises(ValueError, match="sherpa or parakeet"):
        streaming.build_recognizer(settings)


# --------------------------------------------------------------------------
# resampler equivalence (design.md D5): framed must match whole
# --------------------------------------------------------------------------


class _StubStream:
    def feed(self, pcm_16k_mono: np.ndarray) -> tuple[str, bool]:
        return "", False

    def finalize(self) -> str:
        return ""

    def close(self) -> None:
        pass


class _StubContext:
    def stream(self, lang: str) -> _StubStream:
        return _StubStream()


class _CapturingStream(_StubStream):
    """Records exactly what push()/finalize() feed the C-API, so tests can
    inspect the resampled audio without reaching into private methods."""

    def __init__(self) -> None:
        self.fed: list[np.ndarray] = []

    def feed(self, pcm_16k_mono: np.ndarray) -> tuple[str, bool]:
        self.fed.append(np.array(pcm_16k_mono, copy=True))
        return "x", False  # non-empty so the session accumulates text


class _CapturingContext:
    def __init__(self) -> None:
        self.stream_obj = _CapturingStream()

    def stream(self, lang: str) -> _CapturingStream:
        return self.stream_obj


def _tone(seconds: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(
        np.float32
    )


def _push_all(session: _ParakeetSession, sig: np.ndarray, sample_rate: int, frame: int) -> None:
    for i in range(0, len(sig), frame):
        session.push(sig[i : i + frame], sample_rate)


def test_resample_framed_matches_whole_once_finalized() -> None:
    """A stateless per-frame resample would introduce a discontinuity at every
    50ms frame boundary; soxr.ResampleStream must not — and finalize() must
    flush its last buffered samples rather than silently drop them."""
    soxr = pytest.importorskip("soxr")

    sr_in = 24000
    sig = _tone(2.0, sr_in)
    whole = soxr.resample(sig, sr_in, 16000, quality="HQ")

    ctx = _CapturingContext()
    session = _ParakeetSession(ctx, "en", endpoint_silence=999.0, min_utterance_length=999.0)
    _push_all(session, sig, sr_in, frame=1200)  # 50ms @ 24kHz, realtime.py's cadence
    session.finalize()
    framed = np.concatenate(ctx.stream_obj.fed)

    # Equal length, not just an equal prefix: without flushing the last chunk
    # via finalize(), framed comes back short (soxr buffers an algorithmic
    # delay internally) and a prefix-only comparison would miss that.
    assert len(framed) == len(whole)
    np.testing.assert_allclose(whole, framed, atol=1e-5)


def test_without_finalize_trailing_audio_is_not_yet_fed() -> None:
    """Guards the bug finalize()'s flush fixes: mid-session chunks never emit
    the last ~tens of ms of resampled audio on their own — that only comes out
    once finalize() flushes, not before."""
    pytest.importorskip("soxr")
    sig = _tone(3.0, 24000)
    ctx = _CapturingContext()
    session = _ParakeetSession(ctx, "en", endpoint_silence=999.0, min_utterance_length=999.0)
    _push_all(session, sig, 24000, frame=1200)  # no finalize() yet
    fed_so_far = sum(len(c) for c in ctx.stream_obj.fed)
    assert fed_so_far < len(sig) * 2 / 3


def test_finalize_flushes_resampler_and_feeds_the_tail() -> None:
    pytest.importorskip("soxr")
    sig = _tone(3.0, 24000)
    ctx = _CapturingContext()
    session = _ParakeetSession(ctx, "en", endpoint_silence=999.0, min_utterance_length=999.0)
    _push_all(session, sig, 24000, frame=1200)
    session.finalize()
    fed_total = sum(len(c) for c in ctx.stream_obj.fed)
    # 24kHz -> 16kHz is exactly 2:3; length should match within one sample of
    # rounding, not drift with signal length the way an approximate ratio would.
    assert abs(fed_total - len(sig) * 2 / 3) <= 2


def test_resample_passthrough_at_native_rate() -> None:
    sig = _tone(0.5, 16000)
    ctx = _CapturingContext()
    session = _ParakeetSession(ctx, "en", endpoint_silence=999.0, min_utterance_length=999.0)
    _push_all(session, sig, 16000, frame=800)
    fed = np.concatenate(ctx.stream_obj.fed)
    np.testing.assert_array_equal(fed, sig.astype(np.float32))


# --------------------------------------------------------------------------
# session behaviour against a stub (no model needed)
# --------------------------------------------------------------------------


def test_silence_with_nothing_decoded_stays_non_final() -> None:
    session = _ParakeetSession(
        _StubContext(), "en", endpoint_silence=0.1, min_utterance_length=999.0
    )
    silence = np.zeros(800, dtype=np.float32)  # 50ms @ 16kHz, all zero -> RMS 0
    for _ in range(10):
        result = session.push(silence, 16000)
        assert result.text == ""
        assert not result.is_final


def test_endpoint_fires_after_trailing_silence_once_text_exists() -> None:
    class _TalkThenQuietStream(_StubStream):
        def __init__(self) -> None:
            self.calls = 0

        def feed(self, pcm_16k_mono: np.ndarray) -> tuple[str, bool]:
            self.calls += 1
            return ("hello" if self.calls == 1 else ""), False

    class _Ctx:
        def stream(self, lang: str) -> _TalkThenQuietStream:
            return _TalkThenQuietStream()

    session = _ParakeetSession(_Ctx(), "en", endpoint_silence=0.1, min_utterance_length=999.0)
    loud = np.full(800, 0.5, dtype=np.float32)  # 50ms @ 16kHz, well above the silence threshold
    silence = np.zeros(800, dtype=np.float32)

    first = session.push(loud, 16000)
    assert first.text == "hello"
    assert not first.is_final

    # Two silent 50ms frames exceed the 0.1s threshold.
    session.push(silence, 16000)
    result = session.push(silence, 16000)
    assert result.is_final
    assert result.text == "hello"


# --------------------------------------------------------------------------
# the real engine (opt-in, needs the library and a downloaded GGUF)
# --------------------------------------------------------------------------

LIBRARY_PATH = os.environ.get("SOFIA_STT_PARAKEET_LIBRARY_PATH", "libparakeet.so")
MODEL_PATH = os.environ.get("SOFIA_STT_PARAKEET_MODEL_PATH", "")
TEST_WAV = os.environ.get("SOFIA_STT_PARAKEET_TEST_WAV", "")


def _library_loadable() -> bool:
    from sofia_galileo.audio import parakeet_capi

    try:
        parakeet_capi.load_library(LIBRARY_PATH)
        return True
    except parakeet_capi.ParakeetLibraryError:
        return False


_skip_no_real_engine = pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).is_file() or not TEST_WAV or not _library_loadable(),
    reason=(
        "set SOFIA_STT_PARAKEET_LIBRARY_PATH, SOFIA_STT_PARAKEET_MODEL_PATH "
        "and SOFIA_STT_PARAKEET_TEST_WAV to run against the real engine"
    ),
)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, w.getframerate()


@_skip_no_real_engine
def test_real_engine_transcribes_and_endpoints() -> None:
    recognizer = ParakeetRecognizer(
        SttSettings(
            parakeet_library_path=LIBRARY_PATH,
            parakeet_model_path=Path(MODEL_PATH),
            parakeet_endpoint_silence=0.8,
            default_language="it",
            backend="streaming",
            streaming_engine="parakeet",
        )
    )
    audio, sample_rate = _read_wav(Path(TEST_WAV))
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
    assert len(partials) > 3, f"expected progressive partials, got {partials}"


@_skip_no_real_engine
def test_stream_feed_releases_the_gil() -> None:
    """realtime.py dispatches push() through anyio.to_thread.run_sync so a slow
    chunk cannot delay reading the next one off the socket. That offload is
    only worth anything if the ctypes call underneath actually releases the
    GIL — ctypes CDLL calls do this by default, but "by default" is exactly
    the kind of assumption worth a real assertion."""
    import threading
    import time

    from sofia_galileo.audio import parakeet_capi

    lib = parakeet_capi.load_library(LIBRARY_PATH)
    ctx = parakeet_capi.ParakeetContext(lib, MODEL_PATH)
    stream = ctx.stream("it")

    audio, sample_rate = _read_wav(Path(TEST_WAV))
    assert sample_rate == 16000, "test wav must already be 16kHz for this to feed in one call"

    progressed = threading.Event()

    def spin() -> None:
        # If the GIL were held for the whole feed() call, this thread would
        # not run until feed() returns, and progressed would only ever be set
        # after the fact rather than concurrently with it.
        count = 0
        while not stop.is_set():
            count += 1
            if count > 1000:
                progressed.set()

    stop = threading.Event()
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        # The whole clip in one feed() call, so it takes long enough (tens of
        # ms) for a concurrent thread to visibly make progress during it.
        start = time.monotonic()
        stream.feed(audio)
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        t.join(timeout=5)

    assert elapsed > 0.01, "feed call was too fast to meaningfully test GIL release"
    assert progressed.is_set(), "background thread made no progress during feed() — GIL held?"
