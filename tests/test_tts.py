"""KokoroEngine's chunk generator: equivalence with the old eager path, using
a fake pipeline so the test is deterministic. Kokoro's own inference is NOT
deterministic between independent calls (confirmed empirically: two
unmodified synthesize() calls on the same input already differ, max abs diff
~0.08) -- bit-comparing against the real model would test Kokoro's own
randomness, not this refactor.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sofia_galileo.audio import tts_app
from sofia_galileo.audio.config import TtsSettings
from sofia_galileo.audio.tts_app import KokoroEngine, _encode, _pcm16_bytes


class _FakePipeline:
    """Stands in for kokoro.KPipeline: yields (graphemes, phonemes, audio)."""

    def __init__(self, segments: list[np.ndarray | None]) -> None:
        self._segments = segments

    def __call__(self, text: str, voice: str, speed: float) -> _FakePipeline:
        return self

    def __iter__(self):
        for seg in self._segments:
            yield ("g", "p", seg)


def _engine_with_fake_pipeline(segments: list[np.ndarray | None]) -> KokoroEngine:
    engine = KokoroEngine.__new__(KokoroEngine)  # skip __init__: no torch/model needed
    engine._pipelines = {"a": _FakePipeline(segments)}  # noqa: SLF001 -- test-only setup
    return engine


def test_synthesize_chunks_yields_one_array_per_segment() -> None:
    segments = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]
    engine = _engine_with_fake_pipeline(segments)

    chunks = list(engine.synthesize_chunks("hello", "af_heart", 1.0))

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], segments[0])
    np.testing.assert_array_equal(chunks[1], segments[1])


def test_synthesize_chunks_filters_none_segments() -> None:
    segments = [np.array([1.0], dtype=np.float32), None, np.array([2.0], dtype=np.float32)]
    engine = _engine_with_fake_pipeline(segments)

    chunks = list(engine.synthesize_chunks("hello", "af_heart", 1.0))

    assert len(chunks) == 2


def test_synthesize_equals_concatenated_chunks() -> None:
    """The equivalence task 2.1 asks for: synthesize() must be exactly
    np.concatenate(list(synthesize_chunks(...))) for the same underlying run,
    not a second independent (and non-deterministic) model call."""
    segments = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0], dtype=np.float32),
    ]
    engine = _engine_with_fake_pipeline(segments)

    chunks = list(engine.synthesize_chunks("hello", "af_heart", 1.0))
    whole = np.concatenate(chunks)

    # A second engine instance with the same fake segments stands in for a
    # second call against the same (deterministic, here) underlying source.
    engine2 = _engine_with_fake_pipeline(segments)
    synthesized = engine2.synthesize("hello", "af_heart", 1.0)

    np.testing.assert_array_equal(synthesized, whole)


def test_synthesize_empty_when_no_audio_segments() -> None:
    engine = _engine_with_fake_pipeline([None, None])

    result = engine.synthesize("hello", "af_heart", 1.0)

    assert result.dtype == np.float32
    assert len(result) == 0


def test_synthesize_chunks_empty_generator_yields_nothing() -> None:
    engine = _engine_with_fake_pipeline([])

    assert list(engine.synthesize_chunks("hello", "af_heart", 1.0)) == []


# --------------------------------------------------------------------------
# endpoint-level: /v1/audio/speech, with a fake engine so tests are fast and
# deterministic (no real Kokoro/torch involved) -- see spec.md
# --------------------------------------------------------------------------


class _FakeEngine:
    """Stands in for KokoroEngine at the app level: no torch, no model,
    controllable segments so streaming behaviour is directly assertable."""

    def __init__(self, segments_by_text: dict[str, list[np.ndarray]]) -> None:
        self.device = "cpu"
        self._segments_by_text = segments_by_text

    def synthesize_chunks(self, text: str, voice: str, speed: float):
        yield from self._segments_by_text.get(text, [np.zeros(10, dtype=np.float32)])

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        chunks = list(self.synthesize_chunks(text, voice, speed))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = _FakeEngine(
        segments_by_text={
            "multi": [
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.3, 0.4, 0.5], dtype=np.float32),
            ],
        }
    )
    monkeypatch.setattr(tts_app, "KokoroEngine", lambda settings: fake)
    app = tts_app.create_app(TtsSettings(_env_file=None))
    with TestClient(app) as c:
        yield c


async def test_stream_speech_yields_one_chunk_per_segment() -> None:
    """The actual streaming behaviour lives in _stream_speech(); test it
    directly rather than through TestClient, whose ASGI transport coalesces
    separate yields into one blob (confirmed empirically -- exactly the
    buffering risk flagged in tasks.md 1.1) and so cannot verify chunk
    boundaries the way a real socket can (already confirmed against a real
    server in tasks.md 3.2)."""
    segments = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.3, 0.4, 0.5], dtype=np.float32),
    ]
    engine = _FakeEngine(segments_by_text={"multi": segments})

    stream = tts_app._stream_speech(  # noqa: SLF001 -- testing the streaming behavior directly
        engine, "multi", "af_heart", 1.0, "pcm", 24000
    )
    chunks = [c async for c in stream]

    # spec: "Multi-segment input streamed as PCM"
    assert len(chunks) == 2
    assert chunks == [_pcm16_bytes(segments[0]), _pcm16_bytes(segments[1])]

    # spec: "Concatenated stream equals the full synthesis"
    assert b"".join(chunks) == _pcm16_bytes(np.concatenate(segments))


def test_pcm_endpoint_returns_correct_final_content(client: TestClient) -> None:
    """Endpoint-level: correct final bytes over HTTP. Chunk-boundary
    behaviour is covered by the direct generator test above."""
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "multi", "response_format": "pcm", "voice": "af_heart"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/pcm")
    expected = _pcm16_bytes(
        np.concatenate(
            [np.array([0.1, 0.2], dtype=np.float32), np.array([0.3, 0.4, 0.5], dtype=np.float32)]
        )
    )
    assert resp.content == expected


def test_wav_delivers_exactly_one_http_chunk(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={"input": "multi", "response_format": "wav", "voice": "af_heart"},
    ) as resp:
        assert resp.status_code == 200
        raw_chunks = list(resp.iter_bytes())

    # spec: "WAV request is not chunked early"
    assert len(raw_chunks) == 1
    assert raw_chunks[0][:4] == b"RIFF"

    # spec: "Output is unchanged by this capability" -- the endpoint must
    # produce exactly what the (unchanged) _encode() function would for the
    # same audio, proving the generator wrapper adds no discrepancy.
    whole_audio = np.concatenate(
        [np.array([0.1, 0.2], dtype=np.float32), np.array([0.3, 0.4, 0.5], dtype=np.float32)]
    )
    assert raw_chunks[0] == _encode(whole_audio, 24000, "wav")


def test_unsupported_format_rejected_before_synthesis(client: TestClient) -> None:
    resp = client.post("/v1/audio/speech", json={"input": "multi", "response_format": "mp3"})
    assert resp.status_code == 400
    assert "mp3" in resp.json()["error"]["message"]


def test_empty_input_returns_empty_body_without_synthesis(client: TestClient) -> None:
    resp = client.post("/v1/audio/speech", json={"input": "   ", "response_format": "wav"})
    assert resp.status_code == 200
    assert resp.content == b""
