"""Protocol tests for the realtime-transcription websocket.

These run against a fake recogniser, so they check the thing that actually
breaks in integration — the wire format LiveKit's plugin expects — without
needing a 300 MB model on disk. The model itself is exercised separately in
tests/test_streaming_asr.py.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from sofia_galileo.audio.realtime import CLIENT_SAMPLE_RATE, serve_realtime
from sofia_galileo.audio.streaming import Transcript


class ScriptedSession:
    """Returns a scripted Transcript for each push, ignoring the audio."""

    def __init__(self, script: list[Transcript]) -> None:
        self.script = list(script)
        self.pushes = 0

    def push(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        assert sample_rate == CLIENT_SAMPLE_RATE
        self.pushes += 1
        return self.script.pop(0) if self.script else Transcript("", False)

    def finalize(self) -> Transcript:
        return Transcript("Committed text", True)


class ScriptedRecognizer:
    def __init__(self, script: list[Transcript]) -> None:
        self._script = script
        self.sessions: list[ScriptedSession] = []

    def session(self) -> ScriptedSession:
        s = ScriptedSession(self._script)
        self.sessions.append(s)
        return s


def app_for(recognizer: ScriptedRecognizer, *, key: str | None = None) -> FastAPI:
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def route(websocket: WebSocket) -> None:
        await serve_realtime(websocket, recognizer, expected_key=key)

    return app


def audio_frame(ms: int = 50) -> dict:
    """One base64 PCM16 chunk, shaped like what the livekit plugin sends."""
    samples = np.zeros(CLIENT_SAMPLE_RATE * ms // 1000, dtype="<i2")
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(samples.tobytes()).decode(),
    }


def drain(ws, count: int) -> list[dict]:
    return [ws.receive_json() for _ in range(count)]


def test_partials_are_incremental_then_completed() -> None:
    """The plugin accumulates deltas, so each must be only the new suffix."""
    recognizer = ScriptedRecognizer(
        [
            Transcript("Hello", False),
            Transcript("Hello there", False),
            Transcript("Hello there friend", True),
        ]
    )
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        for _ in range(3):
            ws.send_json(audio_frame())
        events = drain(ws, 6)

    kinds = [e["type"] for e in events]
    assert kinds == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]

    deltas = [e["delta"] for e in events if e["type"].endswith("transcription.delta")]
    assert deltas == ["Hello", " there", " friend"]
    assert "".join(deltas) == events[-1]["transcript"]

    item_ids = {e["item_id"] for e in events}
    assert len(item_ids) == 1, "one utterance must carry one item_id"


def test_revised_hypothesis_emits_no_delta() -> None:
    """A rewrite cannot be expressed as a delta; `completed` carries the truth."""
    recognizer = ScriptedRecognizer(
        [
            Transcript("Hello wold", False),
            Transcript("Hello world", False),  # revision, not an extension
            Transcript("Hello world", True),
        ]
    )
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        for _ in range(3):
            ws.send_json(audio_frame())
        events = drain(ws, 4)

    deltas = [e for e in events if e["type"].endswith("transcription.delta")]
    assert len(deltas) == 1 and deltas[0]["delta"] == "Hello wold"
    assert events[-1]["transcript"] == "Hello world"


def test_silence_produces_no_events() -> None:
    recognizer = ScriptedRecognizer([Transcript("", False)] * 3)
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        for _ in range(3):
            ws.send_json(audio_frame())
        # Nothing to receive; prove it by round-tripping a session.update.
        ws.send_json({"type": "session.update", "session": {"x": 1}})
        assert ws.receive_json()["type"] == "session.updated"


def test_two_utterances_get_distinct_item_ids() -> None:
    recognizer = ScriptedRecognizer(
        [
            Transcript("First", True),
            Transcript("Second", True),
        ]
    )
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        ws.send_json(audio_frame())
        first = drain(ws, 4)
        ws.send_json(audio_frame())
        second = drain(ws, 4)

    assert first[-1]["transcript"] == "First"
    assert second[-1]["transcript"] == "Second"
    assert first[-1]["item_id"] != second[-1]["item_id"]


def test_explicit_commit_finalizes() -> None:
    """Honoured even though the plugin only sends it for server-endpointed models."""
    recognizer = ScriptedRecognizer([Transcript("Partial", False)])
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        ws.send_json(audio_frame())
        drain(ws, 2)  # speech_started + delta
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = drain(ws, 2)

    assert events[-1]["type"] == "conversation.item.input_audio_transcription.completed"
    assert events[-1]["transcript"] == "Committed text"


def test_audio_timings_advance_with_the_stream() -> None:
    recognizer = ScriptedRecognizer(
        [Transcript("", False), Transcript("Hi", False), Transcript("Hi", True)]
    )
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        for _ in range(3):
            ws.send_json(audio_frame(ms=50))
        events = drain(ws, 4)

    started = next(e for e in events if e["type"] == "input_audio_buffer.speech_started")
    stopped = next(e for e in events if e["type"] == "input_audio_buffer.speech_stopped")
    assert started["audio_start_ms"] == 100  # speech appeared on the 2nd 50ms frame
    assert stopped["audio_end_ms"] == 150
    assert stopped["audio_end_ms"] > started["audio_start_ms"]


def test_bad_base64_is_reported_not_fatal() -> None:
    recognizer = ScriptedRecognizer([Transcript("Fine", True)])
    client = TestClient(app_for(recognizer))
    with client.websocket_connect("/v1/realtime") as ws:
        ws.send_json({"type": "input_audio_buffer.append", "audio": "!!!not base64!!!"})
        assert ws.receive_json()["type"] == "error"
        # The connection survives and keeps working.
        ws.send_json(audio_frame())
        assert ws.receive_json()["type"] == "input_audio_buffer.speech_started"


@pytest.mark.parametrize(
    ("header", "expected"),
    [("Bearer right-key", "input_audio_buffer.speech_started"), ("Bearer wrong-key", "error")],
)
def test_api_key_is_enforced_when_configured(header: str, expected: str) -> None:
    recognizer = ScriptedRecognizer([Transcript("Hi", False)])
    client = TestClient(app_for(recognizer, key="right-key"))
    with client.websocket_connect("/v1/realtime", headers={"Authorization": header}) as ws:
        ws.send_json(audio_frame())
        assert ws.receive_json()["type"] == expected
