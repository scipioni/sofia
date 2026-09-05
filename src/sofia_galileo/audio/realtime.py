"""OpenAI realtime-transcription websocket, served by our own streaming ASR.

Speaking OpenAI's realtime wire format (rather than something of our own) means
s2s enables streaming ASR by flipping one flag — `livekit.plugins.openai.STT`
already knows this protocol. The exchange, as the plugin implements it:

    client -> {"type": "session.update", "session": {...}}          once, on connect
    client -> {"type": "input_audio_buffer.append", "audio": b64}   every 50 ms
    client -> {"type": "input_audio_buffer.commit"}                 only if it endpoints

    server -> input_audio_buffer.speech_started                     {item_id, audio_start_ms}
    server -> conversation.item.input_audio_transcription.delta     {item_id, delta}
    server -> input_audio_buffer.speech_stopped                     {item_id, audio_end_ms}
    server -> conversation.item.input_audio_transcription.completed {item_id, transcript}

Two details that are easy to get wrong and expensive to debug:

  * Audio arrives as 24 kHz mono PCM16 — LiveKit's rate, not the model's 16 kHz.
    sherpa-onnx resamples internally, so we pass the rate through rather than
    resampling twice.
  * `delta` must be *incremental*. The plugin accumulates deltas into the interim
    transcript, so echoing the whole hypothesis each time would stutter the text.

Because the plugin is not told this is a server-endpointing model, it never sends
`input_audio_buffer.commit`: closing each turn is our job, and sherpa-onnx's own
endpoint detection does it. We honour an explicit commit anyway, so the same
service works if the plugin's behaviour changes.
"""

from __future__ import annotations

import base64
import binascii
import uuid

import anyio.to_thread
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from sofia_galileo.audio.streaming import StreamingRecognizer, StreamingSession
from sofia_galileo.core.logging import get_logger

log = get_logger(__name__)

# What LiveKit sends: 24 kHz mono signed 16-bit little-endian.
CLIENT_SAMPLE_RATE = 24000


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


class _Turn:
    """Tracks one utterance so deltas stay incremental and ids stay stable."""

    def __init__(self) -> None:
        self.item_id: str | None = None
        self.emitted = ""
        self.started_ms = 0

    def open(self, at_ms: int) -> str:
        self.item_id = _new_item_id()
        self.emitted = ""
        self.started_ms = at_ms
        return self.item_id

    def delta_for(self, text: str) -> str | None:
        """The new suffix, or None if the hypothesis was revised rather than extended.

        Greedy transducer decoding almost always extends, but beam search can
        rewrite what it already emitted. The protocol has no way to retract, so
        we stay quiet and let the `completed` event carry the truth — the plugin
        resets its accumulator on every completion, so any drift is transient.
        """
        if not text.startswith(self.emitted):
            return None
        suffix = text[len(self.emitted) :]
        return suffix or None


async def serve_realtime(
    websocket: WebSocket, recognizer: StreamingRecognizer, *, expected_key: str | None = None
) -> None:
    await websocket.accept()

    if expected_key:
        header = websocket.headers.get("authorization", "")
        if header.removeprefix("Bearer ").strip() != expected_key:
            await websocket.send_json(
                {"type": "error", "error": {"message": "invalid api key", "code": "unauthorized"}}
            )
            await websocket.close(code=1008)
            return

    session: StreamingSession = recognizer.session()
    turn = _Turn()
    samples_seen = 0

    def ms() -> int:
        return int(samples_seen / CLIENT_SAMPLE_RATE * 1000)

    async def close_turn(text: str) -> None:
        if turn.item_id is None:
            return
        await websocket.send_json(
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": turn.item_id,
                "audio_end_ms": ms(),
            }
        )
        await websocket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": turn.item_id,
                "transcript": text,
            }
        )
        log.info("stt.turn.final", item_id=turn.item_id, chars=len(text))
        turn.item_id = None
        turn.emitted = ""

    log.info("stt.realtime.connected")
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")

            if kind == "input_audio_buffer.append":
                try:
                    raw = base64.b64decode(message.get("audio", ""))
                except (binascii.Error, ValueError):
                    await websocket.send_json(
                        {"type": "error", "error": {"message": "audio is not valid base64"}}
                    )
                    continue
                if not raw:
                    continue

                pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                samples_seen += len(pcm)

                # Decoding is CPU-bound C++; keeping it off the event loop means a
                # slow chunk cannot delay reading the next one off the socket.
                result = await anyio.to_thread.run_sync(session.push, pcm, CLIENT_SAMPLE_RATE)

                if result.text and turn.item_id is None:
                    item_id = turn.open(ms())
                    await websocket.send_json(
                        {
                            "type": "input_audio_buffer.speech_started",
                            "item_id": item_id,
                            "audio_start_ms": turn.started_ms,
                        }
                    )

                # Emit the delta even on the final chunk, so the deltas a client
                # accumulates add up to exactly the transcript it is about to be
                # handed. Skipping it leaves the last word missing from any
                # interim display.
                if result.text and turn.item_id is not None:
                    delta = turn.delta_for(result.text)
                    if delta:
                        turn.emitted = result.text
                        await websocket.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": turn.item_id,
                                "delta": delta,
                            }
                        )

                if result.is_final:
                    await close_turn(result.text)

            elif kind == "input_audio_buffer.commit":
                result = await anyio.to_thread.run_sync(session.finalize)
                if result.text:
                    if turn.item_id is None:
                        turn.open(ms())
                    await close_turn(result.text)

            elif kind == "session.update":
                # Config is ours, not the caller's: the model, the language and the
                # endpointing rules come from this service's own settings. Ack so
                # the client sees a well-formed session, and ignore the contents.
                await websocket.send_json(
                    {"type": "session.updated", "session": message.get("session", {})}
                )

            elif kind == "input_audio_buffer.clear":
                session = recognizer.session()
                turn = _Turn()

    except WebSocketDisconnect:
        log.info("stt.realtime.disconnected", audio_ms=ms())
    except Exception as exc:
        log.exception("stt.realtime.failed")
        try:
            await websocket.send_json({"type": "error", "error": {"message": str(exc)}})
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
