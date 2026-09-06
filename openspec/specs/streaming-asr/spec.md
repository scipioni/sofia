## Purpose

Defines how the `stt` service transcribes speech while the person is still
talking: which recogniser serves a deployment, how a streaming session behaves
over its websocket, how a turn is opened and closed, and what callers observe
when a recogniser is unavailable or fails mid-session.

## Requirements

### Requirement: Streaming recogniser selection

The service SHALL select one streaming recogniser per deployment from
configuration. The supported values are `sherpa` and `parakeet`, and the default
SHALL be `sherpa`.

The service SHALL refuse to start on an unrecognised value rather than silently
choosing a default, and the resulting error SHALL name both the setting and the
rejected value.

The service SHALL NOT substitute a different recogniser from the one selected,
under any failure condition.

#### Scenario: No engine configured

- **WHEN** the service starts with the streaming backend enabled and no streaming
  engine configured
- **THEN** it serves the `sherpa` recogniser

#### Scenario: Parakeet engine selected

- **WHEN** the streaming engine is configured as `parakeet`
- **THEN** the service serves streaming transcription from the parakeet
  recogniser, and does not load the sherpa recogniser

#### Scenario: Unknown engine value

- **WHEN** the streaming engine is configured as a value that is neither `sherpa`
  nor `parakeet`
- **THEN** the service fails to start
- **AND** the error names the setting and the rejected value

#### Scenario: Selected recogniser cannot be loaded

- **WHEN** the selected recogniser's runtime or model weights cannot be loaded
- **THEN** the service does not fall back to another recogniser
- **AND** the health endpoint does not report the service as ready

### Requirement: Streaming model weights are acquired at runtime

The service SHALL obtain streaming model weights from a configured location at
startup, into a path that survives an image rebuild, rather than carrying them
inside the image.

Weights SHALL be placed at their final path only once the download has completed
in full, so that an interrupted download cannot be mistaken for a usable model on
a later start.

The service SHALL reuse already-present weights without re-downloading them.

#### Scenario: First start with no weights present

- **WHEN** the service starts and the configured model path holds no weights
- **THEN** it downloads them before reporting itself ready

#### Scenario: Download is interrupted

- **WHEN** a weight download is interrupted before completing
- **AND** the service is started again
- **THEN** the partial download is not treated as a usable model
- **AND** the download is attempted again

#### Scenario: Weights already present

- **WHEN** the service starts and complete weights are already at the configured
  path
- **THEN** it does not download them again

### Requirement: Streaming transcription is served in the OpenAI realtime format

The service SHALL expose streaming transcription over a websocket that speaks
OpenAI's realtime transcription wire format, so that OpenAI-compatible clients
need no adaptation. This format SHALL NOT vary by which recogniser is selected.

The service SHALL accept audio as base64-encoded 24 kHz mono signed 16-bit
little-endian PCM.

#### Scenario: Client appends audio

- **WHEN** a client sends an audio append message carrying valid base64 PCM
- **THEN** the audio is transcribed and the client receives realtime-format
  events describing the utterance in progress

#### Scenario: Audio payload is not valid base64

- **WHEN** a client sends an audio append message whose payload is not valid
  base64
- **THEN** the service returns an error message on the same connection
- **AND** the connection stays open

#### Scenario: Engine change is invisible on the wire

- **WHEN** a deployment switches its configured streaming engine and restarts
- **THEN** clients observe the same message types and the same exchange, and
  require no change

### Requirement: Turn lifecycle

The service SHALL open a turn when the recogniser first produces text, and SHALL
identify every event belonging to that turn with a stable identifier that is
unique to it.

The service SHALL close a turn when the recogniser signals the utterance has
ended, emitting both a speech-stopped event and a completion event carrying the
final transcript.

Turn boundaries SHALL be decided by the recogniser, not by a timer in the
protocol layer.

#### Scenario: Speech begins

- **WHEN** the recogniser produces text for the first time in a turn
- **THEN** a speech-started event is emitted carrying a new turn identifier and
  the offset at which the speech began

#### Scenario: Recogniser signals end of utterance

- **WHEN** the recogniser signals that the utterance has ended
- **THEN** a speech-stopped event and a completion event carrying the full
  transcript are emitted for that turn
- **AND** the next speech opens a turn with a different identifier

#### Scenario: Silence with nothing recognised

- **WHEN** audio arrives that the recogniser resolves to no text
- **THEN** no turn is opened and no completion event is emitted

#### Scenario: Client commits explicitly

- **WHEN** a client sends an explicit commit
- **THEN** any utterance in progress is closed and its completion event is
  emitted, even though the recogniser had not yet signalled the end

#### Scenario: Client clears the buffer

- **WHEN** a client clears the audio buffer
- **THEN** decoding state is discarded and the next audio starts a fresh turn

### Requirement: Interim transcripts are emitted incrementally

The service SHALL emit interim transcript events containing only the new text
since the previous event for that turn, so that a client accumulating them
reconstructs exactly the transcript it is later handed.

Where a recogniser revises text it has already emitted, the service SHALL emit
nothing for that revision rather than emit a correction, and SHALL let the
turn's completion event carry the authoritative transcript.

The final portion of a turn's text SHALL be emitted as an interim event as well
as appearing in the completion event.

#### Scenario: Hypothesis is extended

- **WHEN** the recogniser extends its current hypothesis with new text
- **THEN** an interim event is emitted containing only the newly added text

#### Scenario: Hypothesis is revised rather than extended

- **WHEN** the recogniser replaces text it has already emitted for this turn
- **THEN** no interim event is emitted for that revision
- **AND** the turn's completion event carries the corrected transcript

#### Scenario: Accumulated interim events match the completion

- **WHEN** a client concatenates every interim event of a turn in order
- **AND** the recogniser only ever extended its hypothesis
- **THEN** the result equals the transcript in that turn's completion event

### Requirement: Session configuration is owned by the service

The service SHALL determine the recogniser, the transcription language, and the
turn-boundary behaviour from its own configuration, not from the client.

The service SHALL acknowledge a client's session configuration message so the
exchange remains well-formed, and SHALL disregard its contents.

#### Scenario: Client sends session configuration

- **WHEN** a client sends a session configuration message
- **THEN** the service acknowledges it
- **AND** the transcription language and recogniser in use are unchanged

### Requirement: Transcription language is applied to the streaming path

Where the selected recogniser accepts a language, the service SHALL supply the
language from its own configuration, and SHALL interpret language codes the same
way the batch transcription path does, so that a given code selects the same
language on both paths.

Where the configured language is one the selected recogniser cannot serve, the
service SHALL make that visible at startup rather than transcribing into an
unexpected language.

#### Scenario: Configured language reaches the recogniser

- **WHEN** the service is configured for a language
- **AND** the selected recogniser accepts a language
- **THEN** every utterance in every streaming session is transcribed as that
  language

#### Scenario: Same code, same language on both paths

- **WHEN** the same language code is used for batch and for streaming
- **THEN** both paths transcribe in the same language

### Requirement: Client sample rate is handled without introducing artefacts

Where the client's sample rate differs from the one the selected recogniser
requires, the service SHALL convert the audio, and the conversion SHALL be
continuous across the boundaries between received audio frames.

The service SHALL NOT convert audio a recogniser is able to accept directly.

#### Scenario: Recogniser requires a different sample rate

- **WHEN** audio arrives at a sample rate the selected recogniser does not accept
- **THEN** it is converted before decoding
- **AND** the conversion carries state between frames, so that no discontinuity
  is introduced at frame boundaries

#### Scenario: Recogniser accepts the client rate

- **WHEN** the selected recogniser accepts the client's sample rate directly
- **THEN** the audio is passed through without conversion

### Requirement: Transcript formatting reflects the recogniser

Where a recogniser produces punctuated and cased text, the service SHALL emit it
unaltered. Where a recogniser produces unpunctuated or uppercase text, the
service SHALL be configurable to normalise casing before emitting it, because
downstream consumers are trained on ordinary prose.

#### Scenario: Recogniser produces punctuated, cased text

- **WHEN** the selected recogniser emits punctuation and casing
- **THEN** transcripts are emitted with that punctuation and casing intact

#### Scenario: Recogniser produces uppercase, unpunctuated text

- **WHEN** the selected recogniser emits uppercase text
- **AND** case normalisation is enabled
- **THEN** the emitted transcript reads as ordinary prose rather than as
  uppercase

### Requirement: Streaming backend availability and failure

Where the streaming backend is not enabled, the service SHALL accept a websocket
connection, return an error naming the setting that would enable it, and close
the connection.

Where a session fails mid-connection, the service SHALL report the failure to the
client and close the connection, rather than leaving it open and silent.

A client disconnecting SHALL NOT be treated as a failure.

#### Scenario: Streaming backend disabled

- **WHEN** a client connects while the streaming backend is not enabled
- **THEN** it receives an error naming the setting that enables it
- **AND** the connection is closed

#### Scenario: Recogniser fails mid-session

- **WHEN** the recogniser fails while a session is in progress
- **THEN** the client receives an error message and the connection is closed

#### Scenario: Client disconnects

- **WHEN** a client disconnects
- **THEN** the session's resources are released and no error is reported

### Requirement: Authenticated streaming access

Where an API key is configured, the service SHALL require a matching bearer
credential on the websocket connection, SHALL reject a connection without one,
and SHALL close it. Where no key is configured, connections SHALL be accepted
without a credential.

#### Scenario: Valid credential

- **WHEN** an API key is configured and a client connects with a matching bearer
  credential
- **THEN** the session proceeds

#### Scenario: Missing or wrong credential

- **WHEN** an API key is configured and a client connects without a matching
  bearer credential
- **THEN** it receives an unauthorised error and the connection is closed

#### Scenario: No key configured

- **WHEN** no API key is configured
- **THEN** a client connecting without a credential is accepted
