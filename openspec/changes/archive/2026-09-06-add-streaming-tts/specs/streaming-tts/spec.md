## Purpose

Defines how the `tts` service delivers synthesised speech over
`POST /v1/audio/speech`: which response formats are delivered incrementally as
audio becomes available versus as one complete body, and what a caller
observes in either case.

## ADDED Requirements

### Requirement: Incremental delivery for raw PCM

Where the requested response format carries no in-band total-length or
container framing (raw PCM), the service SHALL begin sending synthesised
audio before the entire input has been synthesised, delivering each portion
as soon as it is ready.

#### Scenario: Multi-segment input streamed as PCM

- **WHEN** a client requests `response_format=pcm` for input that synthesises
  as more than one internal segment
- **THEN** the client receives the audio for the first segment before the
  service has finished synthesising later segments

#### Scenario: Concatenated stream equals the full synthesis

- **WHEN** a client requests `response_format=pcm`
- **THEN** concatenating everything received, in the order received, yields
  exactly the same audio samples as synthesising the same input in one block

### Requirement: Complete-body delivery for self-describing formats

Where the requested response format requires declaring its total data length
or other container framing before the data (`wav`, `flac`), the service SHALL
deliver a single complete, correctly-formed body, synthesising the entire
input before sending any of it.

#### Scenario: WAV request is not chunked early

- **WHEN** a client requests `response_format=wav`
- **THEN** the response is a single, complete, valid WAV file
- **AND** no partial or malformed body is ever observable, regardless of
  input length

#### Scenario: Output is unchanged by this capability

- **WHEN** the same input and voice are requested as `wav`
- **THEN** the resulting audio is identical to what the same request produced
  before this capability existed

### Requirement: Format selection is independent of delivery mechanism

The service SHALL accept a request for any supported format through the same
endpoint, and SHALL NOT require a caller to know or care whether its chosen
format is delivered incrementally or as one body to receive a correct
response.

#### Scenario: Caller ignores chunking entirely

- **WHEN** a client reads the full response body before doing anything with
  it, regardless of requested format
- **THEN** it receives the complete, correct audio either way

### Requirement: Existing input and format validation is preserved

The service SHALL continue to reject an unsupported response format and to
return an empty, correctly-typed body for empty input, unchanged by how
delivery works internally.

#### Scenario: Unsupported format

- **WHEN** a client requests a response format the service does not support
- **THEN** it receives an error naming the supported formats, before any
  synthesis is attempted

#### Scenario: Empty input

- **WHEN** a client requests synthesis of empty or whitespace-only text
- **THEN** it receives an empty body of the requested format's media type,
  and no synthesis is attempted

### Requirement: A disconnected client does not affect the service

Where a client disconnects or stops reading before a response completes, the
service SHALL release any resources associated with that request and SHALL
continue serving other requests normally.

#### Scenario: Client disconnects mid-stream

- **WHEN** a client requesting `response_format=pcm` disconnects after
  receiving only some segments
- **THEN** the service stops synthesising further segments for that request
- **AND** other requests are served without degradation
