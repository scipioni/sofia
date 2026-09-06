## Purpose

Defines the external contract of the qaa-agent service — the OpenAI-compatible
reasoning brain behind the Sofia voice agent. s2s drives it with a stock
OpenAI client, so this contract is what makes the brain swappable with any
upstream LLM and exercisable with plain `curl`; any implementation of this
service MUST preserve it.

## ADDED Requirements

### Requirement: OpenAI Chat Completions surface

The service SHALL serve `POST /v1/chat/completions` speaking the OpenAI Chat
Completions protocol, in both streaming (SSE chunks ending with `[DONE]`) and
non-streaming (a single completion object) modes, and SHALL serve `GET
/v1/models` advertising a configured model name. An empty `messages` list
SHALL be rejected with an OpenAI-shaped 400 error. A stock OpenAI client
SHALL be able to drive the service with no adaptation.

#### Scenario: Non-streaming completion

- **WHEN** a client POSTs a completion request with `stream: false` and one user message
- **THEN** the response is a single OpenAI-shaped completion object whose message contains the assistant text and a `stop` finish reason

#### Scenario: Streaming completion

- **WHEN** a client POSTs a completion request with `stream: true`
- **THEN** the response is an SSE stream of OpenAI-shaped chunks, opening with a role-only delta and closing with a `finish_reason` chunk followed by `data: [DONE]`

#### Scenario: Empty messages rejected

- **WHEN** a client POSTs a completion request with `messages: []`
- **THEN** the service responds 400 with an OpenAI-shaped error body

### Requirement: Instant text deltas

While streaming, assistant text SHALL be emitted to the caller as soon as it
arrives from upstream, in arrival order — the caller MUST NOT have to wait for
the full reply before receiving its first text delta.

#### Scenario: Upstream text streams through incrementally

- **WHEN** the upstream streams the fragments "Hi " then "there."
- **THEN** the caller receives two content deltas, "Hi " then "there.", in that order, before the stream ends

### Requirement: Persona prompt stacked with caller system message

The service SHALL prepend its own persona system prompt to every conversation
and SHALL preserve the caller's own system message alongside it — neither may
silently replace the other.

#### Scenario: Caller system message kept

- **WHEN** a request arrives whose messages include a caller-authored system message
- **THEN** the upstream receives both the service's persona prompt and the caller's system message

#### Scenario: Session context injected

- **WHEN** a request carries Sofia session context (spoken language, participant identity)
- **THEN** the upstream receives a system message conveying that context, and the context fields are not forwarded as user-visible content

### Requirement: Server-side tools run inline and invisibly

Tools owned by the service (e.g. the default time tool) SHALL be executed
inside the service: when the model calls one, the service runs it, feeds the
result back upstream, and continues toward a final answer. The caller MUST
NOT see the internal tool call — only the resulting text. The upstream
follow-up request SHALL carry the assistant tool-call message and the tool
result message. A tool call split across streamed fragments SHALL be
reassembled correctly.

#### Scenario: Tool call executed and answer continued

- **WHEN** the upstream's first response is a tool call for a service-owned tool, streamed as fragments across chunks, and its second response is the final text
- **THEN** the caller receives only the final text, and the second upstream request contains the complete assistant tool-call and the tool result

### Requirement: Caller tools advertised and handed back unexecuted

Tools declared by the caller in the request SHALL be advertised upstream
alongside the service's own tools. When the model calls a caller-declared
tool, the service SHALL return that tool call to the caller unexecuted (with a
`tool_calls` finish reason) and SHALL NOT run it, execute any other pending
work, or make further upstream requests for that turn.

#### Scenario: Caller tool handed back

- **WHEN** the caller declares a tool and the upstream responds by calling it
- **THEN** the caller receives the tool call (name and the arguments exactly as the model emitted them) with a `tool_calls` finish reason, and the upstream received exactly one request for that turn

#### Scenario: Caller tools forwarded alongside service tools

- **WHEN** the caller declares tools in the request
- **THEN** the upstream request advertises both the caller's tools and the service's own tools

### Requirement: Bounded tool loop

The number of upstream request rounds per turn SHALL be capped at a configured
maximum. If the cap is exhausted without a final answer, the service SHALL
speak a short fallback text rather than looping or going silent.

#### Scenario: Model stuck calling the same tool

- **WHEN** the upstream calls a service-owned tool on every round
- **THEN** the upstream is called at most cap-plus-one times and the turn ends with a spoken fallback and a normal finish reason

### Requirement: Never-silent failure

If the upstream is unreachable or fails before any text has been streamed to
the caller, the service SHALL speak a short apology so the caller hears
something. If a partial answer was already streamed, the service SHALL end the
turn cleanly without appending an apology to the half-spoken sentence.

#### Scenario: Upstream down before any text

- **WHEN** the upstream connection fails before any content delta was emitted
- **THEN** the caller receives a non-empty apology text and a clean end of stream

#### Scenario: Upstream fails mid-answer

- **WHEN** the upstream fails after some text was already streamed
- **THEN** the stream ends cleanly with no apology appended to the partial text

### Requirement: Per-request sampling settings with a token ceiling

Per-request `temperature`, `top_p`, and `stop` SHALL be honored, falling back
to the service's configured defaults when absent. The requested `max_tokens`
MAY lower the effective limit but MUST NOT raise it above the service's
configured ceiling.

#### Scenario: Caller tries to raise the token ceiling

- **WHEN** a request asks for more max tokens than the service's configured ceiling
- **THEN** the upstream request carries the ceiling value, not the requested one

#### Scenario: Caller supplies sampling settings

- **WHEN** a request supplies temperature, top_p, or stop values
- **THEN** the upstream request carries those values

### Requirement: Multimodal content passthrough

Message content supplied as a structured part list (e.g. text-plus-image
parts) SHALL be passed through to upstream unmodified, so a vision-capable
upstream remains usable through the service.

#### Scenario: Multimodal message forwarded intact

- **WHEN** a request contains a message whose content is a list of typed parts
- **THEN** the upstream receives that content list unchanged
