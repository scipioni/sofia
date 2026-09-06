# Design — qaa-agent on Pydantic AI

## Context

qaa-agent is a stateless FastAPI service that *serves* OpenAI Chat Completions
while *consuming* Chat Completions from an upstream LLM (see proposal.md —
Why). Today the consumption side is hand-rolled in `qaa/engine.py`: streamed
SSE parsing, tool-call fragment folding by index, ours-vs-theirs dispatch, a
bounded tool loop, and an error-as-text recovery path. The repo conventions
that constrain the design:

- Every internal interface speaks the OpenAI protocol; `s2s` drives this
  service with `livekit.plugins.openai.LLM(base_url=...)` and must keep doing
  so unmodified.
- Wire schemas are `extra="allow"`; `sofia_room` / `sofia_participant` /
  `sofia_language` ride in `extra_body`.
- The engine streams text deltas the instant they arrive; time-to-first-audio
  is the metric that matters downstream.
- Every service owns a `config.py` (`SOFIA_QAA_*` env vars, 1:1 fields).
- Tests drive the engine against a stub upstream over real HTTP
  (`tests/test_engine.py`), not mocks.

## Goals / Non-Goals

**Goals:**

- The tool loop, fragment folding, round caps, retry semantics, and streaming
  plumbing are owned and maintained by Pydantic AI, not by us.
- The refactor is behavior-preserving at the service boundary; the new
  `qaa-brain` spec codifies exactly what must not drift.
- New capabilities become reachable as config/framework features rather than
  new hand-rolled machinery: MCP toolsets, structured output, non-OpenAI
  upstream providers, run cancellation (the hook for future barge-in).
- Dependency churn is contained inside the qaa container.

**Non-Goals:**

- No changes to `app.py`, `schemas.py`, the SSE rendering, or any env var.
- No adoption of Pydantic AI's realtime (speech-to-speech) module — Sofia's
  architecture is deliberately modular local STT → LLM → TTS; the realtime
  module would replace that design, not this service.
- No agent framework inside `s2s` — LiveKit Agents already owns the turn loop
  there; two frameworks would fight over it. Pydantic AI lives only inside
  qaa-agent.
- No persistence, sessions, or history storage — qaa stays stateless; LiveKit
  resends full history every turn.

## Decisions

### D1. Facade-preserving internal swap

`QaaEngine.stream(req)` keeps yielding `TextDelta | ToolCallsDelta | Done`;
`app.py` and `schemas.py` are untouched. Pydantic AI sits strictly behind the
facade.

*Why*: the OpenAI facade is the load-bearing decision of the whole system —
it makes every component swappable and curl-drivable. The library must stay an
implementation detail of one container.

*Alternative rejected*: replacing the s2s↔qaa hop with Pydantic AI UI adapters
(AG-UI etc.) — forfeits the protocol uniformity the repo exists on.

### D2. One Agent at lifespan; our tools on it, caller tools per run

A single `Agent` is constructed in `app.py`'s lifespan: `instructions` from
`QaaSettings.system_prompt`, model `OpenAIChatModel(base_url, api_key,
http_client, ...)` built from the existing settings fields. Our tools
(current `build_default_registry()`) register once as a `FunctionToolset` —
typed async functions, JSON schema generated from signatures and docstrings,
no more hand-written schema dicts. Caller-declared tools are *not* on the
Agent: per request, an `ExternalToolset` is built from `req.tools` (name,
description, parameters JSON schema straight off the wire) and passed via the
run's `toolsets=`.

*Why per-run toolsets*: s2s declares a different tool set per agent/job;
tool identity is per-request state. Pydantic AI supports per-run toolsets
precisely for this.

*Alternative rejected*: keeping the hand-written `ToolRegistry` and adapting
it to Pydantic AI tools — preserves dead abstraction; the registry exists only
to feed the loop the framework now owns.

### D3. Per-tool timeout survives, hand-written inside the tool

Pydantic AI has no built-in per-tool timeout. Each tool body wraps its work in
`asyncio.timeout(settings-derived)` and returns the error string on failure —
the current "never raise, report to the model as text" contract, expressed
once in a small shared helper instead of in the registry dispatch.

### D4. Drive runs with `agent.iter()`, not `run_stream()`

`run_stream()` treats the *first* output as final: if the model streams text
and then calls tools, the default `end_strategy` ends the run and the tool
calls never execute — a silent behavior regression against today's engine
(which speaks the text *and* runs the tools). We drive `agent.iter()` and
translate events:

```
PartDeltaEvent(TextPartDelta)      -> TextDelta(text)
run ends, output is str            -> Done(finish_reason="stop", usage)
run ends, output is DeferredToolRequests
                                   -> ToolCallsDelta(to OpenAI dicts)
                                   -> Done(finish_reason="tool_calls", usage)
UsageLimitExceeded                 -> fallback text if nothing streamed
Provider errors (OpenAIError etc.) -> UPSTREAM_ERROR_REPLY if nothing streamed
```

The "only if nothing streamed" guard matches engine.py's current rule: never
stitch an apology onto a half-spoken sentence.

### D5. `end_strategy='graceful'` for text-then-tool parity

Set explicitly so a response that mixes text and tool calls keeps today's
behavior: text is streamed (and therefore spoken) and the tools still run.
Pinned by a new test — today no test covers this case in either
implementation.

### D6. Round cap becomes `UsageLimits(requests=max_tool_rounds + 1)`

Declarative, enforced by the framework. The exhaustion fallback ("Sorry, I
couldn't work that one out.") moves to the `UsageLimitExceeded` handler.

### D7. Message adapter: OpenAI history → `ModelMessage` (the lasting glue)

Convert the caller's `list[ChatMessage]` to Pydantic AI `ModelMessage` parts:
system → instructions context, user/assistant text → text parts, assistant
`tool_calls` → `ToolCallPart`s, tool results → `ToolReturnPart`s, multimodal
content lists passed through as content parts. This adapter is where protocol
fidelity lives after the refactor; it is table-driven and exhaustively tested.
The reverse mapping exists only for `DeferredToolRequests`: `ToolCallPart` →
`{"id", "type": "function", "function": {"name", "arguments"}}` for the
existing `ToolCallsDelta` path.

Session context (`sofia_language`, `sofia_participant`) stays a computed
system-context message inserted after the persona prompt, as today.

### D8. Sampling settings per request; max-token ceiling preserved

Per run: `temperature` / `top_p` / `stop` from the request with settings
fallbacks; `max_tokens = min(requested, settings.max_tokens)` — the caller may
lower, never raise. `tool_choice` forwarded when the caller supplies it.

### D9. `pydantic-ai-slim[openai]`, pinned to major

Slim keeps the qaa image lean (`openai` and `httpx` are already dependencies);
the `openai` extra provides `OpenAIChatModel`. Majors pinned because the
library moves fast; the facade makes that churn a qaa-internal affair.

### D10. Tests keep the HTTP stub seam

`OpenAIChatModel` accepts a custom `httpx.AsyncClient`, so `StubUpstream`
keeps serving scripted SSE over ASGITransport and the existing assertions on
upstream request bodies still assert the real wire. (Pydantic AI's
`FunctionModel` would be a cleaner in-process seam, but it stops asserting
what actually goes over the wire — not the trade this suite is built on.)

## Risks / Trade-offs

- [API churn across Pydantic AI versions] → pin major; facade isolates churn;
  adapter code is the only upgrade surface and it is test-covered.
- [`stop` / `tool_choice` / `parallel_tool_calls` support in
  `OpenAIModelSettings` unverified] → spike as the first implementation task;
  anything unsupported is logged-and-dropped with a warning rather than
  silently lost.
- [External-tool arguments are validated and re-serialized on handback, so
  the arguments string may differ cosmetically from what the model emitted;
  schema-violating args trigger a model retry instead of a raw handback] →
  **Resolved by observation (2.40.0)**: the external-tool validator checks
  JSON-object-ness only, and the raw arguments string passes through verbatim
  — schema-violating args reach the caller as emitted, exactly like the
  hand-rolled engine. No diff exists; pinned by two tests.
- [Mixed ours+theirs calls in one response: today ours are silently dropped
  when theirs are present; framework precedence may differ] → **Resolved**:
  the caller-visible contract is unchanged — the run ends handing theirs back
  unexecuted with `finish_reason="tool_calls"` (pinned by test). Whether our
  own tool in the same response also executed is framework-internal and not
  caller-observable.
- [`tool_choice='required'` (and structured/list forms)] → **Resolved**: the
  framework *forbids* forcing tool calls on a run that may end in a handback
  (raises before any request). Engine policy: `auto`/`none` pass through;
  anything else is logged-and-dropped and the framework resolves its own
  default (`auto`). Pinned by tests.
- [Unknown tool names (neither ours nor caller's)] → framework reports the
  miss to the model as a retry prompt rather than today's log-and-stop; same
  observable outcome for the caller (no tool call surfaces). Accepted.
- [Event-translation overhead on the delta path] → per-event cost is µs
  against per-token costs of ms; confirm time-to-first-audio unchanged via the
  existing `task ask` / benchmark flow after implementation.
- [Concurrent runs share one Agent] → the framework supports concurrent runs
  of a single Agent (state lives in the run, not the Agent); the per-run
  toolsets and settings are run-scoped by construction.

## Migration Plan

1. Spike D8 settings support against the pinned Pydantic AI version.
2. Land the adapter + toolset code with the facade in place; flip the engine
   internals; keep `StubUpstream` green.
3. Add the three pin tests (text-then-tool, handback re-serialization,
   ours+theirs precedence) — these define acceptance.
4. `task check`; then `task ask Q="che ore sono?"` against a live upstream and
   `task console` for the full loop.
5. Rebuild only the qaa image (`docker compose build qaa-agent`); no other
   service is touched. Rollback = revert the commit and rebuild; the service
   is stateless and the wire contract unchanged, so rollback carries no data
   or protocol migration.

## Open Questions

- ~~Exact `OpenAIModelSettings` field support~~ **Resolved by spike (pydantic-ai 2.40.0)**: base
  `ModelSettings` natively carries `temperature`, `top_p`, `max_tokens`,
  `stop_sequences` (rendered as OpenAI `stop` on the wire), `tool_choice`, and
  `parallel_tool_calls`. One wire nuance found and pinned: pydantic-ai
  defaults to rendering the token ceiling as `max_completion_tokens`; Sofia's
  upstreams are lowest-common-denominator OpenAI servers, so the model profile
  pins the legacy `max_tokens` mapping (`upstream_model`). For `tool_choice`,
  see the Risks resolution below — scalars `auto`/`none` pass through, forced
  and structured forms are logged-and-dropped.
