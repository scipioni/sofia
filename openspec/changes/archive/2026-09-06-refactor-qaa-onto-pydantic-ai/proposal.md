# Refactor qaa-agent onto Pydantic AI

## Why

The heart of qaa-agent is ~130 lines of hand-rolled agent loop — SSE tool-call
fragment folding, ours-vs-theirs tool dispatch, tool-round caps, error-as-text
recovery. It works and it is tested, but it is the subtlest code in the project,
and the problem it solves is now a solved problem upstream: Pydantic AI provides
all of it as a maintained, standard library (pydantic lineage, v2-era API), plus
capabilities we currently have no path to (structured output, MCP toolsets,
multi-provider upstream, run cancellation). Adopting it makes the library the
platform the brain stands on, instead of glue we maintain by hand.

The constraint that makes this safe: the refactor happens *inside* the service,
behind the existing OpenAI Chat Completions facade. s2s cannot tell the
difference, and neither can `curl`.

## What Changes

- Replace the hand-rolled upstream loop in `qaa/engine.py` with a Pydantic AI
  `Agent` (`OpenAIChatModel` pointed at the same upstream), kept behind the
  existing `stream()` facade (`TextDelta | ToolCallsDelta | Done`) — unchanged
  public surface.
- Replace the hand-written tool registry in `qaa/tools.py` with a Pydantic AI
  `FunctionToolset`: typed async functions, JSON schema generated from
  signatures/docstrings, `ModelRetry` semantics replacing the catch-all
  error-as-text wrapper, per-tool timeout preserved.
- Caller-declared tools (LiveKit `function_tool`s from s2s) become an
  `ExternalToolset` built per request from `req.tools` — schema-only, never
  executed in-process, returned as `DeferredToolRequests` mapped back to the
  existing `ToolCallsDelta` + `finish_reason="tool_calls"` handback.
- New adapters (the only lasting hand-written glue): OpenAI message history →
  `ModelMessage`, and per-request settings (temperature, top_p, min-capped
  max_tokens, stop) → model settings.
- `max_tool_rounds` becomes `UsageLimits(requests=...)`; over-limit still
  speaks the "couldn't work that one out" fallback.
- Add `pydantic-ai-slim[openai]` to the `qaa` install extra (slim to keep the
  image lean); pin major.
- Pin the framework's behavioral edges with tests: text-then-tool-call in one
  response (requires `end_strategy='graceful'`), verbatim passthrough of
  caller-tool arguments (the framework validates JSON-object-ness only), mixed
  ours+theirs precedence, and the dropped `tool_choice` forcing forms the
  framework forbids on handback-capable runs.
- Unchanged: the OpenAI wire protocol surface (`app.py` except three lifespan
  wiring lines: registry → toolset builder), `schemas.py`, the system-prompt
  stacking and `sofia_*` session context, the never-silent
  `UPSTREAM_ERROR_REPLY` failure mode, s2s, docker topology.

## Capabilities

### New Capabilities

- `qaa-brain`: The external contract of the qaa-agent service that this
  refactor must preserve — OpenAI Chat Completions surface, instant streaming
  text deltas, inline invisible server-side tools, unexecuted handback of
  caller tools, tool-round cap, never-silent failure, per-request sampling
  settings with a non-raisable max-token ceiling.

### Modified Capabilities

(none — no existing spec covers the qaa brain; the wire behavior itself is
preserved, which is exactly what the new spec codifies)

## Impact

- **Code**: `src/sofia_galileo/qaa/engine.py` (rewritten internals), `tools.py`
  (registry → toolset adapter); `app.py` and `schemas.py` untouched.
- **Dependencies**: `pydantic-ai-slim[openai]` added to the `qaa` extra in
  `pyproject.toml` (pulls pydantic-graph; `openai` and `httpx` already present).
- **Tests**: `tests/test_engine.py` keeps its HTTP StubUpstream seam
  (`OpenAIChatModel` accepts a custom `httpx.AsyncClient`); new tests for the
  three pinned semantic diffs; `tests/test_tools.py` and `test_qaa_api.py`
  updated to the toolset adapter.
- **Systems**: qaa container only; no s2s/stt/tts changes, no compose changes,
  no env-var changes (all existing `SOFIA_QAA_*` settings keep their meaning).
- **Risk accepted**: Pydantic AI moves fast (v2-era churn in current docs) —
  majors pinned, and the facade keeps that churn qaa-internal.
