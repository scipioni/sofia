## 1. Foundation and spike

- [x] 1.1 Add `pydantic-ai-slim[openai]` (pinned major) to the `qaa` extra in `pyproject.toml`; `uv lock && uv sync`; verify `uv run python -c "import pydantic_ai"` works and `task check` stays green
- [x] 1.2 Spike against the pinned version: build `OpenAIChatModel` + `Agent` over the `StubUpstream` ASGI transport and verify which of `stop` / `tool_choice` / `parallel_tool_calls` reach the upstream request body; record the outcome in design.md (resolves open question 1); anything unsupported gets an explicit logged-and-dropped warning path — verify with a unit test

## 2. Adapters (the lasting glue)

- [x] 2.1 Message adapter: OpenAI request history → framework message parts (persona prompt stacking, caller system message kept, session context from `sofia_*` fields, assistant tool-call messages, tool-result messages, multimodal content-list passthrough); verify with table-driven unit tests covering every OpenAI message shape in `schemas.py`
- [x] 2.2 Default tools as a framework toolset: typed async functions with docstrings replacing hand-written schema dicts; per-tool `asyncio.timeout` + error-as-text helper preserving the "always returns a string, never raises" contract; verify the generated JSON schema for `get_current_time` matches the current spec's shape and that timeout/error paths return error text (port the `tests/test_tools.py` cases)
- [x] 2.3 Caller-tool adapter: build an external toolset from `req.tools` wire entries (name, description, parameters schema off the wire, never executed in-process); verify with unit tests that a declared tool is advertised upstream and produces a deferred request when called
- [x] 2.4 Handback mapper: deferred tool-call output → OpenAI-format tool-call dicts (`id`, `type`, `function.name`, `function.arguments` as a valid JSON object); verify with unit tests

## 3. Engine swap behind the facade

- [x] 3.1 Rebuild `QaaEngine.stream()` internals on the framework's run/stream API with `end_strategy='graceful'`: text-part deltas → `TextDelta`, deferred calls → `ToolCallsDelta` + `Done("tool_calls")`, normal end → `Done("stop", usage)`; keep the existing event dataclasses so `app.py` is untouched; verify `tests/test_engine.py` plain-text and streaming tests pass unchanged
- [x] 3.2 Round cap via usage/request limits set from `SOFIA_QAA_MAX_TOOL_ROUNDS`; over-limit speaks the existing "couldn't work that one out" fallback; verify the round-limit test asserts the same cap-plus-one upstream-call count
- [x] 3.3 Upstream-failure path: apology spoken only when nothing has streamed yet, clean cut-off otherwise (keep `UPSTREAM_ERROR_REPLY` and its guard); verify the two failure tests pass unchanged
- [x] 3.4 Per-request settings mapping (temperature/top_p/stop with configured fallbacks, `min(requested, ceiling)` max_tokens, tool_choice per spike outcome); verify the ceiling and sampling tests assert on the upstream request body
- [x] 3.5 `complete()` continues to collect the stream; verify the parametrized stream/non-stream parity test passes

## 4. Behavior pins (acceptance for the spec)

- [x] 4.1 Text-then-tool-call in one upstream response: text is streamed to the caller AND the tool runs — new test, no coverage exists today in either implementation (spec: Server-side tools / Instant text deltas)
- [x] 4.2 Handback arguments are a valid JSON object for the caller-declared schema even when the model's raw string was oddly formatted (spec: Caller tools handed back unexecuted) — new test
- [x] 4.3 Mixed ours+theirs in one response: decide the precedence outcome, record it in design.md (resolves open question 2), pin it with a test asserting caller tools come back unexecuted with `finish_reason="tool_calls"`
- [x] 4.4 `tests/test_qaa_api.py` green unchanged — the HTTP surface (SSE framing, role-only opening chunk, `[DONE]`, error shapes) is untouched

## 5. Cleanup and integration

- [x] 5.1 Remove dead machinery (fragment folding, tool-call accumulation, manual loop, the old `ToolRegistry` if fully replaced); rewrite the `tools.py` module docstring whose "an agent framework would add indirection without buying anything" rationale this change deliberately reverses; refresh AGENTS.md qaa bullets and `.env.example` if any wording changed
- [x] 5.2 `task lint && task test` green with no skips
- [x] 5.3 Live smoke against a real upstream: `task ask Q="che ore sono?"` exercises a full tool round over HTTP; `task console` exercises the voice loop end to end; note time-to-first-audio vs the `docs/benchmark.md` baseline to confirm the delta path gained no latency
