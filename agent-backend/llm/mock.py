"""
Deterministic fake LLM so tests, CI and the demo run without an API key, network or cost.

You script what each caller receives, keyed by `purpose`:

    llm = MockLLM({
        "router":      {"intent": "offensive", "agents": ["offensive"]},   # same answer every call
        "synthesiser": [bad_draft, good_draft],                            # 1st call, then 2nd call
        "analyst":     lambda req: {...},                                  # computed from the request
    })

What a script value can be
  str / dict / list   returned as the reply (dict/list are JSON-encoded with sorted keys)
  [a, b, c]           a queue: call 1 gets a, call 2 gets b ... a further call is an error
  callable(req)       called with a MockRequest; returns any of the above
  Exception instance  raised instead of answering (simulate rate limits, timeouts)

How a purpose is matched
  exact key ("analyst:trend")  ->  prefix before ":" ("analyst")  ->  "*"  ->  `default`
  Nothing matched -> MockScriptError, so an unexpected call in a test fails loudly.

Determinism
  No randomness, no clock, no network. Token counts are an approximation (about 4 characters
  per token) computed from the text, and latency is simulated from the token counts, so
  the same script and the same messages always give identical responses, usage and latency.

Thread safety
  LangGraph runs parallel branches (the analysts) concurrently, so queue pops and the call
  log are protected by a lock. Give each parallel branch its own purpose key
  ("analyst:trend", "analyst:offensive") so their answers don't depend on scheduling order.
"""
from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .base import LLMResponse, Message, Usage

Responder = Any  # str | dict | list | Callable[[MockRequest], Any] | BaseException


class MockScriptError(AssertionError):
    """The test script did not cover a call (no matching purpose, or a queue ran out)."""


@dataclass(frozen=True)
class MockRequest:
    """What a callable responder receives."""
    purpose: str
    messages: tuple[Message, ...]
    json_mode: bool
    call_index: int          # 0-based count of earlier calls for this exact purpose

    @property
    def system(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def last_user(self) -> str:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), "")

    @property
    def text(self) -> str:
        return "\n".join(m.content for m in self.messages)


@dataclass(frozen=True)
class MockCall:
    """One entry in the call log, for assertions in tests."""
    purpose: str
    request: MockRequest
    response: LLMResponse | None
    error: BaseException | None


def count_tokens(text: str) -> int:
    """Deterministic approximation: about 4 characters per token, at least 1 for non-empty text."""
    return math.ceil(len(text) / 4) if text else 0


def _encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    raise TypeError(f"Mock responder returned unsupported type {type(value).__name__}")


class MockLLM:
    """Implements the LLMClient protocol from llm/base.py."""

    def __init__(self, script: dict[str, Responder] | None = None, *, default: Responder | None = None,
                 model: str = "mock-llm", base_latency_ms: float = 50.0, ms_per_output_token: float = 2.0):
        self.model = model
        self._script = dict(script or {})
        self._default = default
        self._queues: dict[str, list] = {}
        self._counts: dict[str, int] = {}
        self._base_latency = base_latency_ms
        self._ms_per_token = ms_per_output_token
        self._lock = threading.Lock()
        self.calls: list[MockCall] = []

    # ------------------------------------------------------------------ public
    def complete(self, messages: list[Message], *, purpose: str, json_mode: bool = False,
                 temperature: float = 0.0) -> LLMResponse:
        with self._lock:
            idx = self._counts.get(purpose, 0)
            self._counts[purpose] = idx + 1
            req = MockRequest(purpose, tuple(messages), json_mode, idx)
            try:
                responder = self._next_responder(purpose)
            except MockScriptError as e:
                self.calls.append(MockCall(purpose, req, None, e))
                raise

        value = responder(req) if callable(responder) and not isinstance(responder, BaseException) else responder
        if isinstance(value, BaseException):
            with self._lock:
                self.calls.append(MockCall(purpose, req, None, value))
            raise value

        content = _encode(value)
        prompt_tokens = sum(count_tokens(m.content) + 4 for m in messages)  # +4 per-message overhead
        completion_tokens = count_tokens(content)
        resp = LLMResponse(content=content, model=self.model, purpose=purpose,
                           usage=Usage(prompt_tokens, completion_tokens),
                           latency_ms=self._base_latency + self._ms_per_token * completion_tokens)
        with self._lock:
            self.calls.append(MockCall(purpose, req, resp, None))
        return resp

    def calls_for(self, purpose: str) -> list[MockCall]:
        """Calls whose purpose equals `purpose` or starts with `purpose:`."""
        return [c for c in self.calls if c.purpose == purpose or c.purpose.startswith(purpose + ":")]

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for c in self.calls:
            if c.response is not None:
                total = total + c.response.usage
        return total

    def reset(self) -> None:
        """Forget calls and rewind every queue (the script itself is kept)."""
        with self._lock:
            self._queues.clear()
            self._counts.clear()
            self.calls.clear()

    # ----------------------------------------------------------------- private
    def _lookup(self, purpose: str) -> tuple[str, Responder]:
        for key in (purpose, purpose.split(":", 1)[0], "*"):
            if key in self._script:
                return key, self._script[key]
        if self._default is not None:
            return "<default>", self._default
        raise MockScriptError(f"MockLLM has no response scripted for purpose '{purpose}'. "
                              f"Scripted keys: {sorted(self._script)}")

    def _next_responder(self, purpose: str) -> Responder:
        key, entry = self._lookup(purpose)
        if not isinstance(entry, list) or key == "<default>":
            return entry
        queue = self._queues.setdefault(key, list(entry))
        if not queue:
            raise MockScriptError(f"MockLLM queue for '{key}' is exhausted "
                                  f"({len(entry)} response(s) scripted, purpose '{purpose}').")
        return queue.pop(0)


# ---------------------------------------------------------------------------
# Helpers for common scripting patterns
# ---------------------------------------------------------------------------
def keyword_responder(rules: Iterable[tuple[str | tuple[str, ...], Responder]],
                      default: Responder | None = None) -> Callable[[MockRequest], Any]:
    """
    Pick a response from keywords in the last user message (case-insensitive).
    A keyword matches at the START of a word, so "card" matches "cards" but "red" does not
    match "scored". A rule matches if ANY of its keywords match. First matching rule wins.

        keyword_responder([(("shot", "goal"), {"intent": "offensive"}),
                           (("card", "foul"), {"intent": "discipline"})],
                          default={"intent": "general_statistics"})
    """
    compiled = []
    for r, resp in rules:
        keywords = (r,) if isinstance(r, str) else tuple(r)
        pattern = re.compile("|".join(r"\b" + re.escape(k) for k in keywords), re.IGNORECASE)
        compiled.append((pattern, resp))

    def respond(req: MockRequest):
        for pattern, resp in compiled:
            if pattern.search(req.last_user):
                return resp(req) if callable(resp) else resp
        if default is None:
            raise MockScriptError(f"keyword_responder: no rule matched {req.last_user!r}")
        return default(req) if callable(default) else default

    return respond


def when_prompt_contains(marker: str, then: Responder, otherwise: Responder) -> Callable[[MockRequest], Any]:
    """
    Answer `then` if `marker` appears anywhere in the prompt, else `otherwise`.
    Used to simulate a model that fixes its answer once the guard's feedback is in the prompt.
    """
    def respond(req: MockRequest):
        chosen = then if marker in req.text else otherwise
        return chosen(req) if callable(chosen) else chosen
    return respond