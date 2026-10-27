"""
The LLM interface every agent node talks to.

Both the mock (llm/mock.py, now) and the Groq client (llm/client.py, later) implement
`LLMClient.complete()`, so agent code never knows or cares which one it is using.

`purpose` names the caller ("router", "analyst:trend", "synthesiser", ...). The mock
uses it to decide what to answer; later the tracker uses it to report tokens and latency
per graph node.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    purpose: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def json(self) -> dict | list:
        """Parse the reply as JSON (agents ask for JSON). Raises LLMOutputError if it isn't."""
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, TypeError) as e:
            raise LLMOutputError(f"{self.purpose}: model did not return valid JSON: {e}",
                                 content=self.content) from e


# ---------------------------------------------------------------------------
# Errors (the real client will map Groq/HTTP errors onto these)
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for every LLM failure."""


class LLMRateLimitError(LLMError):
    """Provider rate limit hit (HTTP 429). Retryable."""


class LLMTimeoutError(LLMError):
    """Request timed out. Retryable."""


class LLMOutputError(LLMError):
    """The model answered, but not in the expected format (e.g. invalid JSON)."""

    def __init__(self, message: str, content: str | None = None):
        super().__init__(message)
        self.content = content


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message], *, purpose: str, json_mode: bool = False,
                 temperature: float = 0.0) -> LLMResponse:
        ...