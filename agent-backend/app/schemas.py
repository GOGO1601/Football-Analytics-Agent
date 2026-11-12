"""
Request and response models for the HTTP API, and the one function that turns the graph's
final AgentState into a response. Nothing here computes a statistic: every value is copied
from the state, and a state that breaks the graph's contract is refused (IncompleteResult),
never patched up.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.state import MAX_QUESTION_CHARS, OUTCOMES, usage_summary

Outcome = Literal["verified", "no_evidence", "fallback", "unverified"]
# outcome -> what grounding_result.passed must be (see agents/graph.py finalise)
EXPECTED_PASSED = {"verified": True, "no_evidence": True, "fallback": True, "unverified": False}
# error kinds whose messages may carry exception text: only the exception class is shown
_OPAQUE_KINDS = {"node_exception", "llm_error", "guard_error"}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"question": "Do teams that take more shots get more points?"}]})

    question: str = Field(description=f"A football statistics question, 1 to {MAX_QUESTION_CHARS} characters "
                                      "after whitespace is collapsed.")

    @field_validator("question", mode="before")
    @classmethod
    def _strict_str(cls, v: Any) -> Any:
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        return v

    @field_validator("question")
    @classmethod
    def _normalise(cls, v: str) -> str:
        q = " ".join(v.split())               # same normalisation as agents.state.new_state
        if not q:
            raise ValueError("question must not be empty or whitespace only")
        if len(q) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
        return q


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class KeyStat(BaseModel):
    label: str
    value: str
    source: str = Field(description="call_id of the tool result this value comes from")


class ToolError(BaseModel):
    code: str
    message: str
    suggestions: list[Any] = []


class ToolResultOut(BaseModel):
    call_id: str
    analyst: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = Field(None, description="the tool's own output (successful calls only)")
    error: ToolError | None = None


class GroundingIssue(BaseModel):
    kind: str
    field: str
    claim: str
    detail: str


class Grounding(BaseModel):
    passed: bool = Field(description="did the RETURNED answer pass the grounding guard")
    checked: int
    unsupported: list[str]
    issues: list[GroundingIssue]


class ErrorRecord(BaseModel):
    node: str
    kind: str
    message: str


class NodeTraceOut(BaseModel):
    node: str
    status: Literal["ok", "error"]
    ms: float


class Usage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_latency_ms: float = 0.0
    models: list[str] = []


class AskResponse(BaseModel):
    request_id: str
    question: str
    outcome: Outcome = Field(description="verified | no_evidence | fallback | unverified")
    verified: bool = Field(description="true only when outcome is 'verified'")
    answer: str
    evidence: list[str]
    key_stats: list[KeyStat]
    explanation: str
    grounding: Grounding
    retries: int
    intent: str | None
    analysts: list[str]
    tool_results: list[ToolResultOut]
    chart_actions: list[dict[str, Any]]
    warnings: list[str] = Field(description="data warnings from successful tools, e.g. incomplete seasons")
    errors: list[ErrorRecord] = Field(description="everything that went wrong along the way, in order")
    usage: Usage
    trace: list[NodeTraceOut]
    duration_ms: float


class Problem(BaseModel):
    """RFC 9457 problem details, used for every error response."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# State -> response
# ---------------------------------------------------------------------------
class IncompleteResult(Exception):
    """The graph returned a state that does not satisfy its own output contract."""


_SECRETISH = re.compile(r"(?i)(api[_-]?key|token|secret|password)\S*|[A-Za-z]:\\\S+|/(?:home|Users|root)/\S+")


def _public_message(kind: str, message: str) -> str:
    if kind in _OPAQUE_KINDS:
        return message.split(":", 1)[0].strip()[:80] or kind     # exception class name only
    return _SECRETISH.sub("[redacted]", message)[:300]


def to_response(state: dict, *, request_id: str, duration_ms: float) -> AskResponse:
    draft = state.get("draft_answer")
    outcome = state.get("outcome")
    g = state.get("grounding_result")
    if not isinstance(draft, dict) or not str(draft.get("answer") or "").strip():
        raise IncompleteResult("graph returned no answer")
    if outcome not in OUTCOMES:
        raise IncompleteResult(f"graph returned no valid outcome ({outcome!r})")
    if not isinstance(g, dict) or not isinstance(g.get("passed"), bool):
        raise IncompleteResult("graph returned no grounding result")
    if g["passed"] is not EXPECTED_PASSED[outcome]:
        raise IncompleteResult(f"outcome '{outcome}' contradicts grounding passed={g['passed']}")

    results = state.get("tool_results") or []
    warnings = list(dict.fromkeys(w for r in results if r.get("ok") and isinstance(r.get("result"), dict)
                                  for w in (r["result"].get("warnings") or [])))
    u = usage_summary(state)
    return AskResponse(
        request_id=request_id, question=state.get("question", ""), outcome=outcome,
        verified=outcome == "verified",
        answer=draft["answer"], evidence=list(draft.get("evidence") or []),
        key_stats=list(draft.get("key_stats") or []), explanation=draft.get("explanation") or "",
        grounding=Grounding(passed=g["passed"], checked=g.get("checked", 0), unsupported=g.get("unsupported") or [],
                            issues=g.get("issues") or []),
        retries=state.get("retry_count", 0), intent=state.get("intent"),
        analysts=list(state.get("selected_agents") or []),
        tool_results=[ToolResultOut(call_id=r["call_id"], analyst=r["agent"], tool=r["tool"], args=r["args"],
                                    ok=r["ok"], result=r["result"] if r["ok"] else None,
                                    error=None if r["ok"] else {**(r.get("error") or {"code": "unknown",
                                                                                      "message": "unknown error"})})
                      for r in results],
        chart_actions=list(state.get("chart_actions") or []),
        warnings=warnings,
        errors=[ErrorRecord(node=e["node"], kind=e["kind"], message=_public_message(e["kind"], e["message"]))
                for e in state.get("errors") or []],
        usage=Usage(llm_calls=u["calls"], prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"],
                    total_tokens=u["total_tokens"], llm_latency_ms=round(u["llm_latency_ms"], 3),
                    models=sorted({t["model"] for t in state.get("token_usage") or []})),
        trace=list(state.get("node_trace") or []),
        duration_ms=round(duration_ms, 3))