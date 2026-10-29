"""
The state passed between LangGraph nodes.

    question -> ROUTER -> ANALYSTS (in parallel) -> TOOLS -> SYNTHESISER -> GUARD -> pass / retry

Each node returns a dict containing only the keys it changes, and LangGraph merges that
into the state. Keys written by several analysts in the SAME step (because they run in
parallel) must have a reducer, otherwise LangGraph rejects the concurrent update. Those
keys are list-typed and use `operator.add`, so each analyst's items are appended.

Everything in the state is plain JSON-compatible data (dicts, lists, str, numbers), so it
can be logged, returned by the API and stored by MLflow later without conversion.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

Intent = Literal["trend", "offensive", "discipline", "comparison", "hypothesis", "general_statistics"]
AgentName = Literal["trend", "offensive", "discipline"]

INTENTS: tuple[str, ...] = ("trend", "offensive", "discipline", "comparison", "hypothesis", "general_statistics")
AGENTS: tuple[str, ...] = ("trend", "offensive", "discipline")

DEFAULT_MAX_RETRIES = 1
MAX_QUESTION_CHARS = 1000


# ---------------------------------------------------------------------------
# Items stored in the state
# ---------------------------------------------------------------------------
class ToolCall(TypedDict):
    id: str                      # e.g. "offensive-1", used to link a result back to its call
    agent: str                   # which analyst asked for it
    tool: str                    # name in tools.registry.TOOLS
    args: dict[str, Any]


class ToolResult(TypedDict):
    call_id: str
    agent: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None    # the tool's output when ok
    error: dict[str, Any] | None     # {"code", "message", "suggestions"} when not ok


class KeyStat(TypedDict):
    label: str                   # "Arsenal points, 2003-04"
    value: str                   # "90" — kept as text exactly as shown to the user
    source: str                  # call_id of the tool result it came from


class DraftAnswer(TypedDict):
    answer: str                  # direct answer to the question
    evidence: list[str]          # supporting points
    key_stats: list[KeyStat]
    explanation: str             # method / caveats in plain language


class ChartAction(TypedDict, total=False):
    page: Literal["q1", "q2", "q3"]
    league: str
    metric: str                  # Q1 metric
    teams: list[str]             # Q1 teams (max 4 on the dashboard)
    x_metric: str                # Q2/Q3 x axis
    season_start: str            # Q2 season range
    season_end: str
    highlight: list[str]         # "League|Team|Season" keys to emphasise


class GroundingIssue(TypedDict):
    kind: str                    # "unsupported_number", "significance_contradicted", "missing_caveat", ...
    field: str                   # where in the draft: "answer", "evidence[1]", "key_stats[0]", ...
    claim: str                   # the offending text, e.g. "45.5%"
    detail: str                  # what is wrong and what the evidence actually says


class GroundingResult(TypedDict):
    passed: bool
    checked: int                 # how many claims were checked
    unsupported: list[str]       # claims not found in the tool results
    feedback: str                # message given back to the synthesiser on retry
    issues: list[GroundingIssue]  # one entry per problem, for the API and metrics


class NodeUsage(TypedDict):
    node: str                    # "router", "analyst:offensive", "synthesiser", ...
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class AgentError(TypedDict):
    node: str
    kind: str                    # "llm_error", "bad_output", "tool_error", ...
    message: str



class NodeTrace(TypedDict):
    node: str                    # graph node name: "router", "analyst_trend", "synthesiser", ...
    status: Literal["ok", "error"]  # "error" = the node raised and the graph used its safe fallback
    ms: float                    # wall-clock time spent in the node


# How the run ended (set once, by the graph's final node):
#   verified    the answer passed the grounding guard (retry_count says whether a retry was needed)
#   no_evidence no tool produced a usable result; the answer says so and reports no statistics
#   fallback    the written answer failed grounding after the retry; replaced by the tools' own
#               summaries, which passed the guard
#   unverified  even the fallback could not be verified; the answer is marked as unverified
Outcome = Literal["verified", "no_evidence", "fallback", "unverified"]
OUTCOMES: tuple[str, ...] = ("verified", "no_evidence", "fallback", "unverified")

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # input
    question: str
    max_retries: int

    # router
    intent: Intent
    selected_agents: list[AgentName]
    plan: dict[str, str]                                  # agent -> what to look into

    # analysts (parallel writers -> reducers)
    tool_calls: Annotated[list[ToolCall], add]
    tool_results: Annotated[list[ToolResult], add]

    # synthesiser / guard loop
    draft_answer: DraftAnswer | None
    chart_actions: list[ChartAction]
    grounding_result: GroundingResult | None
    retry_count: int
    outcome: Outcome | None                               # set by the graph's final node

    # bookkeeping (any node may append, including parallel analysts)
    token_usage: Annotated[list[NodeUsage], add]          # one record per LLM call, incl. latency
    errors: Annotated[list[AgentError], add]
    node_trace: Annotated[list[NodeTrace], add]           # one record per node run, in order


def new_state(question: str, max_retries: int = DEFAULT_MAX_RETRIES) -> AgentState:
    """Initial state for one question. Validates the input; everything else starts empty."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    q = " ".join(question.split())
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    return AgentState(
        question=q, max_retries=max_retries,
        selected_agents=[], plan={},
        tool_calls=[], tool_results=[],
        draft_answer=None, chart_actions=[], grounding_result=None, retry_count=0, outcome=None,
        token_usage=[], errors=[], node_trace=[],
    )


# ---------------------------------------------------------------------------
# Read-only helpers (used by tests now, by the API and MLflow logging later)
# ---------------------------------------------------------------------------
def usage_record(node: str, response) -> NodeUsage:
    """Build a NodeUsage from an llm.base.LLMResponse."""
    return NodeUsage(node=node, model=response.model, prompt_tokens=response.usage.prompt_tokens,
                     completion_tokens=response.usage.completion_tokens, latency_ms=response.latency_ms)


def usage_summary(state: AgentState) -> dict[str, Any]:
    """Totals overall and per node."""
    per_node: dict[str, dict[str, float]] = {}
    for u in state.get("token_usage", []):
        n = per_node.setdefault(u["node"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                            "latency_ms": 0.0})
        n["calls"] += 1
        n["prompt_tokens"] += u["prompt_tokens"]
        n["completion_tokens"] += u["completion_tokens"]
        n["latency_ms"] += u["latency_ms"]
    prompt = sum(n["prompt_tokens"] for n in per_node.values())
    completion = sum(n["completion_tokens"] for n in per_node.values())
    return {"calls": sum(n["calls"] for n in per_node.values()),
            "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "llm_latency_ms": sum(n["latency_ms"] for n in per_node.values()),
            "per_node": per_node}


def successful_results(state: AgentState) -> list[ToolResult]:
    return [r for r in state.get("tool_results", []) if r["ok"]]