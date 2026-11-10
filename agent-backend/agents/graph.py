"""
The LangGraph workflow: connects the existing nodes, adds no analysis of its own.

    START -> router -> analyst_trend | analyst_offensive | analyst_discipline   (parallel, via Send)
          -> synthesiser -> guard -> (passed) ---------------------------> finalise -> END
                                  -> (failed, retry left) -> retry -> synthesiser -> guard -> ...
                                  -> (failed, no retry left) -----------> finalise -> END

What the graph does itself (and nothing more):
  * fan-out: sends the question to exactly the analysts the router selected, in parallel.
    Their list updates are merged by the reducers in agents/state.py, so nothing is overwritten.
  * the retry loop: at most MAX_SYNTHESIS_RETRIES (= 1) extra synthesis attempt after a failed
    grounding check. The retry reuses the tool results already collected: no analyst or tool
    runs again. The synthesiser sees the guard's feedback through state["grounding_result"].
  * the final outcome (state["outcome"]): verified / no_evidence / fallback / unverified.
    If the retry also fails grounding, the written answer is replaced by the synthesiser's
    deterministic fallback (the tools' own summaries), which is checked by the guard again.
  * safety net: every node runs inside _node(), which records its wall-clock time in
    state["node_trace"]. If a node raises an unexpected exception (a bug, not an LLM error,
    which the nodes already handle), the exception is recorded in state["errors"] and the
    node's existing safe fallback is used, so ordinary failures never crash a run.

The graph never calls a tool, never calls an LLM directly and never computes a statistic.

Usage
    from agents.graph import build_graph, ask
    from llm.mock import MockLLM
    app = build_graph(MockLLM({...}))            # or router_llm=..., analyst_llm=..., synthesiser_llm=...
    final = ask(app, "Do teams that take more shots get more points?")
    final["draft_answer"], final["outcome"], final["grounding_result"]["passed"]

    for node, update in stream(app, question): ...   # node-by-node progress (for the API later)

    python -m agents.graph --mermaid                 # print the graph as a Mermaid diagram
"""
from __future__ import annotations

import sys
import uuid
from time import perf_counter
from typing import Any, Callable, Iterator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from llm.base import LLMClient, LLMError

from . import synthesiser as syn
from .analysts import make_analyst
from .guard import make_guard
from .router import AGENT_TOOLS, make_router, rule_based_route
from .state import (AGENTS, DEFAULT_MAX_RETRIES, AgentError, AgentState, GroundingIssue, GroundingResult,
                    NodeTrace, new_state, successful_results)

# Node names (LangGraph reserves ":" in node names, hence analyst_trend rather than analyst:trend)
ROUTER = "router"
ANALYST_NODES = {a: f"analyst_{a}" for a in AGENTS}
UNSUPPORTED = "unsupported_route"
SYNTHESISER = "synthesiser"
GUARD = "guard"
RETRY = "retry"
FINALISE = "finalise"
NODE_NAMES = (ROUTER, *ANALYST_NODES.values(), UNSUPPORTED, SYNTHESISER, GUARD, RETRY, FINALISE)

GRAPH = "graph"                  # `node` used for errors recorded by the graph itself
MAX_SYNTHESIS_RETRIES = 1        # hard cap; state["max_retries"] can only lower it (0 = no retry)
RECURSION_LIMIT = 20             # the longest legal run is 8 steps; anything near 20 is a bug

FALLBACK_REASON = "its draft did not pass the grounding check against the tool results"
UNVERIFIED_NOTE = ("This answer could not be fully verified against the tool results, "
                   "so treat it with caution.")


# ---------------------------------------------------------------------------
# Safety net around every node
# ---------------------------------------------------------------------------
def _node(name: str, fn: Callable[[AgentState], dict],
          on_error: Callable[[AgentState, Exception], dict]) -> Callable[[AgentState], dict]:
    """Run fn; time it; on an unexpected exception record it and use the node's safe fallback."""
    def run(state: AgentState) -> dict:
        t0 = perf_counter()
        try:
            out, status = dict(fn(state)), "ok"
        except Exception as e:  # noqa: BLE001 - recorded below, never silent
            out, status = dict(on_error(state, e)), "error"
            out["errors"] = list(out.get("errors", [])) + [AgentError(
                node=name, kind="node_exception", message=f"{type(e).__name__}: {e}"[:500])]
        out["node_trace"] = [NodeTrace(node=name, status=status, ms=round((perf_counter() - t0) * 1000, 3))]
        return out
    run.__name__ = name
    return run


def _router_fallback(state: AgentState, _e: Exception) -> dict:
    return rule_based_route(state["question"])          # the router's own deterministic fallback


def _no_update(_state: AgentState, _e: Exception) -> dict:
    return {}                                           # an analyst that crashed contributes no results


def _synthesiser_fallback(state: AgentState, e: Exception) -> dict:
    draft = (syn.fallback_draft(state, type(e).__name__) if successful_results(state)
             else syn.no_evidence_draft(state))
    return {"draft_answer": draft, "chart_actions": []}


def _guard_fallback(_state: AgentState, e: Exception) -> dict:
    detail = f"the guard could not check this draft ({type(e).__name__})"
    issue = GroundingIssue(kind="guard_error", field="", claim="", detail=detail)
    return {"grounding_result": GroundingResult(passed=False, checked=0, unsupported=[],
                                                feedback=f"- [guard_error] {detail}", issues=[issue])}


def _retry_fallback(state: AgentState, _e: Exception) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}   # still counts, so the loop stays bounded


def _finalise_fallback(_state: AgentState, _e: Exception) -> dict:
    return {"outcome": "unverified", "chart_actions": []}


# ---------------------------------------------------------------------------
# Graph-level nodes and edges (bookkeeping only)
# ---------------------------------------------------------------------------
def select_analysts(state: AgentState) -> list[Send]:
    """Router -> one Send per selected, implemented analyst. Anything else goes to unsupported_route."""
    selected = state.get("selected_agents") or []
    known = [a for a in AGENTS if a in selected]
    unknown = [a for a in selected if a not in AGENT_TOOLS]
    sends = [Send(ANALYST_NODES[a], state) for a in known]
    if unknown or not known:
        sends.append(Send(UNSUPPORTED, {"intent": state.get("intent"), "requested": list(selected)}))
    return sends


def unsupported_route(payload: dict) -> dict:
    requested = payload.get("requested") or []
    unknown = [a for a in requested if a not in AGENT_TOOLS]
    msg = (f"Router selected analysts that do not exist: {unknown}." if unknown
           else f"Router selected no analyst for intent '{payload.get('intent')}'.")
    return {"errors": [AgentError(node=GRAPH, kind="unsupported_route", message=msg)]}


def allowed_retries(state: AgentState) -> int:
    return max(0, min(int(state.get("max_retries", DEFAULT_MAX_RETRIES)), MAX_SYNTHESIS_RETRIES))


def after_guard(state: AgentState) -> str:
    g = state.get("grounding_result") or {}
    if g.get("passed"):
        return FINALISE
    if not successful_results(state):
        return FINALISE          # the no-evidence draft is deterministic: a retry would repeat it
    if state.get("retry_count", 0) < allowed_retries(state):
        return RETRY
    return FINALISE


def retry(state: AgentState) -> dict:
    n = state.get("retry_count", 0) + 1
    g = state.get("grounding_result") or {}
    issues = "; ".join(f"{i['kind']}: {i['claim']}" for i in g.get("issues", [])[:8]) or "no details"
    return {"retry_count": n,
            "errors": [AgentError(node=GRAPH, kind="grounding_rejected",
                                  message=f"Synthesis attempt {n} failed grounding ({issues})."[:500])]}


def make_finalise(guard_node: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    def finalise(state: AgentState) -> dict:
        has_evidence = bool(successful_results(state))
        g = state.get("grounding_result") or {}
        if g.get("passed"):
            return {"outcome": "verified" if has_evidence else "no_evidence"}

        # Retries used up: never show the rejected draft. Use the synthesiser's deterministic
        # answer and check that one too, so grounding_result always describes draft_answer.
        draft = syn.fallback_draft(state, FALLBACK_REASON) if has_evidence else syn.no_evidence_draft(state)
        checked = guard_node({**state, "draft_answer": draft})
        result = checked["grounding_result"]
        errors = [AgentError(node=GRAPH, kind="grounding_failed",
                             message="The written answer failed grounding after the permitted retry; "
                                     "replaced by the tools' own summaries.")]
        errors += checked.get("errors", [])
        if result["passed"]:
            outcome = "fallback" if has_evidence else "no_evidence"
        else:
            outcome = "unverified"
            draft = {**draft, "explanation": f"{draft['explanation']} {UNVERIFIED_NOTE}".strip()}
        return {"draft_answer": draft, "chart_actions": [], "grounding_result": result,
                "outcome": outcome, "errors": errors}
    return finalise


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_graph(llm: LLMClient | None = None, *, router_llm: LLMClient | None = None,
                analyst_llm: LLMClient | None = None, synthesiser_llm: LLMClient | None = None,
                prompt_version: str | None = None, data=None, checkpointer=None):
    """
    Compile the workflow. `llm` is used for every role unless a role-specific client is given
    (e.g. a small fast model for the router, a stronger one for the synthesiser).
    `data` injects a Dataset into the analysts (tests); `checkpointer` (e.g. LangGraph's
    InMemorySaver) keeps every step of each run, keyed by thread_id, for inspection or replay.
    No LLM is called while building.
    """
    clients = {"router": router_llm or llm, "analyst": analyst_llm or llm, "synthesiser": synthesiser_llm or llm}
    missing = [role for role, c in clients.items() if c is None]
    if missing:
        raise ValueError(f"No LLM client for: {', '.join(missing)}. Pass llm= or the role-specific clients.")

    guard_node = make_guard()
    g = StateGraph(AgentState)
    g.add_node(ROUTER, _node(ROUTER, make_router(clients["router"], prompt_version), _router_fallback))
    for agent, name in ANALYST_NODES.items():
        g.add_node(name, _node(name, make_analyst(agent, clients["analyst"], prompt_version, data), _no_update))
    g.add_node(UNSUPPORTED, _node(UNSUPPORTED, unsupported_route, _no_update))
    g.add_node(SYNTHESISER, _node(SYNTHESISER, syn.make_synthesiser(clients["synthesiser"], prompt_version),
                                  _synthesiser_fallback))
    g.add_node(GUARD, _node(GUARD, guard_node, _guard_fallback))
    g.add_node(RETRY, _node(RETRY, retry, _retry_fallback))
    g.add_node(FINALISE, _node(FINALISE, make_finalise(guard_node), _finalise_fallback))

    g.add_edge(START, ROUTER)
    g.add_conditional_edges(ROUTER, select_analysts, [*ANALYST_NODES.values(), UNSUPPORTED])
    for name in (*ANALYST_NODES.values(), UNSUPPORTED):
        g.add_edge(name, SYNTHESISER)
    g.add_edge(SYNTHESISER, GUARD)
    g.add_conditional_edges(GUARD, after_guard, [RETRY, FINALISE])
    g.add_edge(RETRY, SYNTHESISER)
    g.add_edge(FINALISE, END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------
def run_config(app, thread_id: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"recursion_limit": RECURSION_LIMIT}
    if getattr(app, "checkpointer", None) is not None:
        config["configurable"] = {"thread_id": thread_id or str(uuid.uuid4())}
    return config


def ask(app, question: str, *, max_retries: int = DEFAULT_MAX_RETRIES, thread_id: str | None = None) -> AgentState:
    """Answer one question and return the final state. Raises ValueError for an invalid question."""
    return app.invoke(new_state(question, max_retries), config=run_config(app, thread_id))


def stream(app, question: str, *, max_retries: int = DEFAULT_MAX_RETRIES,
           thread_id: str | None = None) -> Iterator[tuple[str, dict]]:
    """Yield (node, update) as each node finishes: for live progress in the API/frontend later."""
    for chunk in app.stream(new_state(question, max_retries), config=run_config(app, thread_id),
                            stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update or {}


class _UnusedLLM:
    """Placeholder client for drawing the graph; building never calls an LLM."""
    def complete(self, *a, **k):
        raise LLMError("no LLM configured")


def mermaid() -> str:
    return build_graph(_UnusedLLM()).get_graph().draw_mermaid()


if __name__ == "__main__":
    if "--mermaid" in sys.argv:
        print(mermaid())
    else:
        print("usage: python -m agents.graph --mermaid      (python -m agents.demo runs example questions)")