"""
Stage 7: the LangGraph workflow, end to end. Every LLM is MockLLM; every tool result comes
from the real registry on the real data. No API key, no network.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import get_type_hints

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import tools.registry as registry
from agents import graph as GR
from agents import guard as G
from agents import synthesiser as syn
from agents.router import DEFAULT_AGENTS, rule_based_route
from agents.state import INTENTS, OUTCOMES, AgentState, new_state
from llm.base import LLMRateLimitError, LLMTimeoutError
from llm.mock import MockLLM

# ---------------------------------------------------------------------------
# Scripted LLM replies (the numbers in drafts are the real tool outputs)
# ---------------------------------------------------------------------------
SHOTS = {"tool_calls": [{"tool": "correlation_analysis", "arguments": {"x_metric": "AvgShots"}}]}
CARDS = {"tool_calls": [{"tool": "correlation_analysis", "arguments": {"x_metric": "AvgYellowCards"}}]}
ARSENAL = {"tool_calls": [{"tool": "team_season_stats", "arguments": {"team": "Arsenal", "season": "2003-04"}}]}
COVID = {"tool_calls": [{"tool": "home_advantage_test", "arguments": {"league": "Serie A"}}]}
TABLE = {"tool_calls": [{"tool": "league_table", "arguments": {"league": "Premier League", "season": "2004-05"}}]}
TYPO = {"tool_calls": [{"tool": "team_season_stats", "arguments": {"team": "Arsenl", "season": "2003-04"}}]}


def draft(answer, stats=(), charts=(), explanation="Computed by the analysis tools."):
    return {"answer": answer, "evidence": [], "key_stats": list(stats), "explanation": explanation,
            "chart_actions": list(charts)}


SHOTS_OK = draft("Teams that take more shots tend to earn more points (r = 0.597); correlation does not "
                 "establish causation.", [{"label": "Pearson r", "value": "0.597", "source": "offensive-1"}],
                 [{"page": "q2", "x_metric": "AvgShots", "league": "all"}])
SHOTS_BAD = draft("More shots lead to more points (r = 0.9).", [], [{"page": "q2", "x_metric": "AvgShots"}])
CARDS_OK = draft("Teams that receive more yellow cards tend to earn fewer points (r = -0.207); correlation "
                 "does not establish causation.", [{"label": "Pearson r", "value": "-0.207", "source": "discipline-1"}],
                 [{"page": "q3", "x_metric": "AvgYellowCards", "league": "all"}])
ARSENAL_OK = draft("Arsenal went unbeaten in 2003-04: 26 wins and 90 points.",
                   [{"label": "Points", "value": "90", "source": "trend-1"}])
ARSENAL_BAD = draft("Arsenal won the 2003-04 league with 92 points.")
COVID_OK = draft("No. Serie A's home win rate fell from 45.4% to 40.8% in 2020-21, but the change is not "
                 "statistically significant (p = 0.0841).", [{"label": "p-value", "value": "0.0841", "source": "trend-1"}])
TABLE_OK = draft("Chelsea won the 2004-05 Premier League with 95 points.",
                 [{"label": "Chelsea points", "value": "95", "source": "trend-1"}])


def llm_for(router, synth, **analysts):
    script = {"router": router, "synthesiser": synth}
    script.update({f"analyst:{a}": v for a, v in analysts.items()})
    return MockLLM(script)


def route(intent, *agents):
    return {"intent": intent, "agents": list(agents)}


def run(llm, question="q?", **kw):
    return GR.ask(GR.build_graph(llm), question, **kw)


def trace(s):
    return [t["node"] for t in s["node_trace"]]


def kinds(s):
    return [(e["node"], e["kind"]) for e in s["errors"]]


@pytest.fixture
def tool_counter(monkeypatch):
    """Counts real registry calls (the tools still run)."""
    calls, real = [], registry.call_tool

    def counting(name, args, **kw):
        calls.append(name)
        return real(name, args, **kw)
    monkeypatch.setattr(registry, "call_tool", counting)
    return calls


# ---------------------------------------------------------------------------
# 1-7  construction and routing
# ---------------------------------------------------------------------------
def test_graph_builds_with_expected_nodes_and_without_calling_llm():
    llm = MockLLM({})                               # any call would raise MockScriptError
    app = GR.build_graph(llm)
    assert set(GR.NODE_NAMES) <= set(app.get_graph().nodes)
    assert llm.calls == []


def test_build_requires_a_client_for_every_role():
    with pytest.raises(ValueError, match="router"):
        GR.build_graph(analyst_llm=MockLLM(), synthesiser_llm=MockLLM())
    GR.build_graph(router_llm=MockLLM(), analyst_llm=MockLLM(), synthesiser_llm=MockLLM())


def test_trend_question_reaches_trend_analyst_only():
    llm = llm_for(route("trend", "trend"), ARSENAL_OK, trend=ARSENAL)
    s = run(llm, "How did Arsenal do in 2003-04?")
    assert trace(s)[:2] == ["router", "analyst_trend"] and "analyst_offensive" not in trace(s)
    assert [(r["agent"], r["tool"]) for r in s["tool_results"]] == [("trend", "team_season_stats")]
    assert [c.purpose for c in llm.calls] == ["router", "analyst:trend", "synthesiser"]


def test_offensive_question_reaches_offensive_analyst_only():
    llm = llm_for(route("offensive", "offensive"), SHOTS_OK, offensive=SHOTS)
    s = run(llm, "Do teams that take more shots get more points?")
    assert [r["call_id"] for r in s["tool_results"]] == ["offensive-1"]
    assert "analyst_trend" not in trace(s) and "analyst_discipline" not in trace(s)


def test_discipline_question_reaches_discipline_analyst_only():
    llm = llm_for(route("discipline", "discipline"), CARDS_OK, discipline=CARDS)
    s = run(llm, "Do yellow cards cost teams points?")
    assert [r["call_id"] for r in s["tool_results"]] == ["discipline-1"]
    assert s["outcome"] == "verified"


@pytest.mark.parametrize("intent", INTENTS)
def test_every_intent_has_an_intentional_path(intent):
    """Router names only the intent: the documented default analysts run and produce real results."""
    expected = DEFAULT_AGENTS[intent]
    answers = {"trend": ARSENAL, "offensive": SHOTS, "discipline": CARDS}
    llm = llm_for(route(intent), LLMTimeoutError("not needed here"),  # synthesiser falls back
                  **{a: answers[a] for a in expected})
    s = run(llm)
    assert s["intent"] == intent and s["selected_agents"] == expected
    assert sorted({r["agent"] for r in s["tool_results"]}) == sorted(expected)
    assert all(r["ok"] for r in s["tool_results"])
    assert s["outcome"] in OUTCOMES


@pytest.mark.parametrize("question,intent,agents", [
    ("Is the 2020-21 drop in Serie A home advantage statistically significant?", "hypothesis", ["trend"]),
    ("Compare Arsenal and Tottenham", "comparison", ["trend"]),
    ("Who topped the 2004-05 Premier League table?", "general_statistics", ["trend"]),
    ("Compare Arsenal and Chelsea on yellow cards and shots", "comparison", ["offensive", "discipline"]),
])
def test_fallback_routing_of_non_specialist_intents(question, intent, agents):
    """Router LLM down: the rule-based router decides, and the graph follows it."""
    replies = {"trend": ARSENAL, "offensive": SHOTS, "discipline": CARDS}
    llm = llm_for(LLMTimeoutError("down"), ARSENAL_OK, **{a: replies[a] for a in agents})
    s = run(llm, question)
    assert (s["intent"], s["selected_agents"]) == (intent, agents) == tuple(rule_based_route(question).values())[:2]
    assert sorted({r["agent"] for r in s["tool_results"]}) == sorted(agents)


def test_multi_analyst_question_runs_both_in_parallel_and_keeps_both_results():
    llm = llm_for(route("offensive", "offensive", "discipline"), SHOTS_OK, offensive=SHOTS, discipline=CARDS)
    s = run(llm, "Do shots and yellow cards relate to points?")
    assert sorted(r["call_id"] for r in s["tool_results"]) == ["discipline-1", "offensive-1"]
    assert sorted(c["id"] for c in s["tool_calls"]) == ["discipline-1", "offensive-1"]
    assert trace(s).count("synthesiser") == 1                  # synthesiser waits for both analysts
    assert set(trace(s)[1:3]) == {"analyst_offensive", "analyst_discipline"}


@pytest.mark.parametrize("router_reply", [
    {"intent": "weather", "agents": ["trend"]},                 # unknown intent
    {"intent": "offensive", "agents": ["goalkeeping"]},         # unknown analyst
    "not json at all",
    ["offensive"],
])
def test_malformed_router_output_uses_rule_based_route(router_reply):
    llm = llm_for(router_reply, SHOTS_OK, offensive=SHOTS)
    s = run(llm, "Do teams that take more shots get more points?")
    assert ("router", "bad_output") in kinds(s)
    assert s["selected_agents"] == ["offensive"] and s["outcome"] == "verified"


@pytest.mark.parametrize("selected,unknown", [([], False), (["goalkeeping"], True), (["trend", "goalkeeping"], True)])
def test_unsupported_selection_is_explicit_and_safe(monkeypatch, selected, unknown):
    """A router that bypasses validation still cannot reach a non-existent analyst."""
    monkeypatch.setattr(GR, "make_router", lambda llm, v=None: lambda s: {"intent": "trend", "selected_agents": selected})
    llm = llm_for(route("trend"), ARSENAL_OK, trend=ARSENAL)
    s = run(llm, "How did Arsenal do in 2003-04?")
    assert ("graph", "unsupported_route") in kinds(s)
    msg = next(e["message"] for e in s["errors"] if e["kind"] == "unsupported_route")
    assert ("goalkeeping" in msg) == unknown
    if "trend" in selected:
        assert s["outcome"] == "verified" and s["tool_results"]
    else:
        assert s["tool_results"] == [] and s["outcome"] == "no_evidence"
        assert "could not establish" in s["draft_answer"]["answer"]


def test_initial_state_and_final_state_follow_schema():
    s0 = new_state("How did Arsenal do in 2003-04?")
    assert set(s0) <= set(get_type_hints(AgentState)) and s0["outcome"] is None and s0["node_trace"] == []
    s = run(llm_for(route("trend", "trend"), ARSENAL_OK, trend=ARSENAL))
    assert set(s) <= set(get_type_hints(AgentState))
    assert json.loads(json.dumps(s, allow_nan=False)) == s       # plain JSON all the way through


# ---------------------------------------------------------------------------
# 8-12  state preservation
# ---------------------------------------------------------------------------
def test_analyst_results_survive_synthesis_and_reach_the_guard(monkeypatch):
    seen = {}
    real_check = G.check

    def spy(state):
        seen["ids"] = [r["call_id"] for r in state["tool_results"]]
        return real_check(state)
    monkeypatch.setattr(G, "check", spy)
    s = run(llm_for(route("offensive", "offensive", "discipline"), SHOTS_OK, offensive=SHOTS, discipline=CARDS))
    assert sorted(seen["ids"]) == ["discipline-1", "offensive-1"]
    r = next(r for r in s["tool_results"] if r["call_id"] == "offensive-1")
    assert r["result"]["r"] == registry.call_tool("correlation_analysis", {"x_metric": "AvgShots"})["result"]["r"]


def test_errors_and_token_usage_accumulate_across_nodes():
    llm = llm_for(LLMRateLimitError("429"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS)
    s = run(llm, "Do teams that take more shots get more points?")
    assert kinds(s) == [("router", "llm_error"), ("graph", "grounding_rejected")]
    assert [u["node"] for u in s["token_usage"]] == ["analyst:offensive", "synthesiser", "synthesiser"]
    assert sum(u["prompt_tokens"] + u["completion_tokens"] for u in s["token_usage"]) == llm.total_usage.total_tokens


def test_chart_actions_survive_a_verified_answer():
    s = run(llm_for(route("offensive", "offensive"), SHOTS_OK, offensive=SHOTS))
    assert s["chart_actions"] == [{"page": "q2", "league": "all", "x_metric": "AvgShots"}]


def test_chart_actions_from_a_rejected_draft_are_dropped():
    s = run(llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_BAD], offensive=SHOTS))
    assert s["outcome"] == "fallback" and s["chart_actions"] == []


def test_nodes_do_not_overwrite_unrelated_state():
    llm = llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS)
    s = GR.ask(GR.build_graph(llm), "  Do teams that take   more shots get more points? ")
    assert s["question"] == "Do teams that take more shots get more points?"
    assert s["max_retries"] == 1 and s["plan"] == {"offensive": s["question"]}
    assert len(s["tool_calls"]) == len(s["tool_results"]) == 1   # not duplicated by the retry
    assert s["intent"] == "offensive" and s["selected_agents"] == ["offensive"]


# ---------------------------------------------------------------------------
# 13-20  grounding and the bounded retry
# ---------------------------------------------------------------------------
def test_valid_first_draft_ends_without_retry():
    llm = llm_for(route("offensive", "offensive"), [SHOTS_OK], offensive=SHOTS)
    s = run(llm)
    assert s["outcome"] == "verified" and s["retry_count"] == 0 and s["grounding_result"]["passed"]
    assert trace(s)[-3:] == ["synthesiser", "guard", "finalise"] and "retry" not in trace(s)
    assert s["draft_answer"]["answer"] == SHOTS_OK["answer"]


def test_invalid_first_draft_retries_once_with_feedback_and_succeeds(tool_counter):
    llm = llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS)
    s = run(llm)
    assert trace(s) == ["router", "analyst_offensive", "synthesiser", "guard", "retry", "synthesiser", "guard",
                        "finalise"]
    assert s["retry_count"] == 1 and s["outcome"] == "verified" and s["grounding_result"]["passed"]
    assert s["draft_answer"]["answer"] == SHOTS_OK["answer"]
    first, second = llm.calls_for("synthesiser")
    assert "GROUNDING FEEDBACK" not in first.request.last_user
    assert "GROUNDING FEEDBACK" in second.request.last_user
    assert "0.9" in second.request.last_user and "lead to" in second.request.last_user
    assert "0.596685" in second.request.last_user                  # the guard told it the real value
    assert tool_counter == ["correlation_analysis"]                # retry reused the evidence


def test_invalid_retry_does_not_trigger_a_third_attempt_and_falls_back():
    llm = llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_BAD, SHOTS_OK], offensive=SHOTS)
    s = run(llm)
    assert len(llm.calls_for("synthesiser")) == 2 and s["retry_count"] == 1
    assert trace(s).count("synthesiser") == 2 and trace(s).count("guard") == 2
    assert s["outcome"] == "fallback"
    # the final answer is the deterministic fallback, NOT the rejected draft, and it passed the guard
    assert s["draft_answer"]["answer"] != SHOTS_BAD["answer"] and "0.9" not in json.dumps(s["draft_answer"])
    assert "r = 0.597" in " ".join(s["draft_answer"]["evidence"])
    assert s["grounding_result"]["passed"]
    assert [k for _, k in kinds(s)] == ["grounding_rejected", "grounding_failed"]


def test_final_state_never_reports_a_failed_check_as_passed(monkeypatch):
    """If even the fallback fails the guard, the run says so: unverified, passed = False."""
    real_check = G.check
    monkeypatch.setattr(G, "check", lambda st: {**real_check(st), "passed": False,
                                                "issues": [{"kind": "x", "field": "", "claim": "", "detail": "d"}]})
    s = run(llm_for(route("offensive", "offensive"), [SHOTS_OK, SHOTS_OK], offensive=SHOTS))
    assert s["outcome"] == "unverified" and s["grounding_result"]["passed"] is False
    assert GR.UNVERIFIED_NOTE in s["draft_answer"]["explanation"]
    assert s["chart_actions"] == []


@pytest.mark.parametrize("max_retries,attempts", [(0, 1), (1, 2), (5, 2)])
def test_maximum_synthesis_attempts_is_enforced(max_retries, attempts):
    llm = llm_for(route("offensive", "offensive"), lambda req: SHOTS_BAD, offensive=SHOTS)
    s = run(llm, max_retries=max_retries)
    assert len(llm.calls_for("synthesiser")) == attempts and s["retry_count"] == attempts - 1
    assert s["outcome"] == "fallback"


def test_no_evidence_question_returns_safe_answer_without_synthesis_llm():
    llm = llm_for(route("trend", "trend"), [], trend=TYPO)            # synthesiser queue empty: must not be called
    s = run(llm, "How did Arsenl do in 2003-04?")
    assert s["outcome"] == "no_evidence" and s["grounding_result"]["passed"]
    assert "could not establish" in s["draft_answer"]["answer"] and s["draft_answer"]["key_stats"] == []
    assert llm.calls_for("synthesiser") == [] and s["retry_count"] == 0


def test_no_evidence_draft_is_not_retried_even_if_rejected(monkeypatch):
    """With no successful tool result the synthesiser is deterministic, so a retry could only repeat it."""
    monkeypatch.setattr(G, "check", lambda st: {"passed": False, "checked": 0, "unsupported": [], "feedback": "f",
                                                "issues": [{"kind": "x", "field": "", "claim": "", "detail": "d"}]})
    s = run(llm_for(route("trend", "trend"), [], trend=TYPO), "How did Arsenl do in 2003-04?")
    assert trace(s).count("synthesiser") == 1 and s["retry_count"] == 0 and "retry" not in trace(s)
    assert s["outcome"] == "unverified" and s["grounding_result"]["passed"] is False


def test_failed_tool_numbers_do_not_become_evidence():
    poisoned = draft("Arsenl had 90 points in 2003-04.", [{"label": "Points", "value": "90", "source": "trend-1"}])
    llm = llm_for(route("trend", "trend"), [poisoned, poisoned],
                  trend={"tool_calls": [TYPO["tool_calls"][0], ARSENAL["tool_calls"][0]]})
    s = run(llm, "How did Arsenal and Arsenl do?")
    failed = next(r for r in s["tool_results"] if not r["ok"])
    assert failed["call_id"] == "trend-1"
    # the synthesiser refuses a key stat citing a failed call (twice), then falls back to tool text
    rejected = [e["message"] for e in s["errors"] if e["kind"] == "bad_output"]
    assert len(rejected) == 2 and all("trend-1" in m for m in rejected)
    assert s["draft_answer"]["answer"] != poisoned["answer"] and s["draft_answer"]["key_stats"] == []
    assert s["grounding_result"]["passed"] and s["outcome"] == "verified"


def test_failed_tool_cannot_back_a_prose_claim():
    """No key stat this time: the guard itself rejects a figure that exists only in a failed call."""
    failed = {"call_id": "trend-1", "agent": "trend", "tool": "team_season_stats", "args": {}, "ok": False,
              "result": None, "error": {"code": "unknown_team", "message": "Arsenl had 999 points", "suggestions": []}}
    s = new_state("How did Arsenal do?")
    s.update(tool_results=[failed, {**failed, "call_id": "trend-2", "ok": True, "error": None,
                                    "result": registry.call_tool("team_season_stats", ARSENAL["tool_calls"][0]["arguments"])["result"]}],
             draft_answer=draft("Arsenal had 999 points; the other lookup failed."))
    assert G.check(s)["unsupported"] == ["999"]


# ---------------------------------------------------------------------------
# 21-26  error handling
# ---------------------------------------------------------------------------
def test_router_llm_failure_uses_rule_based_route():
    llm = llm_for(LLMRateLimitError("429"), COVID_OK, trend=COVID)
    s = run(llm, "Was home advantage significantly lower during COVID in Serie A?")
    assert ("router", "llm_error") in kinds(s)
    assert (s["intent"], s["selected_agents"]) == ("hypothesis", ["trend"])
    assert s["outcome"] == "verified"


def test_analyst_llm_failure_does_not_crash_and_other_analyst_still_counts():
    llm = llm_for(route("offensive", "offensive", "discipline"), SHOTS_OK,
                  offensive=SHOTS, discipline=LLMTimeoutError("slow"))
    s = run(llm)
    assert ("analyst:discipline", "llm_error") in kinds(s)
    assert [r["call_id"] for r in s["tool_results"]] == ["offensive-1"] and s["outcome"] == "verified"


def test_all_analysts_failing_gives_no_evidence_answer():
    s = run(llm_for(route("trend", "trend"), [], trend=LLMTimeoutError("slow")))
    assert s["outcome"] == "no_evidence" and ("analyst:trend", "llm_error") in kinds(s)


def test_synthesiser_llm_fallback_is_preserved():
    s = run(llm_for(route("trend", "trend"), LLMTimeoutError("slow"), trend=COVID))
    assert ("synthesiser", "llm_error") in kinds(s)
    assert "could not be generated" in s["draft_answer"]["answer"]
    assert s["outcome"] == "verified" and s["grounding_result"]["passed"]     # tools' own text passes


def test_guard_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(G, "check", lambda st: 1 / 0)
    llm = llm_for(route("offensive", "offensive"), [SHOTS_OK, SHOTS_OK], offensive=SHOTS)
    s = run(llm)
    assert len(llm.calls_for("synthesiser")) == 2                  # bounded even when the guard is broken
    assert s["outcome"] == "unverified" and s["grounding_result"]["passed"] is False
    assert s["grounding_result"]["issues"][0]["kind"] == "guard_error"
    assert [k for n, k in kinds(s) if n == "guard"].count("guard_error") == 3     # two checks + the fallback check


@pytest.mark.parametrize("broken", ["router", "analyst:offensive", "synthesiser"])
def test_unexpected_node_exception_is_recorded_and_run_completes(broken):
    script = {"router": route("offensive", "offensive"), "analyst:offensive": SHOTS, "synthesiser": SHOTS_OK}
    script[broken] = lambda req: (_ for _ in ()).throw(RuntimeError(f"bug in {req.purpose}"))
    s = run(MockLLM(script), "Do teams that take more shots get more points?")
    node = {"router": "router", "analyst:offensive": "analyst_offensive", "synthesiser": "synthesiser"}[broken]
    assert (node, "node_exception") in kinds(s)
    assert any(t["node"] == node and t["status"] == "error" for t in s["node_trace"])
    assert s["outcome"] in ("verified", "no_evidence") and s["draft_answer"]["answer"]
    if broken == "router":
        assert s["selected_agents"] == ["offensive"]               # rule-based route used instead


def test_earlier_errors_are_not_lost_later_in_the_run():
    llm = llm_for(LLMRateLimitError("429"), [SHOTS_BAD, SHOTS_BAD],
                  offensive={"tool_calls": [SHOTS["tool_calls"][0], {"tool": "league_table", "arguments": {}}]})
    s = run(llm, "Do teams that take more shots get more points?")
    assert [k for _, k in kinds(s)] == ["llm_error", "tool_not_allowed", "grounding_rejected", "grounding_failed"]


# ---------------------------------------------------------------------------
# 27-32  integration, isolation, determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,router,agent,reply,good,needle", [
    ("Who won the 2004-05 Premier League?", route("general_statistics"), "trend", TABLE, TABLE_OK, "95 points"),
    ("Do teams that take more shots get more points?", route("offensive"), "offensive", SHOTS, SHOTS_OK, "0.597"),
    ("Do yellow cards cost teams points?", route("discipline"), "discipline", CARDS, CARDS_OK, "-0.207"),
])
def test_complete_runs(question, router, agent, reply, good, needle):
    llm = llm_for(router, good, **{agent: reply})
    s = run(llm, question)
    assert s["outcome"] == "verified" and s["grounding_result"]["passed"] and s["errors"] == []
    assert needle in s["draft_answer"]["answer"]
    assert s["draft_answer"]["key_stats"][0]["source"] == f"{agent}-1"
    assert [t["status"] for t in s["node_trace"]] == ["ok"] * 5
    assert all(t["ms"] >= 0 for t in s["node_trace"])


def test_runs_are_deterministic():
    def once():
        s = run(llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS))
        s.pop("node_trace")                                        # wall-clock times differ
        return json.dumps(s, sort_keys=True)
    assert once() == once()


def test_no_real_llm_or_api_key_needed(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = run(llm_for(route("offensive", "offensive"), SHOTS_OK, offensive=SHOTS))
    assert s["outcome"] == "verified"
    assert all(u["model"] == "mock-llm" for u in s["token_usage"])


def test_graph_module_never_touches_tools_or_llm_directly(tool_counter):
    src = inspect.getsource(GR)
    assert "registry" not in src and "call_tool" not in src and ".complete(" not in src
    s = run(llm_for(route("offensive", "offensive", "discipline"), SHOTS_OK, offensive=SHOTS, discipline=CARDS))
    assert sorted(tool_counter) == sorted(c["tool"] for c in s["tool_calls"])   # only analysts ran tools


def test_stream_yields_node_updates_in_order():
    app = GR.build_graph(llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS))
    steps = list(GR.stream(app, "Do teams that take more shots get more points?"))
    assert [n for n, _ in steps] == ["router", "analyst_offensive", "synthesiser", "guard", "retry",
                                     "synthesiser", "guard", "finalise"]
    assert steps[-1][1]["outcome"] == "verified"


def test_checkpointer_records_every_step_per_thread():
    app = GR.build_graph(llm_for(route("offensive", "offensive"), [SHOTS_BAD, SHOTS_OK], offensive=SHOTS),
                         checkpointer=InMemorySaver())
    s = GR.ask(app, "Do teams that take more shots get more points?", thread_id="t-1")
    snap = app.get_state({"configurable": {"thread_id": "t-1"}})
    assert snap.values["outcome"] == s["outcome"] == "verified" and snap.next == ()
    history = list(app.get_state_history({"configurable": {"thread_id": "t-1"}}))
    rejected = [h for h in history if h.next == ("retry",)]              # paused right after the failed check
    assert len(rejected) == 1                                            # audit trail of the rejected draft
    assert rejected[0].values["draft_answer"]["answer"] == SHOTS_BAD["answer"]
    assert rejected[0].values["grounding_result"]["passed"] is False


def test_docs_diagram_lists_every_graph_node():
    doc = Path(__file__).resolve().parents[2] / "docs" / "graph-workflow.md"
    text = doc.read_text(encoding="utf-8")
    assert "```mermaid" in text
    for node in GR.NODE_NAMES:
        assert node in text, f"{node} missing from {doc.name}"


def test_manual_demo_scenarios_end_as_described():
    from agents import demo
    expected = {"verified": ("verified", 0), "retry": ("verified", 1), "fallback": ("fallback", 1),
                "router_down": ("verified", 0), "two_analysts": ("verified", 0), "no_evidence": ("no_evidence", 0)}
    assert set(expected) == set(demo.SCENARIOS)
    for name, (outcome, retries) in expected.items():
        s = demo.run_scenario(name)
        assert (s["outcome"], s["retry_count"]) == (outcome, retries), name