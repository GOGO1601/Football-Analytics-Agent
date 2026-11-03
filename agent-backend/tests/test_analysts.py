"""
Stage 4: specialist analysts. MockLLM decides; the real registry tools compute on the real data.
No API key, no network.
"""
from __future__ import annotations

from operator import add

import pytest

import tools.registry as registry
from agents.analysts import (MAX_TOOL_CALLS, build_messages, make_analyst, make_analysts, tool_catalog,
                             tool_signature)
from agents.router import AGENT_TOOLS, make_router, rule_based_route
from agents.state import new_state
from llm.base import LLMRateLimitError, LLMTimeoutError
from llm.mock import MockLLM, count_tokens


def routed(question: str, agents: list[str] | None = None) -> dict:
    """A state as it looks after the router (deterministic route unless agents are given)."""
    s = new_state(question)
    route = rule_based_route(question)
    if agents:
        route = {**route, "selected_agents": agents, "plan": {a: question for a in agents}}
    s.update(route)
    return s


def call(tool: str, **arguments) -> dict:
    return {"tool": tool, "arguments": arguments}


def reply(*calls, reason="r") -> dict:
    return {"tool_calls": list(calls), "reason": reason}


@pytest.fixture
def spy(monkeypatch):
    """Record every registry.call_tool invocation while still running the real tool."""
    calls = []
    real = registry.call_tool

    def wrapper(name, args=None, **kw):
        calls.append((name, args))
        return real(name, args, **kw)
    monkeypatch.setattr(registry, "call_tool", wrapper)
    return calls


# ---------------------------------------------------------------------------
# Prompt and catalogue
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("agent", ["trend", "offensive", "discipline"])
def test_catalogue_lists_exactly_the_allowed_tools(agent):
    cat = tool_catalog(agent)
    listed = [line[2:].split("(")[0] for line in cat.splitlines() if line.startswith("- ")]
    assert listed == AGENT_TOOLS[agent]


def test_signature_comes_from_registry_schema():
    sig = tool_signature("correlation_analysis")
    for arg in registry.TOOLS["correlation_analysis"].args_model.model_fields:
        assert f"{arg}:" in sig
    assert sig.startswith("correlation_analysis(x_metric: str, y_metric: str = \"Points\"")
    assert "group_a: {league, season_start, season_end, teams}" in tool_signature("hypothesis_test")


@pytest.mark.parametrize("agent", ["trend", "offensive", "discipline"])
def test_prompt_is_filled_and_bounded(agent):
    system, user = build_messages(agent, "Who scored most?", "Find top scorers.", "v1")
    assert "$" not in system.content.replace("$$", "")           # every placeholder filled
    assert agent in system.content and "AvgYellowCards" in system.content and "Premier League" in system.content
    assert "Who scored most?" not in system.content              # user text stays out of the system prompt
    assert user.content == "Question: Who scored most?\nYour task from the router: Find top scorers."
    assert count_tokens(system.content) < 1500


def test_unknown_agent_rejected():
    with pytest.raises(ValueError):
        make_analyst("referee", MockLLM())


# ---------------------------------------------------------------------------
# Router -> analyst -> registry -> result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,agent,tool_call", [
    ("How have Liverpool's points changed over the years?", "trend",
     call("season_trend", team="Liverpool", metric="Points")),
    ("Do teams that take more shots get more points?", "offensive",
     call("correlation_analysis", x_metric="AvgShots")),
    ("Do yellow cards hurt league position?", "discipline",
     call("correlation_analysis", x_metric="AvgYellowCards", require_discipline_fields=True)),
])
def test_question_reaches_the_right_analyst_and_tool_runs(question, agent, tool_call, spy):
    llm = MockLLM({f"analyst:{agent}": reply(tool_call)})
    state = routed(question)
    assert state["selected_agents"] == [agent]
    nodes = make_analysts(llm)
    updates = {a: node(state) for a, node in nodes.items()}
    # only the selected analyst talked to the LLM and ran a tool
    assert [c.purpose for c in llm.calls] == [f"analyst:{agent}"]
    assert spy == [(tool_call["tool"], tool_call["arguments"])]
    out = updates[agent]
    assert out["tool_calls"][0] == {"id": f"{agent}-1", "agent": agent, "tool": tool_call["tool"],
                                    "args": tool_call["arguments"]}
    res = out["tool_results"][0]
    assert res["ok"] and res["call_id"] == f"{agent}-1" and res["error"] is None
    assert res["result"] == registry.call_tool(tool_call["tool"], tool_call["arguments"])["result"]
    assert out["token_usage"][0]["node"] == f"analyst:{agent}" and out["errors"] == []


def test_end_to_end_with_mock_router():
    llm = MockLLM({"router": {"intent": "offensive", "agents": ["offensive"],
                              "plan": {"offensive": "Correlate shots per game with points."}},
                   "analyst:offensive": reply(call("correlation_analysis", x_metric="AvgShots",
                                                   league="Premier League"))})
    state = new_state("Do teams that take more shots get more points in the Premier League?")
    state.update(make_router(llm)(state))
    out = make_analyst("offensive", llm)(state)
    r = out["tool_results"][0]["result"]
    assert r["league"] == "Premier League" and r["x_metric"] == "AvgShots" and 0 < r["r"] < 1
    prompt = llm.calls_for("analyst:offensive")[0].request.last_user
    assert "Correlate shots per game with points." in prompt     # router plan reached the analyst


def test_unselected_analyst_does_nothing():
    llm = MockLLM()  # any LLM call would raise MockScriptError
    out = make_analyst("discipline", llm)(routed("Arsenal points over time"))
    assert out == {"tool_calls": [], "tool_results": [], "token_usage": [], "errors": []}


# ---------------------------------------------------------------------------
# Tool restrictions: nothing outside AGENT_TOOLS ever runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("agent,tool,code", [
    ("offensive", "league_table", "tool_not_allowed"),       # real tool, but trend-only
    ("offensive", "home_advantage_test", "tool_not_allowed"),
    ("discipline", "most_improved", "tool_not_allowed"),
    ("discipline", "os.system", "unknown_tool"),
    ("trend", "eval", "unknown_tool"),
    ("trend", "__import__", "unknown_tool"),
    ("trend", "call_tool", "unknown_tool"),
])
def test_disallowed_tools_never_execute(agent, tool, code, spy):
    llm = MockLLM({f"analyst:{agent}": reply(call(tool))})
    out = make_analyst(agent, llm)(routed("q", [agent]))
    assert spy == [] and out["tool_results"] == [] and out["tool_calls"] == []
    assert any(e["kind"] == code for e in out["errors"])
    assert len(llm.calls) == 2                                   # original + one repair attempt


def test_mixed_valid_and_invalid_calls_runs_only_valid(spy):
    llm = MockLLM({"analyst:offensive": reply(call("league_table", league="La Liga", season="2010-11"),
                                              call("top_n", metric="GoalsFor", n=3))})
    out = make_analyst("offensive", llm)(routed("q", ["offensive"]))
    assert spy == [("top_n", {"metric": "GoalsFor", "n": 3})]
    assert [e["kind"] for e in out["errors"]] == ["tool_not_allowed"]
    assert len(llm.calls) == 1                                   # something ran, so no repair turn


# ---------------------------------------------------------------------------
# Argument validation happens BEFORE execution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_call", [
    call("correlation_analysis", metric_x="shots", metric_y="points"),   # wrong argument names
    call("correlation_analysis"),                                         # missing required x_metric
    call("top_n", metric="Points", n="lots"),                             # wrong type
    call("hypothesis_test", metric="Points", group_a={"leage": "Serie A"}, group_b={"league": "La Liga"}),
])
def test_invalid_arguments_rejected_before_execution(bad_call, spy):
    agent = "offensive"
    llm = MockLLM({f"analyst:{agent}": reply(bad_call)})
    out = make_analyst(agent, llm)(routed("q", [agent]))
    assert spy == [] and out["tool_results"] == []
    assert any(e["kind"] == "invalid_arguments" for e in out["errors"])


def test_repair_turn_fixes_bad_arguments(spy):
    llm = MockLLM({"analyst:offensive": [reply(call("correlation_analysis", metric_x="shots")),
                                         reply(call("correlation_analysis", x_metric="AvgShots"))]})
    out = make_analyst("offensive", llm)(routed("Do shots matter?", ["offensive"]))
    assert spy == [("correlation_analysis", {"x_metric": "AvgShots"})]
    assert out["tool_results"][0]["ok"] and len(out["token_usage"]) == 2
    repair_prompt = llm.calls[1].request.messages[-1].content
    assert "metric_x" in repair_prompt and "x_metric: str" in repair_prompt   # told exactly what was wrong


def test_args_alias_accepted():
    llm = MockLLM({"analyst:trend": {"tool_calls": [{"tool": "data_coverage", "args": {"league": "Serie A"}}]}})
    out = make_analyst("trend", llm)(routed("q", ["trend"]))
    assert out["tool_results"][0]["ok"]


# ---------------------------------------------------------------------------
# Multiple tools / multiple analysts
# ---------------------------------------------------------------------------
def test_multiple_tools_in_one_analyst(spy):
    llm = MockLLM({"analyst:trend": reply(call("team_season_stats", team="Arsenal", season="2003-04"),
                                          call("league_table", league="Premier League", season="2003-04"))})
    out = make_analyst("trend", llm)(routed("Arsenal 2003-04 and the table", ["trend"]))
    assert [c["id"] for c in out["tool_calls"]] == ["trend-1", "trend-2"]
    assert [t for t, _ in spy] == ["team_season_stats", "league_table"]
    assert out["tool_results"][0]["result"]["rows"][0]["Points"] == 90
    assert out["tool_results"][1]["result"]["table"][0]["Team"] == "Arsenal"


def test_duplicate_calls_run_once(spy):
    c = call("top_n", metric="Points", n=1)
    out = make_analyst("trend", MockLLM({"analyst:trend": reply(c, c)}))(routed("q", ["trend"]))
    assert len(spy) == 1 and len(out["tool_results"]) == 1


def test_tool_call_cap(spy):
    calls = [call("top_n", metric="Points", n=i) for i in range(1, MAX_TOOL_CALLS + 3)]
    out = make_analyst("trend", MockLLM({"analyst:trend": reply(*calls)}))(routed("q", ["trend"]))
    assert len(spy) == MAX_TOOL_CALLS
    assert any(e["kind"] == "too_many_tool_calls" for e in out["errors"])


def test_offensive_and_discipline_run_independently_and_merge(spy):
    question = "Compare Arsenal and Chelsea on shots and yellow cards"
    state = routed(question)
    assert state["selected_agents"] == ["offensive", "discipline"]
    llm = MockLLM({
        "analyst:offensive": reply(call("team_comparison", teams=["Arsenal", "Chelsea"], metric="AvgShots")),
        "analyst:discipline": reply(call("team_comparison", teams=["Arsenal", "Chelsea"], metric="AvgYellowCards")),
    })
    nodes = make_analysts(llm)
    off, disc = nodes["offensive"](state), nodes["discipline"](state)
    # what LangGraph's operator.add reducers do with two parallel updates
    merged = {k: add(add(state[k], off[k]), disc[k]) for k in ("tool_calls", "tool_results", "token_usage", "errors")}
    assert [r["call_id"] for r in merged["tool_results"]] == ["offensive-1", "discipline-1"]
    assert {r["result"]["metric"] for r in merged["tool_results"]} == {"AvgShots", "AvgYellowCards"}
    assert all(r["ok"] for r in merged["tool_results"]) and len(merged["token_usage"]) == 2
    # independence: neither analyst saw the other's output
    for c in llm.calls:
        assert "offensive-1" not in c.request.text and "discipline-1" not in c.request.text


# ---------------------------------------------------------------------------
# Hypothesis questions reach the hypothesis tools
# ---------------------------------------------------------------------------
def test_covid_home_advantage_uses_home_advantage_test(spy):
    question = "Was home advantage significantly lower during COVID in Serie A?"
    state = routed(question)
    assert state["intent"] == "hypothesis" and state["selected_agents"] == ["trend"]
    llm = MockLLM({"analyst:trend": reply(call("home_advantage_test", league="Serie A"))})
    out = make_analyst("trend", llm)(state)
    assert spy == [("home_advantage_test", {"league": "Serie A"})]
    t = out["tool_results"][0]["result"]["tests"]["home_win_rate_two_proportion_z"]
    assert 0 <= t["p_value"] <= 1 and "ci95" in t and "cohens_h" in t


def test_group_comparison_uses_hypothesis_test(spy):
    llm = MockLLM({"analyst:discipline": reply(call(
        "hypothesis_test", metric="AvgYellowCards",
        group_a={"league": "Premier League", "season_start": "2010-11"},
        group_b={"league": "La Liga", "season_start": "2010-11"}))})
    out = make_analyst("discipline", llm)(routed("Do La Liga teams get significantly more yellow cards?",
                                                 ["discipline"]))
    res = out["tool_results"][0]
    assert res["ok"] and spy[0][0] == "hypothesis_test"
    assert res["result"]["tests"]["welch_t"]["p_value"] < 0.05


# ---------------------------------------------------------------------------
# LLM and tool failures never crash the analyst
# ---------------------------------------------------------------------------
def test_malformed_json_twice_gives_structured_error(spy):
    llm = MockLLM({"analyst:trend": ["not json", "still not json"]})
    out = make_analyst("trend", llm)(routed("q", ["trend"]))
    assert spy == [] and out["tool_results"] == []
    assert [e["kind"] for e in out["errors"]] == ["bad_output", "bad_output"]
    assert len(out["token_usage"]) == 2


def test_malformed_json_then_valid_recovers():
    llm = MockLLM({"analyst:trend": ["oops", reply(call("top_n", n=1))]})
    out = make_analyst("trend", llm)(routed("q", ["trend"]))
    assert out["tool_results"][0]["ok"] and [e["kind"] for e in out["errors"]] == ["bad_output"]


@pytest.mark.parametrize("exc,kind", [(LLMTimeoutError("slow"), "llm_error"), (LLMRateLimitError("429"), "llm_error")])
def test_llm_errors(exc, kind, spy):
    out = make_analyst("offensive", MockLLM({"analyst:offensive": exc}))(routed("q", ["offensive"]))
    assert spy == [] and out["tool_results"] == [] and out["token_usage"] == []
    assert out["errors"][0]["kind"] == kind and out["errors"][0]["node"] == "analyst:offensive"


def test_no_tool_can_answer():
    llm = MockLLM({"analyst:offensive": reply(reason="The data has no xG.")})
    out = make_analyst("offensive", llm)(routed("What was Arsenal's xG?", ["offensive"]))
    assert out["tool_results"] == [] and out["errors"][0] == {"node": "analyst:offensive", "kind": "no_tool_calls",
                                                            "message": "The data has no xG."}


def test_tool_level_error_is_recorded_not_raised():
    llm = MockLLM({"analyst:trend": reply(call("team_season_stats", team="Arsenl"))})
    out = make_analyst("trend", llm)(routed("q", ["trend"]))
    res = out["tool_results"][0]
    assert not res["ok"] and res["error"]["code"] == "unknown_team" and "Arsenal" in res["error"]["suggestions"]
    assert out["errors"][0]["kind"] == "tool_error"


def test_unexpected_tool_exception_is_contained(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug in tool")
    monkeypatch.setattr(registry, "call_tool", boom)
    out = make_analyst("trend", MockLLM({"analyst:trend": reply(call("top_n"))}))(routed("q", ["trend"]))
    assert out["tool_results"][0]["error"]["code"] == "internal_error"


# ---------------------------------------------------------------------------
# The analyst never computes: every number comes from the registry
# ---------------------------------------------------------------------------
def test_results_come_only_from_registry(monkeypatch):
    sentinel = {"ok": True, "tool": "correlation_analysis", "result": {"r": 0.123456789, "n": 42}}
    seen = []
    monkeypatch.setattr(registry, "call_tool", lambda name, args=None, **kw: seen.append(name) or sentinel)
    llm = MockLLM({"analyst:offensive": reply(call("correlation_analysis", x_metric="AvgShots"))})
    out = make_analyst("offensive", llm)(routed("q", ["offensive"]))
    assert seen == ["correlation_analysis"]
    assert out["tool_results"][0]["result"] == {"r": 0.123456789, "n": 42}   # passed through untouched