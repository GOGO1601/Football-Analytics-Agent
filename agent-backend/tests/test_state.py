"""Stage 2: the graph state. No LangGraph needed yet; reducers are checked directly."""
from __future__ import annotations

import json
from operator import add
from typing import get_args, get_origin, get_type_hints

import pytest

from agents.state import (AGENTS, DEFAULT_MAX_RETRIES, INTENTS, AgentName, AgentState, Intent,
                          new_state, successful_results, usage_record, usage_summary)
from llm.base import Message
from llm.mock import MockLLM


def test_new_state_defaults():
    s = new_state("  Who won   the league in 2004-05? ")
    assert s["question"] == "Who won the league in 2004-05?"     # whitespace normalised
    assert s["max_retries"] == DEFAULT_MAX_RETRIES and s["retry_count"] == 0
    assert s["tool_calls"] == [] and s["tool_results"] == [] and s["token_usage"] == [] and s["errors"] == []
    assert s["draft_answer"] is None and s["grounding_result"] is None and s["chart_actions"] == []


@pytest.mark.parametrize("bad", ["", "   ", None, 42, "x" * 1001])
def test_new_state_rejects_bad_questions(bad):
    with pytest.raises(ValueError):
        new_state(bad)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_new_state_rejects_bad_max_retries(bad):
    with pytest.raises(ValueError):
        new_state("q", max_retries=bad)


def test_intent_and_agent_constants_match_type_literals():
    assert set(INTENTS) == set(get_args(Intent))
    assert set(AGENTS) == set(get_args(AgentName))
    assert {"trend", "offensive", "discipline", "comparison", "hypothesis", "general_statistics"} == set(INTENTS)


def test_parallel_keys_have_add_reducers():
    """LangGraph rejects concurrent writes to a key without a reducer; analysts run in parallel."""
    hints = get_type_hints(AgentState, include_extras=True)
    for key in ("tool_calls", "tool_results", "token_usage", "errors"):
        meta = getattr(hints[key], "__metadata__", ())
        assert add in meta, f"{key} needs an operator.add reducer"
    for key in ("question", "intent", "draft_answer", "retry_count", "grounding_result"):
        assert not getattr(hints[key], "__metadata__", ()), f"{key} should be overwritten, not accumulated"


def test_reducer_merges_parallel_analyst_updates():
    """Simulate what LangGraph does when two analysts return tool results in the same step."""
    s = new_state("shots and cards vs points")
    trend_update = {"tool_results": [{"call_id": "trend-1", "agent": "trend", "tool": "top_n", "args": {},
                                      "ok": True, "result": {"x": 1}, "error": None}]}
    disc_update = {"tool_results": [{"call_id": "discipline-1", "agent": "discipline", "tool": "top_n",
                                     "args": {}, "ok": False, "result": None,
                                     "error": {"code": "unknown_team", "message": "m", "suggestions": []}}]}
    merged = add(add(s["tool_results"], trend_update["tool_results"]), disc_update["tool_results"])
    assert [r["call_id"] for r in merged] == ["trend-1", "discipline-1"]
    assert [r["call_id"] for r in successful_results({"tool_results": merged})] == ["trend-1"]


def test_usage_record_and_summary_from_mock_llm():
    llm = MockLLM({"router": {"intent": "trend"}, "synthesiser": "answer"})
    r1 = llm.complete([Message("user", "q")], purpose="router")
    r2 = llm.complete([Message("user", "q")], purpose="synthesiser")
    r3 = llm.complete([Message("user", "q again")], purpose="synthesiser")
    s = new_state("q")
    s["token_usage"] = [usage_record("router", r1), usage_record("synthesiser", r2), usage_record("synthesiser", r3)]
    summ = usage_summary(s)
    assert summ["calls"] == 3
    assert summ["total_tokens"] == llm.total_usage.total_tokens
    assert summ["per_node"]["synthesiser"]["calls"] == 2
    assert summ["llm_latency_ms"] == r1.latency_ms + r2.latency_ms + r3.latency_ms


def test_usage_summary_on_empty_state():
    assert usage_summary(new_state("q"))["total_tokens"] == 0


def test_state_is_json_serialisable():
    s = new_state("q")
    s.update(intent="offensive", selected_agents=["offensive"], plan={"offensive": "shots vs points"},
             draft_answer={"answer": "a", "evidence": ["e"], "key_stats": [{"label": "r", "value": "0.597",
                                                                             "source": "offensive-1"}],
                           "explanation": "x"},
             chart_actions=[{"page": "q2", "x_metric": "AvgShots", "league": "all"}],
             grounding_result={"passed": True, "checked": 1, "unsupported": [], "feedback": ""})
    assert json.loads(json.dumps(s)) == s