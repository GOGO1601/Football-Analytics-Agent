"""Stage 3: the router. All LLM behaviour comes from MockLLM; no key, no network."""
from __future__ import annotations

import pytest

from agents import prompting
from agents.router import (AGENT_TOOLS, DEFAULT_AGENTS, build_messages, make_router, rule_based_route,
                           tool_catalog)
from agents.state import AGENTS, INTENTS, new_state
from llm.base import LLMRateLimitError, LLMTimeoutError
from llm.mock import MockLLM
from tools.registry import TOOLS


def run(llm, question):
    return make_router(llm)(new_state(question))


# ---------------------------------------------------------------------------
# Catalogue and prompt
# ---------------------------------------------------------------------------
def test_agent_tools_exist_in_registry():
    for agent, tools in AGENT_TOOLS.items():
        assert agent in AGENTS
        for t in tools:
            assert t in TOOLS, f"{agent} lists unknown tool {t}"


def test_every_registry_tool_is_reachable_by_some_agent():
    reachable = {t for tools in AGENT_TOOLS.values() for t in tools}
    assert reachable == set(TOOLS)


def test_hypothesis_tools_available_to_all_analysts():
    assert all("hypothesis_test" in AGENT_TOOLS[a] for a in AGENTS)
    assert "home_advantage_test" in AGENT_TOOLS["trend"]


def test_default_agents_cover_every_intent():
    assert set(DEFAULT_AGENTS) == set(INTENTS)


def test_prompt_contains_catalogue_and_question():
    msgs = build_messages("Do shots win games?", "v1")
    system, user = msgs[0].content, msgs[1].content
    assert msgs[0].role == "system" and msgs[1].role == "user"
    assert "correlation_analysis" in system and "home_advantage_test" in system
    assert "$tool_catalog" not in system           # placeholder was filled
    assert all(i in system for i in INTENTS)
    assert user == "Question: Do shots win games?"


def test_prompt_errors_are_explicit():
    with pytest.raises(prompting.PromptError, match="not found"):
        prompting.render("router", "v999", tool_catalog="x")
    with pytest.raises(prompting.PromptError, match="needs a value"):
        prompting.render("router", "v1")
    with pytest.raises(prompting.PromptError, match="no section"):
        prompting.render("nope", "v1")


def test_catalogue_lists_every_agent_and_its_tools():
    cat = tool_catalog()
    for agent, tools in AGENT_TOOLS.items():
        line = next(l for l in cat.splitlines() if l.startswith(f"- {agent}:"))
        assert all(t in line for t in tools)


def test_router_prompt_stays_small():
    """Paid on every question: keep it well under 1,000 tokens (about 4 chars per token)."""
    from llm.mock import count_tokens
    assert count_tokens(build_messages("q", "v1")[0].content) < 1000


# ---------------------------------------------------------------------------
# LLM decisions (scripted)
# ---------------------------------------------------------------------------
def test_valid_llm_decision_is_used():
    llm = MockLLM({"router": {"intent": "offensive", "agents": ["offensive"],
                              "plan": {"offensive": "Correlate shots with points."}, "reason": "r"}})
    out = run(llm, "Do teams that shoot more get more points?")
    assert out["intent"] == "offensive" and out["selected_agents"] == ["offensive"]
    assert out["plan"] == {"offensive": "Correlate shots with points."}
    assert out["errors"] == []
    assert out["token_usage"][0]["node"] == "router" and out["token_usage"][0]["prompt_tokens"] > 0


def test_multiple_agents_are_canonically_ordered_and_deduplicated():
    llm = MockLLM({"router": {"intent": "comparison", "agents": ["discipline", "offensive", "discipline"]}})
    out = run(llm, "Compare shots and cards for Arsenal and Chelsea")
    assert out["selected_agents"] == ["offensive", "discipline"]


def test_missing_agents_fall_back_to_intent_defaults():
    llm = MockLLM({"router": {"intent": "hypothesis"}})
    out = run(llm, "Is home advantage significantly lower during COVID?")
    assert out["selected_agents"] == ["trend"]
    assert out["plan"] == {"trend": "Is home advantage significantly lower during COVID?"}


def test_plan_is_restricted_to_selected_agents():
    llm = MockLLM({"router": {"intent": "offensive", "agents": ["offensive"],
                              "plan": {"offensive": "x", "discipline": "should be dropped"}}})
    assert run(llm, "q")["plan"] == {"offensive": "x"}


def test_case_and_whitespace_tolerated():
    llm = MockLLM({"router": {"intent": " Discipline ", "agents": ["DISCIPLINE"]}})
    out = run(llm, "red cards")
    assert out["intent"] == "discipline" and out["selected_agents"] == ["discipline"]


# ---------------------------------------------------------------------------
# Failures -> deterministic fallback, recorded in errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reply", [
    "not json at all",
    ["a", "list"],
    {"intent": "weather", "agents": ["trend"]},          # unknown intent
    {"intent": "trend", "agents": ["referee"]},          # unknown analyst
    {"agents": ["trend"]},                               # missing intent
])
def test_bad_output_falls_back(reply):
    out = run(MockLLM({"router": reply}), "How many yellow cards did Chelsea get in 2010-11?")
    assert out["intent"] == "discipline" and out["selected_agents"] == ["discipline"]
    assert out["errors"][0]["node"] == "router" and out["errors"][0]["kind"] == "bad_output"
    assert len(out["token_usage"]) == 1                  # the failed call still cost tokens


@pytest.mark.parametrize("exc", [LLMRateLimitError("429"), LLMTimeoutError("slow")])
def test_llm_error_falls_back(exc):
    out = run(MockLLM({"router": exc}), "Do shots lead to points?")
    assert out["intent"] == "offensive"
    assert out["errors"][0]["kind"] == "llm_error" and out["token_usage"] == []


def test_router_never_returns_unknown_values():
    """Whatever the model says, downstream only ever sees known intents and analysts."""
    replies = ["", "{}", '{"intent": null}', {"intent": "trend", "agents": "trend"},
               {"intent": "trend", "agents": [], "plan": "not a dict"}]
    for reply in replies:
        out = run(MockLLM({"router": reply}), "Arsenal points over time")
        assert out["intent"] in INTENTS
        assert out["selected_agents"] and set(out["selected_agents"]) <= set(AGENTS)
        assert set(out["plan"]) == set(out["selected_agents"])


# ---------------------------------------------------------------------------
# Rule-based router on realistic questions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,intent,agents", [
    ("Do teams that take more shots get more points?", "offensive", ["offensive"]),
    ("How many goals has Arsenal scored?", "offensive", ["offensive"]),
    ("Do yellow cards hurt league position?", "discipline", ["discipline"]),
    ("Which team committed the most fouls in Serie A 2015-16?", "discipline", ["discipline"]),
    ("How have Liverpool's points changed over the years?", "trend", ["trend"]),
    ("Which teams improved the most in La Liga in 2023-24?", "trend", ["trend"]),
    ("Compare Arsenal and Tottenham since 2010", "comparison", ["trend"]),
    ("Compare Arsenal vs Chelsea yellow cards and shots", "comparison", ["offensive", "discipline"]),
    ("Was home advantage significantly lower during COVID?", "hypothesis", ["trend"]),
    ("Is there a statistically significant link between fouls and points?", "hypothesis", ["discipline"]),
    ("Who won the Premier League in 2004-05?", "general_statistics", ["trend"]),
    ("Arsenal goals since 2010", "offensive", ["offensive"]),
])
def test_rule_based_route(question, intent, agents):
    out = rule_based_route(question)
    assert (out["intent"], out["selected_agents"]) == (intent, agents)
    assert set(out["plan"]) == set(agents)


def test_red_does_not_match_scored():
    assert rule_based_route("How many goals were scored?")["selected_agents"] == ["offensive"]


def test_router_does_not_call_tools_or_compute(monkeypatch):
    """The router decides; it must never execute a tool."""
    import tools.registry as reg
    monkeypatch.setattr(reg, "call_tool", lambda *a, **k: pytest.fail("router called a tool"))
    run(MockLLM({"router": {"intent": "trend"}}), "Arsenal points")
    rule_based_route("Arsenal points")