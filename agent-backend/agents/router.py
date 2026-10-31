"""
Router node: decides WHAT kind of question this is and WHICH analysts should handle it.
It never calculates anything.

    make_router(llm) -> node(state) -> {"intent", "selected_agents", "plan", "token_usage", "errors"}

1. The LLM (the cheap model in production) classifies the question, using a prompt that
   lists each analyst's tools straight from tools.registry, so the router always sees
   the real capabilities.
2. Its JSON reply is validated. Unknown intents or analysts are never passed on.
3. If the LLM fails (rate limit, timeout, invalid JSON, invalid values), a deterministic
   keyword router takes over and the failure is recorded in `errors`. The question still
   gets answered, and the fallback rate becomes a metric we can track later.
"""
from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

from llm.base import LLMClient, LLMError, Message
from tools.registry import TOOLS

from .prompting import render
from .state import AGENTS, INTENTS, AgentError, AgentState, usage_record

# ---------------------------------------------------------------------------
# Which tools each analyst may call (validated against the registry in tests)
# ---------------------------------------------------------------------------
AGENT_TOOLS: dict[str, list[str]] = {
    "trend": ["team_season_stats", "league_table", "top_n", "season_trend", "team_comparison",
              "head_to_head", "most_improved", "home_advantage", "home_advantage_test",
              "hypothesis_test", "data_coverage"],
    "offensive": ["team_season_stats", "top_n", "season_trend", "team_comparison",
                  "correlation_analysis", "regression_analysis", "hypothesis_test", "data_coverage"],
    "discipline": ["team_season_stats", "top_n", "season_trend", "team_comparison",
                   "correlation_analysis", "regression_analysis", "hypothesis_test", "data_coverage"],
}

AGENT_DESCRIPTIONS = {
    "trend": "results over time: points, wins/draws/losses, league tables, improvement, home advantage",
    "offensive": "goals, shots, shots on target and how attacking output relates to points",
    "discipline": "fouls, yellow and red cards and how discipline relates to points",
}

# Analysts used when the router names an intent but no valid analysts.
DEFAULT_AGENTS = {
    "trend": ["trend"], "offensive": ["offensive"], "discipline": ["discipline"],
    "comparison": ["trend"], "hypothesis": ["trend"], "general_statistics": ["trend"],
}


def tool_catalog() -> str:
    """
    Analyst -> what it covers + its tool names, for the router prompt.
    The router picks analysts, not tools, so full tool descriptions are left to the analyst
    prompts. This keeps the router prompt (paid on every question) small.
    """
    lines = []
    for agent in AGENTS:
        missing = [t for t in AGENT_TOOLS[agent] if t not in TOOLS]
        if missing:
            raise KeyError(f"AGENT_TOOLS['{agent}'] lists tools not in the registry: {missing}")
        lines.append(f"- {agent}: {AGENT_DESCRIPTIONS[agent]}. Tools: {', '.join(AGENT_TOOLS[agent])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------
class RouterDecision(BaseModel):
    intent: str
    agents: list[str] = Field(default_factory=list)
    plan: dict[str, str] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in INTENTS:
            raise ValueError(f"unknown intent '{v}'")
        return v

    @field_validator("agents")
    @classmethod
    def _known_agents(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip().lower() for a in v]
        unknown = [a for a in cleaned if a not in AGENTS]
        if unknown:
            raise ValueError(f"unknown analysts {unknown}")
        return list(dict.fromkeys(cleaned))  # dedupe, keep order


def normalise_decision(d: RouterDecision, question: str) -> dict:
    """Fill gaps: default analysts for the intent, and a plan line for every selected analyst."""
    agents = d.agents or DEFAULT_AGENTS[d.intent]
    agents = [a for a in AGENTS if a in agents]  # canonical order: trend, offensive, discipline
    plan = {a: (d.plan.get(a) or "").strip() or question for a in agents}
    return {"intent": d.intent, "selected_agents": agents, "plan": plan}


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
def _rx(*words: str) -> re.Pattern:
    """Match any word/phrase at the start of a word, case-insensitive."""
    return re.compile("|".join(r"\b" + re.escape(w) for w in words), re.IGNORECASE)


_HYPOTHESIS = _rx("significant", "statistically", "hypothesis", "p-value", "p value", "covid",
                  "empty stadium", "behind closed doors", "null hypothesis")
_COMPARISON = _rx("compare", "comparison", " vs", "versus", "head to head", "head-to-head", "better than")
_DISCIPLINE = _rx("card", "yellow", "red card", "sent off", "sending off", "foul", "disciplin", "booking")
_OFFENSIVE = _rx("shot", "shoot", "on target", "goal", "scor", "attack", "offensive", "striker")
_TREND = _rx("trend", "over time", "over the years", "across seasons", "improv", "declin", "progress",
             "history", "since", "each season", "every season", "home advantage", "points per")


def rule_based_route(question: str) -> dict:
    """Keyword router used when the LLM is unavailable or returns something invalid."""
    q = question
    domains = [a for a, rx in (("trend", _TREND), ("offensive", _OFFENSIVE), ("discipline", _DISCIPLINE))
               if rx.search(q)]
    if _HYPOTHESIS.search(q):
        intent = "hypothesis"
    elif _COMPARISON.search(q):
        intent = "comparison"
    elif "discipline" in domains:
        intent = "discipline"
    elif "offensive" in domains:
        intent = "offensive"
    elif "trend" in domains:
        intent = "trend"
    else:
        intent = "general_statistics"
    # Specialists (offensive/discipline) can also do trends for their own metrics,
    # so the trend analyst is only used when no specialist topic is mentioned.
    specialists = [d for d in domains if d != "trend"]
    decision = RouterDecision(intent=intent, agents=specialists or DEFAULT_AGENTS[intent])
    return normalise_decision(decision, question)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def build_messages(question: str, prompt_version: str | None = None) -> list[Message]:
    system = render("router", prompt_version, tool_catalog=tool_catalog())
    return [Message("system", system), Message("user", f"Question: {question}")]


def make_router(llm: LLMClient, prompt_version: str | None = None) -> Callable[[AgentState], dict]:
    def router(state: AgentState) -> dict:
        question = state["question"]
        update: dict = {"token_usage": [], "errors": []}
        try:
            resp = llm.complete(build_messages(question, prompt_version), purpose="router",
                                json_mode=True, temperature=0.0)
        except LLMError as e:
            update["errors"].append(AgentError(node="router", kind="llm_error",
                                               message=f"{type(e).__name__}: {e}"))
            return {**update, **rule_based_route(question)}

        update["token_usage"].append(usage_record("router", resp))
        try:
            raw = resp.json()
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object")
            decision = RouterDecision(**raw)
        except (LLMError, ValidationError, ValueError, TypeError) as e:
            update["errors"].append(AgentError(node="router", kind="bad_output",
                                               message=f"{type(e).__name__}: {str(e)[:300]}"))
            return {**update, **rule_based_route(question)}
        return {**update, **normalise_decision(decision, question)}

    return router