"""
Run the full graph on a few example questions with scripted MockLLMs (no API key, no network).
The LLM replies are scripted; every number comes from the real tools on the real data.

    python -m agents.demo                 # all scenarios
    python -m agents.demo retry           # one scenario by name
    python -m agents.demo --json          # print the final states as JSON

This is a manual check only. Real LLM clients arrive in a later stage.
"""
from __future__ import annotations

import json
import sys

from llm.base import LLMRateLimitError
from llm.mock import MockLLM

from .graph import ask, build_graph
from .state import usage_summary

SHOTS_CALL = {"tool_calls": [{"tool": "correlation_analysis", "arguments": {"x_metric": "AvgShots"}}]}
SHOTS_DRAFT = {"answer": "Teams that take more shots tend to earn more points (r = 0.597); correlation does not "
                         "establish causation.",
               "evidence": ["Pearson correlation across all leagues and seasons with shot data."],
               "key_stats": [{"label": "Pearson r, shots vs points", "value": "0.597", "source": "offensive-1"}],
               "explanation": "Association only.", "chart_actions": [{"page": "q2", "x_metric": "AvgShots",
                                                                       "league": "all"}]}
SHOTS_BAD = {**SHOTS_DRAFT, "answer": "Taking more shots causes more points (r = 0.9)."}

SCENARIOS: dict[str, tuple[str, dict]] = {
    "verified": ("Do teams that take more shots get more points?", {
        "router": {"intent": "offensive", "agents": ["offensive"]},
        "analyst:offensive": SHOTS_CALL,
        "synthesiser": SHOTS_DRAFT}),
    "retry": ("Do teams that take more shots get more points?", {
        "router": {"intent": "offensive", "agents": ["offensive"]},
        "analyst:offensive": SHOTS_CALL,
        "synthesiser": [SHOTS_BAD, SHOTS_DRAFT]}),           # 1st draft rejected, 2nd accepted
    "fallback": ("Do teams that take more shots get more points?", {
        "router": {"intent": "offensive", "agents": ["offensive"]},
        "analyst:offensive": SHOTS_CALL,
        "synthesiser": [SHOTS_BAD, SHOTS_BAD]}),             # both rejected -> tools' own summaries
    "router_down": ("Was home advantage significantly lower during COVID in Serie A?", {
        "router": LLMRateLimitError("429 from the provider"),  # rule-based routing takes over
        "analyst:trend": {"tool_calls": [{"tool": "home_advantage_test", "arguments": {"league": "Serie A"}}]},
        "synthesiser": {"answer": "No. Serie A's home win rate fell from 45.4% to 40.8% in 2020-21, but the change "
                                  "is not statistically significant (p = 0.0841).", "evidence": [],
                        "key_stats": [{"label": "p-value", "value": "0.0841", "source": "trend-1"}],
                        "explanation": "Two-proportion z-test, 2010-11 to 2018-19 against 2020-21."}}),
    "two_analysts": ("Do shots and yellow cards relate to points?", {
        "router": {"intent": "offensive", "agents": ["offensive", "discipline"]},
        "analyst:offensive": SHOTS_CALL,
        "analyst:discipline": {"tool_calls": [{"tool": "correlation_analysis",
                                               "arguments": {"x_metric": "AvgYellowCards"}}]},
        "synthesiser": {"answer": "More shots go with more points (r = 0.597), more yellow cards with slightly "
                                  "fewer (r = -0.207); neither establishes causation.", "evidence": [],
                        "key_stats": [{"label": "r, shots", "value": "0.597", "source": "offensive-1"},
                                      {"label": "r, yellow cards", "value": "-0.207", "source": "discipline-1"}],
                        "explanation": "Pearson correlations across all team-seasons."}}),
    "no_evidence": ("How did Arsenl do in 2003-04?", {
        "router": {"intent": "trend", "agents": ["trend"]},
        "analyst:trend": {"tool_calls": [{"tool": "team_season_stats",
                                          "arguments": {"team": "Arsenl", "season": "2003-04"}}]}}),
}


def run_scenario(name: str) -> dict:
    question, script = SCENARIOS[name]
    return ask(build_graph(MockLLM(script)), question)


def show(name: str, s: dict) -> None:
    g = s["grounding_result"] or {}
    print(f"\n=== {name}: {s['question']}")
    print(f"route      {s['intent']} -> {', '.join(s['selected_agents'])}")
    for r in s["tool_results"]:
        print(f"tool       {r['call_id']}: {r['tool']}({json.dumps(r['args'])}) ok={r['ok']}")
    print(f"path       {' -> '.join(t['node'] for t in s['node_trace'])}")
    print(f"outcome    {s['outcome']}  (grounding passed={g.get('passed')}, retries={s['retry_count']})")
    print(f"answer     {s['draft_answer']['answer']}")
    for line in s["draft_answer"]["evidence"][:3]:
        print(f"  evidence {line[:140]}")
    for e in s["errors"]:
        print(f"  error    [{e['node']}] {e['kind']}: {e['message'][:120]}")
    u = usage_summary(s)
    ms = sum(t["ms"] for t in s["node_trace"])
    print(f"cost       {u['calls']} LLM calls, {u['total_tokens']} tokens (mock), {ms:.0f} ms in nodes")


def main(argv: list[str]) -> None:
    names = [a for a in argv if not a.startswith("--")] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
    for n in names:
        s = run_scenario(n)
        print(json.dumps(s, indent=2)) if "--json" in argv else show(n, s)


if __name__ == "__main__":
    main(sys.argv[1:])