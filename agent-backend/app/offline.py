"""
The API's default "LLM": a deterministic, offline MockLLM, so the server runs with no API key,
no network and no cost. It is NOT a language model and it never computes a statistic:

  router       routes with the existing rule-based router (agents.router.rule_based_route)
  analysts     pick ONE tool call from a few keyword rules; the analyst still validates it and
               runs it through the tool registry exactly as with a real LLM
  synthesiser  writes the answer from the tools' own `interpretation` text (no numbers of its own);
               the grounding guard checks it like any other draft

So offline answers are only as good as these keyword rules: fine for trying the API, not a
replacement for the real LLM client (a later stage).

OfflineLLM is MockLLM without the call log: MockLLM records every call for test assertions,
which would grow without limit in a long-running server.
"""
from __future__ import annotations

import json
import re

from agents.router import rule_based_route
from llm.mock import MockLLM, MockRequest
from tools.common import LEAGUES

MODEL_NAME = "offline-mock"
OFFLINE_NOTE = ("Offline mode: no language model is configured, so this answer is assembled from the "
                "analysis tools' own summaries.")


class OfflineLLM(MockLLM):
    """Same scripting as MockLLM, but keeps no call log (safe for a long-running process)."""
    def complete(self, *args, **kwargs):
        try:
            return super().complete(*args, **kwargs)
        finally:
            with self._lock:
                self.calls.clear()


def _question(req: MockRequest) -> str:
    first = req.last_user.split("\n", 1)[0]
    return first.removeprefix("Question:").strip()


def _league(q: str) -> str | None:
    return next((lg for lg in LEAGUES if lg.lower() in q.lower()), None)


def _season(q: str) -> str | None:
    m = re.search(r"\b((?:19|20)\d{2})\s*[-/]\s*(\d{2}|\d{4})\b", q)
    return f"{m.group(1)}-{m.group(2)[-2:]}" if m else None


# (analyst, pattern, tool, args) in priority order; each analyst only uses its own topic's rules,
# and the league is added when the question names one.
_RULES: list[tuple[str, str, str, dict]] = [
    ("trend", r"\bcovid|\bhome advantage|\bhome win", "home_advantage_test", {}),
    ("offensive", r"\bon target", "correlation_analysis", {"x_metric": "AvgShotsOnTarget"}),
    ("offensive", r"\bshot|\bshoot", "correlation_analysis", {"x_metric": "AvgShots"}),
    ("offensive", r"\bgoal|\bscor", "correlation_analysis", {"x_metric": "GoalsFor"}),
    ("discipline", r"\bred card|\bsent off|\bsending off", "correlation_analysis", {"x_metric": "AvgRedCards"}),
    ("discipline", r"\byellow|\bcard|\bbooking", "correlation_analysis", {"x_metric": "AvgYellowCards"}),
    ("discipline", r"\bfoul", "correlation_analysis", {"x_metric": "AvgFouls"}),
]
_DEFAULT_X = {"offensive": "AvgShots", "discipline": "AvgYellowCards"}


def choose_tool(agent: str, question: str) -> dict:
    """One tool call for this analyst, always one it is allowed to use."""
    league, season = _league(question), _season(question)
    where = {"league": league} if league else {}
    for owner, pattern, tool, args in _RULES:
        if owner == agent and re.search(pattern, question, re.IGNORECASE):
            return {"tool": tool, "arguments": {**args, **where}}
    if agent in _DEFAULT_X:
        return {"tool": "correlation_analysis", "arguments": {"x_metric": _DEFAULT_X[agent], **where}}
    if league and season:
        return {"tool": "league_table", "arguments": {"league": league, "season": season}}
    return {"tool": "data_coverage", "arguments": where}


def _router(req: MockRequest) -> dict:
    d = rule_based_route(_question(req))
    return {"intent": d["intent"], "agents": d["selected_agents"], "reason": "offline keyword routing"}


def _analyst(req: MockRequest) -> dict:
    agent = req.purpose.split(":", 1)[-1]
    return {"tool_calls": [choose_tool(agent, _question(req))], "reason": "offline keyword rules"}


def _synthesiser(req: MockRequest) -> dict:
    m = re.search(r"<evidence>\n(.*)\n</evidence>", req.last_user, re.DOTALL)
    evidence = json.loads(m.group(1)) if m else {}
    lines = []
    for r in evidence.get("tool_results", []):
        text = (r.get("result") or {}).get("interpretation") if r.get("ok") else None
        if text:
            lines.append(text)
    answer = " ".join(lines[:2]) or "The tool results are returned below without a written summary."
    return {"answer": answer, "evidence": lines, "key_stats": [], "explanation": OFFLINE_NOTE,
            "chart_actions": []}


def offline_llm() -> OfflineLLM:
    return OfflineLLM({"router": _router, "analyst": _analyst, "synthesiser": _synthesiser}, model=MODEL_NAME)