"""
Synthesiser node: turns the analysts' structured tool results into the answer the user reads.
It is an explanation layer, not a statistical engine: it never calls a tool and never
computes a statistic. Every number it may use is already in state["tool_results"].

    make_synthesiser(llm) -> node(state) -> {"draft_answer", "chart_actions", "token_usage", "errors"}

How it works
  1. Package the evidence: every tool result (successes AND failures) plus analyst issues
     such as "no tool can answer this", as compact JSON. Floats are kept to 6 significant
     figures; if the package is too long, long row lists are trimmed and the trimming is
     stated inside the evidence. Summary statistics, warnings and errors are never dropped.
  2. If there is no successful tool result at all, do NOT call the LLM. Return a
     deterministic "could not establish an evidence-based answer" draft that repeats the
     tools' own error messages (so the LLM has no chance to fill the gap from memory).
  3. Otherwise ask the LLM (system = instructions only; user message = question + evidence)
     for a DraftAnswer as JSON. Validate it with Pydantic, including that every key_stat
     cites the call_id of a successful tool result. One repair turn on invalid output.
  4. If the LLM fails or never produces valid output, return a deterministic fallback that
     reports what was computed using only the tools' own interpretation text and warnings.
  5. Chart actions are validated against what the dashboard pages actually support;
     invalid ones are dropped and recorded, they never block the answer.

On a retry after the grounding guard (Stage 6) rejects a draft, the guard's feedback in
state["grounding_result"] is added to the user message.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm.base import LLMClient, LLMError, Message
from tools.common import LEAGUES

from .prompting import render
from .state import AgentError, AgentState, ChartAction, DraftAnswer, usage_record

NODE = "synthesiser"
MAX_REPAIRS = 1
MAX_EVIDENCE_CHARS = 16_000          # ~4k tokens; keeps the strong model's call within free-tier limits
ROW_LIMITS = (40, 20, 10, 5)         # tried in order when the evidence is over budget
SIG_FIGS = 6

# What the dashboard pages accept (index.html, q2-*.html, q3-*.html option values).
Q1_METRICS = ("Points", "GoalsFor", "GoalDiff", "Wins")
Q2_X_METRICS = ("GoalsFor", "AvgShotsOnTarget", "AvgShots")
Q3_X_METRICS = ("AvgFouls", "AvgYellowCards")
Q1_MAX_TEAMS = 4


# ---------------------------------------------------------------------------
# Evidence packaging
# ---------------------------------------------------------------------------
def _round(x: Any) -> Any:
    if isinstance(x, float) and math.isfinite(x):
        return float(f"{x:.{SIG_FIGS}g}")
    return x


def _compact(obj: Any, max_rows: int | None) -> Any:
    """Round floats; optionally trim long lists of row-dicts (lists of strings are never trimmed)."""
    if isinstance(obj, dict):
        return {k: _compact(v, max_rows) for k, v in obj.items()}
    if isinstance(obj, list):
        if max_rows is not None and len(obj) > max_rows and all(isinstance(i, dict) for i in obj):
            kept = [_compact(i, max_rows) for i in obj[:max_rows]]
            return kept + [{"_rows_omitted": len(obj) - max_rows,
                            "_note": "list trimmed to fit the prompt; summary fields above are complete"}]
        return [_compact(i, max_rows) for i in obj]
    return _round(obj)


def _analyst_issues(state: AgentState) -> list[dict]:
    """Analyst errors that matter to the reader. Tool errors are already in the results."""
    return [{"analyst": e["node"].split(":", 1)[-1], "kind": e["kind"], "message": e["message"]}
            for e in state.get("errors", [])
            if e["node"].startswith("analyst:") and e["kind"] != "tool_error"]


def package_evidence(state: AgentState) -> tuple[dict, bool]:
    """Return (evidence, trimmed). Successful results, failed results and analyst issues, nothing else."""
    results = []
    for r in state.get("tool_results", []):
        item = {"id": r["call_id"], "analyst": r["agent"], "tool": r["tool"], "arguments": r["args"], "ok": r["ok"]}
        if r["ok"]:
            item["result"] = r["result"]
        else:
            item["error"] = r["error"]
        results.append(item)
    base = {"intent": state.get("intent"), "tool_results": results, "analyst_issues": _analyst_issues(state)}

    evidence = _compact(base, None)
    if len(_dumps(evidence)) <= MAX_EVIDENCE_CHARS:
        return evidence, False
    for n in ROW_LIMITS:
        evidence = _compact(base, n)
        if len(_dumps(evidence)) <= MAX_EVIDENCE_CHARS:
            break
    return evidence, True


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def successful_ids(state: AgentState) -> set[str]:
    return {r["call_id"] for r in state.get("tool_results", []) if r["ok"]}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def build_messages(state: AgentState, evidence: dict, prompt_version: str | None = None) -> list[Message]:
    system = render("synthesiser", prompt_version,
                    q1_metrics=", ".join(Q1_METRICS), q2_x_metrics=", ".join(Q2_X_METRICS),
                    q3_x_metrics=", ".join(Q3_X_METRICS), leagues=", ".join(LEAGUES),
                    q1_max_teams=str(Q1_MAX_TEAMS))
    parts = [f"<question>\n{state['question']}\n</question>",
             f"<evidence>\n{_dumps(evidence)}\n</evidence>"]
    g = state.get("grounding_result")
    if g and not g.get("passed", True):
        previous = (state.get("draft_answer") or {}).get("answer", "")
        parts.append("GROUNDING FEEDBACK: your previous draft was rejected because some claims are not "
                     "supported by the evidence.\n"
                     f"Unsupported: {json.dumps(g.get('unsupported', []), ensure_ascii=False)}\n"
                     f"Details: {g.get('feedback', '')}\n"
                     f"Previous answer: {previous}\n"
                     "Rewrite the draft using only values present in the evidence.")
    return [Message("system", system), Message("user", "\n\n".join(parts))]


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------
class KeyStatOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("value", mode="before")
    @classmethod
    def _as_text(cls, v: Any) -> Any:
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class SynthesisOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    answer: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    key_stats: list[KeyStatOut] = Field(default_factory=list)
    explanation: str = ""
    chart_actions: list[dict[str, Any]] = Field(default_factory=list)


class ChartActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: Literal["q1", "q2", "q3"]
    league: str | None = None
    metric: str | None = None
    teams: list[str] | None = None
    x_metric: str | None = None
    season_start: str | None = None
    season_end: str | None = None
    highlight: list[str] | None = None

    def check(self) -> None:
        if self.league is not None and self.league not in LEAGUES and self.league != "all":
            raise ValueError(f"league must be one of {LEAGUES} or 'all'")
        if self.page == "q1":
            if self.league in (None, "all"):
                raise ValueError("q1 needs a single league")
            if self.metric is not None and self.metric not in Q1_METRICS:
                raise ValueError(f"q1 metric must be one of {Q1_METRICS}")
            if self.teams is not None and not 1 <= len(self.teams) <= Q1_MAX_TEAMS:
                raise ValueError(f"q1 shows 1 to {Q1_MAX_TEAMS} teams")
        if self.page == "q2" and self.x_metric is not None and self.x_metric not in Q2_X_METRICS:
            raise ValueError(f"q2 x_metric must be one of {Q2_X_METRICS}")
        if self.page == "q3" and self.x_metric is not None and self.x_metric not in Q3_X_METRICS:
            raise ValueError(f"q3 x_metric must be one of {Q3_X_METRICS}")


def validate_chart_actions(raw: list[dict]) -> tuple[list[ChartAction], list[str]]:
    ok, problems = [], []
    for i, a in enumerate(raw):
        try:
            m = ChartActionModel.model_validate(a)
            m.check()
        except (ValidationError, ValueError) as e:
            problems.append(f"chart_actions[{i}]: {str(e)[:200]}")
            continue
        ok.append(ChartAction(**m.model_dump(exclude_none=True)))
    return ok, problems


def parse_output(resp_text_json: Any, valid_sources: set[str]) -> SynthesisOut:
    if not isinstance(resp_text_json, dict):
        raise ValueError("expected a JSON object")
    out = SynthesisOut.model_validate(resp_text_json)
    bad = [k.source for k in out.key_stats if k.source not in valid_sources]
    if bad:
        raise ValueError(f"key_stats cite sources that are not successful tool results: {bad}. "
                         f"Valid sources: {sorted(valid_sources)}")
    return out


# ---------------------------------------------------------------------------
# Deterministic answers (no LLM, no numbers of our own)
# ---------------------------------------------------------------------------
def no_evidence_draft(state: AgentState) -> DraftAnswer:
    """No successful tool result: say so, and pass on the tools' own explanations."""
    lines = []
    for r in state.get("tool_results", []):
        e = r["error"] or {}
        msg = f"{r['tool']} failed: {e.get('message', 'unknown error')}"
        if e.get("suggestions"):
            msg += f" Suggestions: {', '.join(map(str, e['suggestions'][:5]))}."
        lines.append(msg)
    for issue in _analyst_issues(state):
        lines.append(f"{issue['analyst']} analyst: {issue['message']}")
    if not lines:
        lines.append("No analysis was run for this question.")
    return DraftAnswer(
        answer="I could not establish an evidence-based answer to this question from the available football data.",
        evidence=lines, key_stats=[],
        explanation="No statistic is reported because none of the analysis tools returned a usable result.")


def fallback_draft(state: AgentState, reason: str) -> DraftAnswer:
    """The LLM could not write the answer: report what the tools computed, in their own words."""
    lines, warnings = [], []
    for r in state.get("tool_results", []):
        if r["ok"]:
            res = r["result"] if isinstance(r["result"], dict) else {}
            text = res.get("interpretation")
            lines.append(f"{r['tool']} ({r['call_id']}): {text}" if text
                         else f"{r['tool']} ({r['call_id']}) returned a result; see the tool output for the figures.")
            warnings.extend(res.get("warnings") or [])
        else:
            lines.append(f"{r['tool']} failed: {(r['error'] or {}).get('message', 'unknown error')}")
    return DraftAnswer(
        answer="The analysis ran, but the written answer could not be generated, so only the tools' own "
               "summaries are shown below.",
        evidence=lines + list(dict.fromkeys(warnings)), key_stats=[],
        explanation=f"The language model could not produce a valid answer ({reason}).")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_synthesiser(llm: LLMClient, prompt_version: str | None = None) -> Callable[[AgentState], dict]:
    def synthesiser(state: AgentState) -> dict:
        update: dict = {"token_usage": [], "errors": [], "chart_actions": []}

        def err(kind: str, message: str) -> None:
            update["errors"].append(AgentError(node=NODE, kind=kind, message=message[:500]))

        sources = successful_ids(state)
        if not sources:
            err("no_evidence", "No successful tool results; returned a no-evidence answer without calling the LLM.")
            return {**update, "draft_answer": no_evidence_draft(state)}

        evidence, trimmed = package_evidence(state)
        if trimmed:
            err("evidence_trimmed", f"Evidence exceeded {MAX_EVIDENCE_CHARS} characters; long row lists were trimmed.")
        messages = build_messages(state, evidence, prompt_version)

        for attempt in range(MAX_REPAIRS + 1):
            try:
                resp = llm.complete(messages, purpose=NODE, json_mode=True, temperature=0.0)
            except LLMError as e:
                err("llm_error", f"{type(e).__name__}: {e}")
                return {**update, "draft_answer": fallback_draft(state, type(e).__name__)}
            update["token_usage"].append(usage_record(NODE, resp))
            try:
                out = parse_output(resp.json(), sources)
            except (LLMError, ValidationError, ValueError) as e:
                err("bad_output", f"{type(e).__name__}: {e}")
                if attempt < MAX_REPAIRS:
                    messages = messages + [
                        Message("assistant", resp.content),
                        Message("user", f"Your reply could not be used: {str(e)[:400]}\n"
                                        "Reply again with JSON only in the required shape.")]
                continue
            actions, problems = validate_chart_actions(out.chart_actions)
            for p in problems:
                err("bad_chart_action", p)
            draft = DraftAnswer(answer=out.answer.strip(), evidence=out.evidence,
                                key_stats=[k.model_dump() for k in out.key_stats],
                                explanation=out.explanation.strip())
            return {**update, "draft_answer": draft, "chart_actions": actions}

        return {**update, "draft_answer": fallback_draft(state, "invalid output after repair")}

    return synthesiser