"""
Tool registry: the only surface the agents will see.

Each tool gets a JSON schema generated from its Python signature (so schemas cannot drift
from the code), arguments are validated with pydantic before the call, and every call
returns an envelope instead of raising:

    {"ok": True,  "tool": name, "result": {...}}
    {"ok": False, "tool": name, "error": {"code": ..., "message": ..., "suggestions": [...]}}
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ConfigDict, ValidationError, create_model

from . import correlation, hypothesis, stats
from .common import ToolError, jsonable


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., dict]
    description: str
    args_model: type

    def schema(self) -> dict:
        s = self.args_model.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": s}


def _args_model(fn: Callable) -> type:
    fields: dict[str, Any] = {}
    for name, p in inspect.signature(fn).parameters.items():
        if name == "data":  # injected dataset, never exposed to the LLM
            continue
        ann = fn.__annotations__.get(name, Any)
        if isinstance(ann, str):
            ann = eval(ann, fn.__globals__)  # resolve `from __future__ import annotations`
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (ann, default)
    return create_model(f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields)


def _tool(fn: Callable, description: str) -> Tool:
    return Tool(fn.__name__, fn, description, _args_model(fn))


TOOLS: dict[str, Tool] = {t.name: t for t in [
    _tool(stats.team_season_stats, "Season-by-season stats for one team (or one season), with totals and per-game averages."),
    _tool(stats.league_table, "Final league table for one league and season, computed from match results."),
    _tool(stats.top_n, "Top (or bottom) team-seasons by a metric, optionally filtered by league and season."),
    _tool(stats.season_trend, "How a team's metric changed over seasons: series, change, best/worst, linear trend."),
    _tool(stats.team_comparison, "Compare 2-6 teams on a metric across seasons, including head-to-head records."),
    _tool(stats.head_to_head, "Head-to-head record (wins, draws, goals) between two teams."),
    _tool(stats.most_improved, "Teams whose metric improved or declined most between two seasons in a league."),
    _tool(stats.home_advantage, "Home vs away win/draw/loss rates, goals and points per game; league-wide or for one team."),
    _tool(stats.data_coverage, "Which seasons exist per league, which are incomplete, and when shots/fouls/cards start."),
    _tool(correlation.correlation_analysis, "Correlation (Pearson/Spearman) between two team-season metrics, e.g. shots vs points."),
    _tool(correlation.regression_analysis, "Linear regression of one team-season metric on another with slope CI and p-value."),
    _tool(hypothesis.home_advantage_test, "Test whether home advantage changed between two periods (default: pre-COVID vs 2020-21)."),
    _tool(hypothesis.hypothesis_test, "Test whether a team-season metric differs between two groups (leagues, eras or teams)."),
]}


def list_tools() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def call_tool(name: str, args: dict | None = None, *, data=None) -> dict:
    """Validate arguments and run a tool. Never raises for bad input; returns an error envelope."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "tool": name, "error": {"code": "unknown_tool", "message": f"No tool named '{name}'.",
                                                     "suggestions": list(TOOLS)}}
    try:
        parsed = tool.args_model(**(args or {}))
    except ValidationError as e:
        return {"ok": False, "tool": name, "error": {"code": "invalid_arguments", "message": str(e),
                                                     "suggestions": []}}
    kwargs = {k: getattr(parsed, k) for k in type(parsed).model_fields}
    if data is not None:
        kwargs["data"] = data
    try:
        return {"ok": True, "tool": name, "result": jsonable(tool.fn(**kwargs))}
    except ToolError as e:
        return {"ok": False, "tool": name, "error": e.to_dict()}
