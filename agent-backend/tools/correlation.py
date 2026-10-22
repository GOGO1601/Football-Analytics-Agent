"""
Relationships between team-season metrics (the questions behind Q2 and Q3).

Selection rules match the dashboard by default:
  * Q2 style (attack vs points): rows where x and y are present, optional league and season range
  * Q3 style (discipline vs points): additionally require AvgFouls, AvgYellowCards and AvgRedCards
    (`require_discipline_fields=True`), exactly like q3-chart.js
Every result includes `dashboard_regression`, computed with the ported linearRegression(),
so an agent can quote the same R² the chart shows.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from data.aggregate import js_to_fixed, linear_regression

from .common import (Dataset, ToolError, _ds, check_metric, effect_label, jsonable, p_text,
                     quality_warnings, resolve_league, season_bounds)

MIN_N = 3
DISCIPLINE = ["AvgFouls", "AvgYellowCards", "AvgRedCards"]


def _select(d: Dataset, x: str, y: str, league, season_start, season_end,
            require_discipline_fields: bool, complete_seasons_only: bool) -> tuple[pd.DataFrame, str | None]:
    check_metric(x)
    check_metric(y)
    if x == y:
        raise ToolError("invalid_argument", "x_metric and y_metric must differ.")
    lg = resolve_league(league, d)
    lo, hi = season_bounds(season_start, season_end, d, lg)
    rows = d.ts[d.ts["SeasonStart"].between(lo, hi)]
    if lg:
        rows = rows[rows["League"] == lg]
    keep = rows[x].notna() & rows[y].notna()
    if require_discipline_fields:
        for c in DISCIPLINE:
            keep &= rows[c].notna()
    rows = rows[keep]
    if complete_seasons_only and len(rows):
        ok = d.coverage[d.coverage["complete"]][["League", "Season"]]
        rows = rows.merge(ok, on=["League", "Season"])
    if len(rows) < MIN_N:
        raise ToolError("insufficient_data", f"Only {len(rows)} team-seasons have both {x} and {y} "
                        f"for this selection; at least {MIN_N} are needed.", n=len(rows))
    return rows, lg


def _dashboard(rows: pd.DataFrame, x: str, y: str) -> dict | None:
    reg = linear_regression(rows, x, y)
    if reg is None:
        return None
    return {**reg, "r2_label": js_to_fixed(reg["r2"], 3)}


def correlation_analysis(x_metric: str, y_metric: str = "Points", league: str | None = None,
                         season_start: str | None = None, season_end: str | None = None,
                         method: str = "pearson", require_discipline_fields: bool = False,
                         complete_seasons_only: bool = False, data: Dataset | None = None) -> dict:
    """Pearson or Spearman correlation with p-value, 95% CI (Pearson, Fisher z) and strength label."""
    d = _ds(data)
    if method not in ("pearson", "spearman"):
        raise ToolError("invalid_argument", "method must be 'pearson' or 'spearman'.")
    rows, lg = _select(d, x_metric, y_metric, league, season_start, season_end,
                       require_discipline_fields, complete_seasons_only)
    x, y = rows[x_metric].to_numpy(float), rows[y_metric].to_numpy(float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ToolError("insufficient_data", "One of the metrics is constant in this selection.")
    n = len(rows)
    if method == "pearson":
        res = sps.pearsonr(x, y)
        r, p = float(res.statistic), float(res.pvalue)
        ci = None
        if n > 3:
            lo, hi = res.confidence_interval(0.95)
            ci = [float(lo), float(hi)]
    else:
        res = sps.spearmanr(x, y)
        r, p = float(res.statistic), float(res.pvalue)
        ci = None
        if n > 3:  # Fisher z with Bonett-Wright standard error for Spearman
            z, se = math.atanh(max(min(r, 0.999999), -0.999999)), math.sqrt((1 + r * r / 2) / (n - 3))
            ci = [math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se)]
    strength = effect_label(r, "r")
    direction = "positive" if r > 0 else "negative"
    sig = p < 0.05
    interp = (f"{strength.capitalize()} {direction} {method} correlation between {x_metric} and {y_metric} "
              f"(r = {r:.3f}, n = {n}, {p_text(p)}); "
              + ("statistically significant at the 5% level." if sig else "not statistically significant at the 5% level.")
              + " Correlation does not establish causation.")
    return jsonable({
        "x_metric": x_metric, "y_metric": y_metric, "league": lg, "method": method, "n": n,
        "r": r, "r_squared": r * r, "p_value": p, "ci95": ci, "strength": strength,
        "direction": direction, "significant_at_0_05": sig,
        "dashboard_regression": _dashboard(rows, x_metric, y_metric),
        "interpretation": interp,
        "warnings": quality_warnings(d, rows),
    })


def regression_analysis(x_metric: str, y_metric: str = "Points", league: str | None = None,
                        season_start: str | None = None, season_end: str | None = None,
                        require_discipline_fields: bool = False, complete_seasons_only: bool = False,
                        data: Dataset | None = None) -> dict:
    """Simple OLS y ~ x: slope with standard error, t-test, 95% CI, intercept and R²."""
    d = _ds(data)
    rows, lg = _select(d, x_metric, y_metric, league, season_start, season_end,
                       require_discipline_fields, complete_seasons_only)
    x, y = rows[x_metric].to_numpy(float), rows[y_metric].to_numpy(float)
    if np.ptp(x) == 0:
        raise ToolError("insufficient_data", f"{x_metric} is constant in this selection.")
    n = len(rows)
    lr = sps.linregress(x, y)
    df = n - 2
    tcrit = float(sps.t.ppf(0.975, df))
    resid = y - (lr.intercept + lr.slope * x)
    interp = (f"Each extra unit of {x_metric} is associated with {lr.slope:+.3f} {y_metric} "
              f"(95% CI {lr.slope - tcrit * lr.stderr:+.3f} to {lr.slope + tcrit * lr.stderr:+.3f}); "
              f"R² = {lr.rvalue ** 2:.3f}, n = {n}, {p_text(lr.pvalue)}.")
    return jsonable({
        "x_metric": x_metric, "y_metric": y_metric, "league": lg, "n": n,
        "slope": lr.slope, "slope_stderr": lr.stderr,
        "slope_ci95": [lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr],
        "intercept": lr.intercept, "intercept_stderr": lr.intercept_stderr,
        "t_statistic": lr.slope / lr.stderr if lr.stderr else None, "df": df,
        "p_value": lr.pvalue, "r_squared": lr.rvalue ** 2,
        "residual_std": float(np.sqrt(np.sum(resid ** 2) / df)) if df > 0 else None,
        "dashboard_regression": _dashboard(rows, x_metric, y_metric),
        "interpretation": interp,
        "warnings": quality_warnings(d, rows),
    })
