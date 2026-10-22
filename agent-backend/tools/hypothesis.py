"""
Hypothesis tests with full reporting: test statistic, p-value, confidence interval,
effect size, sample sizes and a templated plain-English interpretation built only from
the computed numbers.

home_advantage_test   match-level: did home advantage change between two periods?
                      Default is the COVID natural experiment: 2010-11..2018-19 (crowds)
                      vs 2020-21 (played almost entirely behind closed doors). 2019-20 is
                      excluded by default because it mixes both conditions.
hypothesis_test       team-season level: is a metric different between two groups
                      (leagues, eras or sets of teams)?
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as sps

from .common import (Dataset, ToolError, _ds, check_metric, effect_label, jsonable, p_text,
                     quality_warnings, resolve_league, resolve_team, season_bounds)

Z975 = float(sps.norm.ppf(0.975))


def _period_matches(d: Dataset, league: str | None, start: str, end: str) -> pd.DataFrame:
    lo, hi = season_bounds(start, end, d, league)
    m = d.matches[d.matches["SeasonStart"].between(lo, hi)]
    if league:
        m = m[m["League"] == league]
    return m


def _block(m: pd.DataFrame, start: str, end: str) -> dict:
    n = len(m)
    hw, dr, aw = (int((m["FTR"] == r).sum()) for r in ("H", "D", "A"))
    gd = (m["FTHG"] - m["FTAG"]).dropna()
    return {"seasons": f"{start} to {end}", "matches": n, "home_wins": hw, "draws": dr, "away_wins": aw,
            "home_win_rate": hw / n if n else None, "draw_rate": dr / n if n else None,
            "away_win_rate": aw / n if n else None,
            "mean_home_goal_margin": float(gd.mean()) if len(gd) else None,
            "home_points_per_game": (3 * hw + dr) / n if n else None}


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t-test of mean(b) - mean(a), with Welch-Satterthwaite CI and Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    res = sps.ttest_ind(b, a, equal_var=False)
    dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se else float("nan")
    tcrit = float(sps.t.ppf(0.975, dfw)) if se else float("nan")
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if na + nb > 2 else 0.0
    d = (mb - ma) / sp if sp else 0.0
    j = 1 - 3 / (4 * (na + nb) - 9) if na + nb > 2 else 1.0  # Hedges small-sample correction
    return {"mean_a": ma, "mean_b": mb, "difference": mb - ma, "t_statistic": float(res.statistic),
            "df": dfw, "p_value": float(res.pvalue),
            "ci95": [mb - ma - tcrit * se, mb - ma + tcrit * se],
            "cohens_d": d, "hedges_g": d * j, "effect_size_label": effect_label(d, "d")}


def home_advantage_test(league: str | None = None,
                        period_a_start: str = "2010-11", period_a_end: str = "2018-19",
                        period_b_start: str = "2020-21", period_b_end: str = "2020-21",
                        data: Dataset | None = None) -> dict:
    """Compare home advantage between two non-overlapping season ranges (B minus A)."""
    d = _ds(data)
    lg = resolve_league(league, d)
    a_lo, a_hi = season_bounds(period_a_start, period_a_end, d, lg)
    b_lo, b_hi = season_bounds(period_b_start, period_b_end, d, lg)
    if not (a_hi < b_lo or b_hi < a_lo):
        raise ToolError("invalid_argument", "The two periods overlap; they must be disjoint.")
    ma = _period_matches(d, lg, period_a_start, period_a_end)
    mb = _period_matches(d, lg, period_b_start, period_b_end)
    if len(ma) < 30 or len(mb) < 30:
        raise ToolError("insufficient_data", f"Need at least 30 matches per period "
                        f"(got {len(ma)} and {len(mb)}).", n_a=len(ma), n_b=len(mb))
    A, B = _block(ma, period_a_start, period_a_end), _block(mb, period_b_start, period_b_end)

    # 1. two-proportion z-test on the home win rate
    x1, n1, x2, n2 = A["home_wins"], A["matches"], B["home_wins"], B["matches"]
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 else 0.0
    p_z = float(2 * sps.norm.sf(abs(z)))
    se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))
    z_test = {"statistic": z, "p_value": p_z, "difference": p2 - p1,
              "ci95": [p2 - p1 - Z975 * se1, p2 - p1 + Z975 * se1],
              "cohens_h": h, "effect_size_label": effect_label(h, "h")}

    # 2. Fisher's exact test = exact permutation test of the same 2x2 table
    fisher = sps.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
    # 3. chi-square on the full H/D/A distribution
    table = np.array([[A["home_wins"], A["draws"], A["away_wins"]],
                      [B["home_wins"], B["draws"], B["away_wins"]]])
    chi2, p_chi, dof, _ = sps.chi2_contingency(table)
    cramers_v = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    # 4. Welch t-test on home goal margin per match
    margin = _welch((ma["FTHG"] - ma["FTAG"]).dropna().to_numpy(float),
                    (mb["FTHG"] - mb["FTAG"]).dropna().to_numpy(float))

    direction = "fell" if p2 < p1 else "rose"
    interp = (f"Home win rate {direction} from {100 * p1:.1f}% ({period_a_start} to {period_a_end}, n = {n1}) "
              f"to {100 * p2:.1f}% ({period_b_start} to {period_b_end}, n = {n2}), a change of "
              f"{100 * (p2 - p1):+.1f} percentage points (95% CI {100 * z_test['ci95'][0]:+.1f} to "
              f"{100 * z_test['ci95'][1]:+.1f}). Two-proportion z = {z:.2f}, {p_text(p_z)}; "
              f"Cohen's h = {h:.3f} ({z_test['effect_size_label']}). "
              + ("The change is statistically significant at the 5% level." if p_z < 0.05
                 else "The change is not statistically significant at the 5% level.")
              + f" Mean home goal margin changed by {margin['difference']:+.3f} goals per match ({p_text(margin['p_value'])}).")
    return jsonable({
        "league": lg, "comparison": "period_b minus period_a", "period_a": A, "period_b": B,
        "tests": {"home_win_rate_two_proportion_z": z_test,
                  "home_win_rate_fisher_exact": {"odds_ratio": float(fisher.statistic), "p_value": float(fisher.pvalue)},
                  "result_distribution_chi_square": {"statistic": float(chi2), "df": int(dof),
                                                     "p_value": float(p_chi), "cramers_v": cramers_v},
                  "home_goal_margin_welch_t": margin},
        "assumptions": ["Matches are treated as independent observations.",
                        "Season ranges are inclusive and based on the season start year."],
        "interpretation": interp,
        "warnings": quality_warnings(d, pd.concat([ma, mb])[["League", "Season"]]),
    })


class GroupFilter(BaseModel):
    """Selects a group of team-seasons. Unknown keys are rejected, so a typo cannot silently widen a group."""
    model_config = ConfigDict(extra="forbid")
    league: str | None = None
    season_start: str | None = None
    season_end: str | None = None
    teams: list[str] | None = Field(default=None, description="Restrict to these teams")


def _group_rows(d: Dataset, g: GroupFilter, metric: str) -> tuple[pd.DataFrame, list[str]]:
    lg = resolve_league(g.league, d)
    lo, hi = season_bounds(g.season_start, g.season_end, d, lg)
    rows = d.ts[d.ts["SeasonStart"].between(lo, hi)]
    if lg:
        rows = rows[rows["League"] == lg]
    teams = []
    if g.teams:
        teams = [resolve_team(t, d, lg) for t in g.teams]
        rows = rows[rows["Team"].isin(teams)]
    return rows[rows[metric].notna()], teams


def hypothesis_test(metric: str, group_a: GroupFilter, group_b: GroupFilter,
                    data: Dataset | None = None) -> dict:
    """Two-sample comparison of a team-season metric (B minus A): Welch t-test + Mann-Whitney U."""
    d = _ds(data)
    check_metric(metric)
    ga = group_a if isinstance(group_a, GroupFilter) else GroupFilter(**group_a)
    gb = group_b if isinstance(group_b, GroupFilter) else GroupFilter(**group_b)
    ra, ta = _group_rows(d, ga, metric)
    rb, tb = _group_rows(d, gb, metric)
    if len(ra) < 2 or len(rb) < 2:
        raise ToolError("insufficient_data", f"Each group needs at least 2 team-seasons with {metric} "
                        f"(got {len(ra)} and {len(rb)}).", n_a=len(ra), n_b=len(rb))
    key = ["League", "Team", "Season"]
    overlap = ra[key].merge(rb[key])
    if len(overlap):
        raise ToolError("invalid_argument", f"The groups share {len(overlap)} team-seasons; they must be disjoint.")
    a, b = ra[metric].to_numpy(float), rb[metric].to_numpy(float)
    if np.ptp(np.concatenate([a, b])) == 0:
        raise ToolError("insufficient_data", f"{metric} is identical for every team-season selected.")
    w = _welch(a, b)
    mw = sps.mannwhitneyu(b, a, alternative="two-sided")
    rank_biserial = 2 * float(mw.statistic) / (len(a) * len(b)) - 1
    interp = (f"Mean {metric} is {w['mean_b']:.3f} in group B (n = {len(b)}) vs {w['mean_a']:.3f} in group A "
              f"(n = {len(a)}), difference {w['difference']:+.3f} (95% CI {w['ci95'][0]:+.3f} to {w['ci95'][1]:+.3f}). "
              f"Welch t = {w['t_statistic']:.2f}, {p_text(w['p_value'])}; Cohen's d = {w['cohens_d']:.3f} "
              f"({w['effect_size_label']}). Mann-Whitney U {p_text(float(mw.pvalue))}. "
              + ("Significant at the 5% level." if w["p_value"] < 0.05 else "Not significant at the 5% level."))
    return jsonable({
        "metric": metric, "comparison": "group_b minus group_a",
        "group_a": {**ga.model_dump(), "teams": ta or None, "n": len(a), "mean": float(a.mean()),
                    "std": float(a.std(ddof=1)), "median": float(np.median(a))},
        "group_b": {**gb.model_dump(), "teams": tb or None, "n": len(b), "mean": float(b.mean()),
                    "std": float(b.std(ddof=1)), "median": float(np.median(b))},
        "tests": {"welch_t": w, "mann_whitney_u": {"statistic": float(mw.statistic), "p_value": float(mw.pvalue),
                                                   "rank_biserial": rank_biserial}},
        "assumptions": ["Team-seasons are treated as independent observations (the same club across "
                        "seasons is not modelled as repeated measures)."],
        "interpretation": interp,
        "warnings": quality_warnings(d, pd.concat([ra, rb])),
    })
