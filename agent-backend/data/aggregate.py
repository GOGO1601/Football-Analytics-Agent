"""
Exact Python port of the dashboard's JavaScript aggregation and view logic.

Ported from (and validated against by tests/test_parity.py):
  frontend/src/data-loader.js  expandMatchesToTeamRows, aggregateToTeamSeasons      (Q2, Q3)
  frontend/src/main.js         transformToTeamMatchData, aggregateTeamSeason,
                               populateTeamMultiSelect defaults                       (Q1)
  frontend/src/q2-chart.js     getFilteredData, updateSummaryCards, linearRegression  (Q2)
  frontend/src/q3-chart.js     getFilteredData, updateSummaryCards, linearRegression  (Q3)

The two JS pipelines (data-loader.js and main.js) are line-for-line identical in logic,
so one Python implementation serves both; the parity test checks it against each.

Deliberately reproduced JS behaviour (do not "fix" here):
  * Result: home team Win iff FTR == "H", Draw iff "D", otherwise Loss
            away team Win iff FTR == "A", Draw iff "D", otherwise Loss
            (an unexpected FTR value therefore counts as a Loss for BOTH teams)
  * Points  = Wins * 3 + Draws (not a sum of per-match points)
  * GoalsFor / GoalsAgainst = d3.sum, which skips NaN/null (treated as 0)
  * Avg* = nullMean: mean over non-null values only; null if none
  * floating point sums are plain left-to-right additions, like d3.sum / d3.mean.
    Python's built-in sum() (3.12+) and pandas group sums use compensated summation
    and can differ in the last bits, so they are NOT used for parity values.
  * seasons are ordered by parseInt(season.split("-")[0])
  * no rounding is applied to stored values; rounding only happens at display time
    via Number.prototype.toFixed, reproduced by js_to_fixed().

Additions that do NOT exist in the JS are kept separate at the bottom of the file
(home/away splits) and are not part of the parity contract.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .loader import load_matches, season_start_year, valid_matches

TEAM_SEASON_COLUMNS = [
    "League", "Season", "Team", "Played", "Wins", "Draws", "Losses", "Points",
    "GoalsFor", "GoalsAgainst", "GoalDiff",
    "AvgShots", "AvgShotsOnTarget", "AvgFouls", "AvgYellowCards", "AvgRedCards",
]
INT_COLUMNS = ["Played", "Wins", "Draws", "Losses", "Points"]
AVG_COLUMNS = ["AvgShots", "AvgShotsOnTarget", "AvgFouls", "AvgYellowCards", "AvgRedCards"]
# team-row field -> aggregate column
_AVG_SOURCE = {"AvgShots": "Shots", "AvgShotsOnTarget": "ShotsOnTarget", "AvgFouls": "Fouls",
               "AvgYellowCards": "YellowCards", "AvgRedCards": "RedCards"}


# ---------------------------------------------------------------------------
# d3 numeric helpers (sequential, NaN/null-skipping)
# ---------------------------------------------------------------------------
def _is_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def js_sum(values: Iterable) -> float:
    """d3.sum: left-to-right addition of numeric values, skipping null/NaN; 0 if none."""
    total = 0.0
    for v in values:
        if _is_num(v):
            total = total + v
    return total


def js_mean(values: Iterable) -> float | None:
    """d3.mean: left-to-right sum / count over numeric values; None (JS undefined) if none."""
    total, count = 0.0, 0
    for v in values:
        if _is_num(v):
            total = total + v
            count += 1
    return total / count if count else None


def js_to_fixed(x: float, digits: int) -> str:
    """Number.prototype.toFixed: round the exact binary value half-up (Python's format() rounds half-even)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(x).quantize(q, rounding=ROUND_HALF_UP))


def js_locale_int(n: int) -> str:
    """(n).toLocaleString() for en-GB/en-US integers: thousands separated by commas."""
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Step 2 — match rows -> per-team rows
# ---------------------------------------------------------------------------
def expand_to_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """expandMatchesToTeamRows / transformToTeamMatchData: one home row then one away row per match."""
    m = valid_matches(matches)
    cols = ["League", "Season", "Team", "GoalsFor", "GoalsAgainst", "Result", "Points",
            "Shots", "ShotsOnTarget", "Fouls", "YellowCards", "RedCards", "Venue"]
    if m.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    ftr = m["FTR"].astype(str)
    home = pd.DataFrame({
        "League": m["League"].values, "Season": m["Season"].values, "Team": m["HomeTeam"].values,
        "GoalsFor": m["FTHG"].values, "GoalsAgainst": m["FTAG"].values,
        "Result": np.where(ftr == "H", "Win", np.where(ftr == "D", "Draw", "Loss")),
        "Points": np.where(ftr == "H", 3, np.where(ftr == "D", 1, 0)),
        "Shots": m["HS"].values, "ShotsOnTarget": m["HST"].values, "Fouls": m["HF"].values,
        "YellowCards": m["HY"].values, "RedCards": m["HR"].values, "Venue": "Home",
        "_order": np.arange(len(m)) * 2,
    })
    away = pd.DataFrame({
        "League": m["League"].values, "Season": m["Season"].values, "Team": m["AwayTeam"].values,
        "GoalsFor": m["FTAG"].values, "GoalsAgainst": m["FTHG"].values,
        "Result": np.where(ftr == "A", "Win", np.where(ftr == "D", "Draw", "Loss")),
        "Points": np.where(ftr == "A", 3, np.where(ftr == "D", 1, 0)),
        "Shots": m["AS"].values, "ShotsOnTarget": m["AST"].values, "Fouls": m["AF"].values,
        "YellowCards": m["AY"].values, "RedCards": m["AR"].values, "Venue": "Away",
        "_order": np.arange(len(m)) * 2 + 1,
    })
    rows = pd.concat([home, away], ignore_index=True).sort_values("_order", kind="stable")
    return rows.drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — per-team rows -> team-season summary
# ---------------------------------------------------------------------------
def aggregate_team_seasons(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    aggregateToTeamSeasons / aggregateTeamSeason.
    Groups League -> Team -> Season in first-appearance order (d3.group semantics), so the
    output row order is the same as the JS array order.
    """
    if team_rows.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in TEAM_SEASON_COLUMNS + ["SeasonStart"]})

    # nested insertion-ordered grouping, exactly like d3.group(rows, League, Team, Season)
    groups: dict = {}
    recs = team_rows[["League", "Team", "Season", "Result", "GoalsFor", "GoalsAgainst",
                      "Shots", "ShotsOnTarget", "Fouls", "YellowCards", "RedCards"]].itertuples(index=False)
    for r in recs:
        groups.setdefault(r.League, {}).setdefault(r.Team, {}).setdefault(r.Season, []).append(r)

    out = []
    for league, teams in groups.items():
        for team, seasons in teams.items():
            for season, ms in seasons.items():
                wins = sum(1 for x in ms if x.Result == "Win")
                draws = sum(1 for x in ms if x.Result == "Draw")
                losses = sum(1 for x in ms if x.Result == "Loss")
                gf = js_sum(x.GoalsFor for x in ms)
                ga = js_sum(x.GoalsAgainst for x in ms)
                row = {
                    "League": league, "Season": season, "Team": team,
                    "Played": len(ms), "Wins": wins, "Draws": draws, "Losses": losses,
                    "Points": wins * 3 + draws,
                    "GoalsFor": gf, "GoalsAgainst": ga, "GoalDiff": gf - ga,
                }
                for col, src in _AVG_SOURCE.items():
                    # nullMean: filter != null, then d3.mean (which also skips NaN)
                    row[col] = js_mean(getattr(x, src) for x in ms)
                out.append(row)

    df = pd.DataFrame(out, columns=TEAM_SEASON_COLUMNS)
    for c in INT_COLUMNS:
        df[c] = df[c].astype(int)
    for c in ("GoalsFor", "GoalsAgainst", "GoalDiff"):
        df[c] = df[c].astype(float)
    for c in AVG_COLUMNS:
        df[c] = df[c].astype(float)  # None -> NaN (missing)
    df["SeasonStart"] = df["Season"].map(season_start_year).astype(int)
    return df


def team_seasons(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full pipeline: loadAllLeagues() equivalent. Loads from /datasets if matches not given."""
    if matches is None:
        matches = load_matches()
    return aggregate_team_seasons(expand_to_team_rows(matches))


# ---------------------------------------------------------------------------
# Dashboard view logic
# ---------------------------------------------------------------------------
Q1_DEFAULT_TEAMS = {
    "Premier League": ["Manchester United", "Arsenal", "Liverpool"],
    "La Liga": ["Barcelona", "Real Madrid", "Atletico Madrid"],
    "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "Werder Bremen"],
    "Serie A": ["Juventus", "AC Milan", "Inter"],
    "Ligue 1": ["Paris SG", "Lyon", "Marseille"],
}


def _js_string_sort(values: Iterable[str]) -> list[str]:
    """Array.prototype.sort() default: compare UTF-16 code units."""
    return sorted(values, key=lambda s: s.encode("utf-16-be"))


def q1_teams(ts: pd.DataFrame, league: str) -> list[str]:
    """populateTeamMultiSelect: [...new Set(teams)].sort()."""
    return _js_string_sort(set(ts.loc[ts["League"] == league, "Team"]))


def q1_default_selection(ts: pd.DataFrame, league: str) -> list[str]:
    """Default selection: preferred names that exist, else the first 3 teams. In option (sorted) order."""
    teams = q1_teams(ts, league)
    preferred = [t for t in Q1_DEFAULT_TEAMS.get(league, []) if t in teams]
    chosen = preferred if preferred else teams[:3]
    return [t for t in teams if t in chosen]


def q1_series(ts: pd.DataFrame, league: str, teams: Sequence[str], metric: str) -> dict[str, list[dict]]:
    """Line-chart data: per team, seasons in chronological order."""
    d = ts[(ts["League"] == league) & (ts["Team"].isin(teams))]
    out = {}
    for t in teams:
        g = d[d["Team"] == t].sort_values("SeasonStart", kind="stable")
        out[t] = [{"Season": s, metric: v} for s, v in zip(g["Season"], g[metric])]
    return out


def q2_filter(ts: pd.DataFrame, x_metric: str, league: str = "all",
              season_start: str | None = None, season_end: str | None = None) -> pd.DataFrame:
    """q2-chart.js getFilteredData. Season bounds default to the full range (dropdown defaults)."""
    lo = season_start_year(season_start) if season_start else ts["SeasonStart"].min()
    hi = season_start_year(season_end) if season_end else ts["SeasonStart"].max()
    keep = ts[x_metric].notna() & ts["Points"].notna()
    if league != "all":
        keep &= ts["League"] == league
    keep &= (ts["SeasonStart"] >= lo) & (ts["SeasonStart"] <= hi)
    return ts[keep]


def q2_missing_count(ts: pd.DataFrame, x_metric: str) -> int:
    """Q2 'missing' card: rows with a null x metric across ALL data (ignores the league/season filters)."""
    return int(ts[x_metric].isna().sum())


def q3_filter(ts: pd.DataFrame, x_metric: str, league: str = "all") -> pd.DataFrame:
    """q3-chart.js getFilteredData: fouls, yellows AND reds must all be present. No season filter."""
    keep = ts["AvgFouls"].notna() & ts["AvgYellowCards"].notna() & ts["AvgRedCards"].notna()
    keep &= ts["Points"].notna() & ts[x_metric].notna()
    if league != "all":
        keep &= ts["League"] == league
    return ts[keep]


def linear_regression(rows: pd.DataFrame, x_key: str, y_key: str) -> dict | None:
    """linearRegression() from q2-chart.js / q3-chart.js (identical in both)."""
    n = len(rows)
    if n < 2:
        return None
    xs = rows[x_key].tolist()
    ys = rows[y_key].tolist()
    mx, my = js_mean(xs), js_mean(ys)
    ssxx = js_sum((x - mx) ** 2 for x in xs)
    ssxy = js_sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ssxx == 0:
        return None
    slope = ssxy / ssxx
    intercept = my - slope * mx
    ss_tot = js_sum((y - my) ** 2 for y in ys)
    ss_res = js_sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"slope": slope, "intercept": intercept, "r2": r2}


def q2_summary(ts: pd.DataFrame, x_metric: str, league: str = "all",
               season_start: str | None = None, season_end: str | None = None) -> dict:
    """Everything the Q2 page displays for a filter state (cards + regression label)."""
    f = q2_filter(ts, x_metric, league, season_start, season_end)
    reg = linear_regression(f, x_metric, "Points")
    return {
        "n": len(f), "card_count": js_locale_int(len(f)),
        "card_seasons": str(f["Season"].nunique()),
        "card_missing": js_locale_int(q2_missing_count(ts, x_metric)),
        "regression": reg, "r2_label": js_to_fixed(reg["r2"], 3) if reg else None,
    }


def q3_summary(ts: pd.DataFrame, x_metric: str, league: str = "all") -> dict:
    """Everything the Q3 page displays for a filter state."""
    f = q3_filter(ts, x_metric, league)
    reg = linear_regression(f, x_metric, "Points")
    mean_fouls = js_mean(f["AvgFouls"].tolist()) if len(f) else None
    mean_yellow = js_mean(f["AvgYellowCards"].tolist()) if len(f) else None
    return {
        "n": len(f), "card_count": js_locale_int(len(f)),
        "card_seasons": str(f["Season"].nunique()),
        "mean_fouls": mean_fouls, "mean_yellow": mean_yellow,
        "card_fouls": js_to_fixed(mean_fouls, 1) if mean_fouls is not None else "—",
        "card_yellow": js_to_fixed(mean_yellow, 2) if mean_yellow is not None else "—",
        "regression": reg, "r2_label": js_to_fixed(reg["r2"], 3) if reg else None,
    }


def row_keys(df: pd.DataFrame) -> list[str]:
    """Sorted 'League|Team|Season' keys (same format the fixture generator hashes)."""
    return _js_string_sort(df["League"] + "|" + df["Team"] + "|" + df["Season"])


# ---------------------------------------------------------------------------
# Extensions (NOT in the JS dashboard; not covered by parity)
# ---------------------------------------------------------------------------
def team_season_home_away(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Team-season stats split by venue (Home/Away), same definitions as the main aggregate."""
    if matches is None:
        matches = load_matches()
    rows = expand_to_team_rows(matches)
    if rows.empty:
        return pd.DataFrame()
    parts = []
    for venue in ("Home", "Away"):
        agg = aggregate_team_seasons(rows[rows["Venue"] == venue])
        agg.insert(3, "Venue", venue)
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)
