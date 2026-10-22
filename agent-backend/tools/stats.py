"""
Descriptive football statistics. Every number comes from the dashboard-identical
team-season aggregate (data.aggregate) or directly from match rows.

All functions take an optional `data: Dataset` (tests inject small synthetic datasets;
in production it defaults to the cached /datasets data) and return JSON-safe dicts.
"""
from __future__ import annotations

import itertools

import pandas as pd
from scipy import stats as sps

from .common import (Dataset, ToolError, _ds, check_metric, jsonable, quality_warnings, records,
                     normalise_season, resolve_league, resolve_season, resolve_team, season_bounds)

ROW_COLS = ["League", "Season", "Team", "Played", "Wins", "Draws", "Losses", "Points",
            "GoalsFor", "GoalsAgainst", "GoalDiff", "PointsPerGame", "GoalsForPerGame",
            "AvgShots", "AvgShotsOnTarget", "AvgFouls", "AvgYellowCards", "AvgRedCards"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["SeasonStart", "League", "Team"], kind="stable")


# ---------------------------------------------------------------------------
def team_season_stats(team: str, season: str | None = None, league: str | None = None,
                      data: Dataset | None = None) -> dict:
    """Season-by-season stats for one team (one season if given), plus career totals/averages."""
    d = _ds(data)
    lg = resolve_league(league, d)
    t = resolve_team(team, d, lg)
    rows = d.ts[d.ts["Team"] == t]
    if lg:
        rows = rows[rows["League"] == lg]
    if season is not None:
        s = resolve_season(season, d, lg)
        rows = rows[rows["Season"] == s]
        if rows.empty:
            raise ToolError("no_data", f"{t} has no matches in {s}.",
                            suggestions=list(_sorted(d.ts[d.ts["Team"] == t])["Season"])[-5:])
    rows = _sorted(rows)
    played = int(rows["Played"].sum())
    summary = {
        "seasons": int(len(rows)), "played": played,
        "wins": int(rows["Wins"].sum()), "draws": int(rows["Draws"].sum()), "losses": int(rows["Losses"].sum()),
        "points": int(rows["Points"].sum()),
        "goals_for": int(rows["GoalsFor"].sum()), "goals_against": int(rows["GoalsAgainst"].sum()),
        "points_per_game": rows["Points"].sum() / played if played else None,
        "goals_for_per_game": rows["GoalsFor"].sum() / played if played else None,
        "goals_against_per_game": rows["GoalsAgainst"].sum() / played if played else None,
        "goals_for_per_season": rows["GoalsFor"].mean() if len(rows) else None,
        "points_per_season": rows["Points"].mean() if len(rows) else None,
    }
    return jsonable({"team": t, "league": lg, "season": normalise_season(season) if season is not None else None,
                     "rows": records(rows, ROW_COLS), "summary": summary,
                     "warnings": quality_warnings(d, rows, teams=[t])})


# ---------------------------------------------------------------------------
def league_table(league: str, season: str, data: Dataset | None = None) -> dict:
    """
    Final standings from the CSV results. Ordered by Points, then GoalDiff, then GoalsFor,
    then team name. (Real competitions use other tie-breakers, e.g. head-to-head in La Liga
    and Serie A, and points deductions are not in the data; see `notes`.)
    """
    d = _ds(data)
    lg = resolve_league(league, d)
    if lg is None:
        raise ToolError("unknown_league", "league_table needs a single league.")
    s = resolve_season(season, d, lg)
    rows = d.ts[(d.ts["League"] == lg) & (d.ts["Season"] == s)].copy()
    rows = rows.sort_values(["Points", "GoalDiff", "GoalsFor", "Team"],
                            ascending=[False, False, False, True], kind="stable")
    rows["position"] = range(1, len(rows) + 1)
    return jsonable({
        "league": lg, "season": s,
        "table": records(rows, ["position", "Team", "Played", "Wins", "Draws", "Losses",
                                "GoalsFor", "GoalsAgainst", "GoalDiff", "Points"]),
        "notes": ["Tie-break order used: points, goal difference, goals scored, name.",
                  "Points deductions and competition-specific tie-breakers are not in the source data."],
        "warnings": quality_warnings(d, rows),
    })


# ---------------------------------------------------------------------------
def top_n(metric: str = "Points", n: int = 10, league: str | None = None, season: str | None = None,
          ascending: bool = False, min_played: int = 1, data: Dataset | None = None) -> dict:
    """Best (or worst, ascending=True) team-seasons by a metric, optionally within a league/season."""
    d = _ds(data)
    check_metric(metric)
    if n < 1:
        raise ToolError("invalid_argument", "n must be at least 1.")
    lg = resolve_league(league, d)
    rows = d.ts if lg is None else d.ts[d.ts["League"] == lg]
    if season is not None:
        rows = rows[rows["Season"] == resolve_season(season, d, lg)]
    rows = rows[rows["Played"] >= min_played]
    excluded = int(rows[metric].isna().sum())
    rows = rows[rows[metric].notna()]
    if rows.empty:
        raise ToolError("insufficient_data", f"No team-seasons have {metric} for this selection.")
    rows = rows.sort_values([metric, "Team"], ascending=[ascending, True], kind="stable").head(n).copy()
    rows["position"] = range(1, len(rows) + 1)
    w = quality_warnings(d, rows)
    if excluded:
        w.append(f"{excluded} team-seasons have no {metric} recorded and were excluded.")
    return jsonable({"metric": metric, "ascending": ascending, "league": lg, "season": normalise_season(season) if season is not None else None,
                     "results": records(rows, ["position", "League", "Season", "Team", "Played"]
                                        + ([metric] if metric != "Played" else [])),
                     "warnings": w})


# ---------------------------------------------------------------------------
def season_trend(team: str, metric: str = "Points", league: str | None = None,
                 season_start: str | None = None, season_end: str | None = None,
                 data: Dataset | None = None) -> dict:
    """A team's metric across seasons: series, change, best/worst season, linear trend per season."""
    d = _ds(data)
    check_metric(metric)
    lg = resolve_league(league, d)
    t = resolve_team(team, d, lg)
    lo, hi = season_bounds(season_start, season_end, d, lg)
    rows = d.ts[(d.ts["Team"] == t) & d.ts["SeasonStart"].between(lo, hi)]
    if lg:
        rows = rows[rows["League"] == lg]
    rows = _sorted(rows)
    valid = rows[rows[metric].notna()]
    if valid.empty:
        raise ToolError("insufficient_data", f"No {metric} values for {t} in this range.")
    first, last = valid.iloc[0], valid.iloc[-1]
    best = valid.loc[valid[metric].idxmax()]
    worst = valid.loc[valid[metric].idxmin()]
    trend = None
    if len(valid) >= 3 and valid["SeasonStart"].nunique() > 1:
        lr = sps.linregress(valid["SeasonStart"], valid[metric])
        trend = {"slope_per_season": lr.slope, "r2": lr.rvalue ** 2, "p_value": lr.pvalue, "n": len(valid)}
    return jsonable({
        "team": t, "metric": metric, "league": lg,
        "series": records(rows, ["Season", "League", "Played", metric]),
        "first": {"season": first["Season"], "value": first[metric]},
        "last": {"season": last["Season"], "value": last[metric]},
        "change": last[metric] - first[metric],
        "best": {"season": best["Season"], "value": best[metric]},
        "worst": {"season": worst["Season"], "value": worst[metric]},
        "mean": valid[metric].mean(), "linear_trend": trend,
        "warnings": quality_warnings(d, rows, teams=[t], metrics=[metric]),
    })


# ---------------------------------------------------------------------------
def head_to_head(team_a: str, team_b: str, season_start: str | None = None,
                 season_end: str | None = None, data: Dataset | None = None) -> dict:
    """Results of matches between two teams."""
    d = _ds(data)
    a, b = resolve_team(team_a, d), resolve_team(team_b, d)
    if a == b:
        raise ToolError("invalid_argument", "head_to_head needs two different teams.")
    lo, hi = season_bounds(season_start, season_end, d)
    m = d.matches
    m = m[m["SeasonStart"].between(lo, hi) &
          (((m["HomeTeam"] == a) & (m["AwayTeam"] == b)) | ((m["HomeTeam"] == b) & (m["AwayTeam"] == a)))]
    a_win = ((m["HomeTeam"] == a) & (m["FTR"] == "H")) | ((m["AwayTeam"] == a) & (m["FTR"] == "A"))
    b_win = ((m["HomeTeam"] == b) & (m["FTR"] == "H")) | ((m["AwayTeam"] == b) & (m["FTR"] == "A"))
    draw = m["FTR"] == "D"
    a_goals = (m["FTHG"].where(m["HomeTeam"] == a, m["FTAG"])).sum()
    b_goals = (m["FTHG"].where(m["HomeTeam"] == b, m["FTAG"])).sum()
    return jsonable({"team_a": a, "team_b": b, "matches": len(m),
                     "team_a_wins": int(a_win.sum()), "team_b_wins": int(b_win.sum()), "draws": int(draw.sum()),
                     "team_a_goals": a_goals, "team_b_goals": b_goals})


def team_comparison(teams: list[str], metric: str = "Points", league: str | None = None,
                    season_start: str | None = None, season_end: str | None = None,
                    data: Dataset | None = None) -> dict:
    """Compare 2-6 teams on a metric across seasons, with per-team summaries and head-to-head records."""
    d = _ds(data)
    check_metric(metric)
    if not 2 <= len(teams) <= 6:
        raise ToolError("invalid_argument", "team_comparison needs between 2 and 6 teams.")
    lg = resolve_league(league, d)
    resolved = [resolve_team(t, d, lg) for t in teams]
    if len(set(resolved)) != len(resolved):
        raise ToolError("invalid_argument", f"Duplicate teams after resolving names: {resolved}.")
    lo, hi = season_bounds(season_start, season_end, d, lg)
    rows = d.ts[d.ts["Team"].isin(resolved) & d.ts["SeasonStart"].between(lo, hi)]
    if lg:
        rows = rows[rows["League"] == lg]
    per_team = {}
    for t in resolved:
        r = rows[(rows["Team"] == t) & rows[metric].notna()]
        per_team[t] = None if r.empty else {
            "seasons": len(r), "mean": r[metric].mean(), "total": r[metric].sum(),
            "best": {"season": r.loc[r[metric].idxmax(), "Season"], "value": r[metric].max()},
            "worst": {"season": r.loc[r[metric].idxmin(), "Season"], "value": r[metric].min()},
        }
    common = set.intersection(*[set(rows.loc[rows["Team"] == t, "Season"]) for t in resolved]) if len(rows) else set()
    h2h = [head_to_head(x, y, season_start, season_end, data=d) for x, y in itertools.combinations(resolved, 2)]
    return jsonable({
        "teams": resolved, "metric": metric, "league": lg,
        "series": records(_sorted(rows), ["Season", "Team", "Played", metric]),
        "summary": per_team, "seasons_in_common": sorted(common, key=lambda s: int(s[:4])),
        "head_to_head": h2h,
        "warnings": quality_warnings(d, rows, teams=resolved, metrics=[metric]),
    })


# ---------------------------------------------------------------------------
def most_improved(league: str, season_from: str, season_to: str, metric: str = "Points",
                  n: int = 5, per_game: bool = False, data: Dataset | None = None) -> dict:
    """
    Teams in the league in BOTH seasons, ranked by change in metric. per_game=True divides by
    matches played first (use it when either season is incomplete in the data).
    """
    d = _ds(data)
    check_metric(metric)
    lg = resolve_league(league, d)
    if lg is None:
        raise ToolError("unknown_league", "most_improved needs a single league.")
    s1, s2 = resolve_season(season_from, d, lg), resolve_season(season_to, d, lg)
    if s1 == s2:
        raise ToolError("invalid_argument", "season_from and season_to must differ.")
    a = d.ts[(d.ts["League"] == lg) & (d.ts["Season"] == s1)].set_index("Team")
    b = d.ts[(d.ts["League"] == lg) & (d.ts["Season"] == s2)].set_index("Team")
    both = a.index.intersection(b.index)
    if len(both) == 0:
        raise ToolError("insufficient_data", f"No team played in {lg} in both {s1} and {s2}.")
    va, vb = a.loc[both, metric], b.loc[both, metric]
    if per_game:
        va, vb = va / a.loc[both, "Played"], vb / b.loc[both, "Played"]
    df = pd.DataFrame({"Team": both, "from_value": va.values, "to_value": vb.values})
    df["change"] = df["to_value"] - df["from_value"]
    df = df.dropna(subset=["change"]).sort_values(["change", "Team"], ascending=[False, True], kind="stable")
    rows = d.ts[(d.ts["League"] == lg) & d.ts["Season"].isin([s1, s2])]
    return jsonable({
        "league": lg, "season_from": s1, "season_to": s2, "metric": metric, "per_game": per_game,
        "teams_in_both": int(len(both)),
        "most_improved": df.head(n).to_dict("records"),
        "most_declined": df.sort_values(["change", "Team"], kind="stable").head(n).to_dict("records"),
        "warnings": quality_warnings(d, rows),
    })


# ---------------------------------------------------------------------------
def _home_away_block(m: pd.DataFrame) -> dict:
    n = len(m)
    hw, dr, aw = int((m["FTR"] == "H").sum()), int((m["FTR"] == "D").sum()), int((m["FTR"] == "A").sum())
    return {
        "matches": n, "home_wins": hw, "draws": dr, "away_wins": aw,
        "home_win_pct": 100 * hw / n, "draw_pct": 100 * dr / n, "away_win_pct": 100 * aw / n,
        "avg_home_goals": m["FTHG"].mean(), "avg_away_goals": m["FTAG"].mean(),
        "home_points_per_game": (3 * hw + dr) / n, "away_points_per_game": (3 * aw + dr) / n,
    }


def home_advantage(league: str | None = None, season_start: str | None = None,
                   season_end: str | None = None, team: str | None = None, by_season: bool = False,
                   data: Dataset | None = None) -> dict:
    """Home vs away results from match rows: win/draw/loss %, goals and points per game."""
    d = _ds(data)
    lg = resolve_league(league, d)
    lo, hi = season_bounds(season_start, season_end, d, lg)
    m = d.matches[d.matches["SeasonStart"].between(lo, hi)]
    if lg:
        m = m[m["League"] == lg]
    t = None
    if team is not None:
        t = resolve_team(team, d, lg)
        home, away = m[m["HomeTeam"] == t], m[m["AwayTeam"] == t]
        if home.empty and away.empty:
            raise ToolError("no_data", f"{t} has no matches in this selection.")
        hp = (3 * (home["FTR"] == "H").sum() + (home["FTR"] == "D").sum()) / max(len(home), 1)
        ap = (3 * (away["FTR"] == "A").sum() + (away["FTR"] == "D").sum()) / max(len(away), 1)
        return jsonable({
            "team": t, "league": lg,
            "home": {"matches": len(home), "wins": int((home["FTR"] == "H").sum()),
                     "draws": int((home["FTR"] == "D").sum()), "losses": int((home["FTR"] == "A").sum()),
                     "points_per_game": hp if len(home) else None,
                     "goals_for_per_game": home["FTHG"].mean(), "goals_against_per_game": home["FTAG"].mean()},
            "away": {"matches": len(away), "wins": int((away["FTR"] == "A").sum()),
                     "draws": int((away["FTR"] == "D").sum()), "losses": int((away["FTR"] == "H").sum()),
                     "points_per_game": ap if len(away) else None,
                     "goals_for_per_game": away["FTAG"].mean(), "goals_against_per_game": away["FTHG"].mean()},
        })
    if m.empty:
        raise ToolError("insufficient_data", "No matches in this selection.")
    out = {"league": lg, "overall": _home_away_block(m)}
    if by_season:
        out["by_season"] = [{"season": s, **_home_away_block(g)}
                            for s, g in sorted(m.groupby("Season"), key=lambda kv: int(kv[0][:4]))]
    out["warnings"] = quality_warnings(d, m[["League", "Season"]])
    return jsonable(out)


# ---------------------------------------------------------------------------
def data_coverage(league: str | None = None, data: Dataset | None = None) -> dict:
    """Which seasons exist, which are incomplete, and which have shots/fouls/cards recorded."""
    d = _ds(data)
    lg = resolve_league(league, d)
    cov = d.coverage if lg is None else d.coverage[d.coverage["League"] == lg]
    ts = d.ts if lg is None else d.ts[d.ts["League"] == lg]
    stat_cov = (ts.groupby(["League", "Season"], sort=False)[["AvgShots", "AvgFouls", "AvgYellowCards"]]
                .apply(lambda g: g.notna().any()).reset_index())
    merged = cov.merge(stat_cov, on=["League", "Season"], how="left").sort_values(["League", "SeasonStart"])
    out = {}
    for league_name, g in merged.groupby("League", sort=False):
        out[league_name] = {
            "seasons": f"{g['Season'].iloc[0]} to {g['Season'].iloc[-1]}", "n_seasons": len(g),
            "incomplete_seasons": [{"season": r.Season, "matches": int(r.matches), "expected": int(r.expected)}
                                   for r in g.itertuples() if not r.complete],
            "first_season_with_shots": next((r.Season for r in g.itertuples() if r.AvgShots), None),
            "first_season_with_fouls_cards": next((r.Season for r in g.itertuples() if r.AvgFouls and r.AvgYellowCards), None),
        }
    return jsonable({"leagues": out})
