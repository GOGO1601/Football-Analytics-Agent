"""
Unit tests for the loader, the aggregate port and the descriptive/correlation tools.

Two kinds of ground truth:
  * real-data facts that are true independently of this code (e.g. Arsenal's unbeaten
    2003-04 season), so the tests cannot pass just because code and test share a bug
  * small synthetic datasets whose answers are worked out by hand (see conftest.mini)
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from data import aggregate as agg
from data.loader import discover, js_nullable_number, js_number, load_matches, season_label
from tools import correlation, stats
from tools.common import Dataset, ToolError, normalise_season
from tools.registry import TOOLS, call_tool, list_tools

from .conftest import make_dataset, make_matches


# ===========================================================================
# Loader
# ===========================================================================
@pytest.mark.parametrize("raw,expected", [
    ("3", 3.0), (" 3 ", 3.0), ("", 0.0), ("  ", 0.0), ("1.5", 1.5), ("1e2", 100.0),
    ("0x10", 16.0), ("Infinity", math.inf), ("abc", math.nan), ("1_000", math.nan), (None, math.nan),
])
def test_js_number_matches_javascript_unary_plus(raw, expected):
    got = js_number(raw)
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


@pytest.mark.parametrize("raw,is_null", [("", True), (None, True), ("0", False), ("4", False)])
def test_nullable_number(raw, is_null):
    assert math.isnan(js_nullable_number(raw)) == is_null


@pytest.mark.parametrize("code,label", [("9394", "1993-94"), ("9900", "1999-00"), ("0001", "2000-01"),
                                        ("2526", "2025-26")])
def test_season_labels_match_js_all_seasons(code, label):
    assert season_label(code) == label


def test_discovers_all_shipped_csvs():
    files = discover()
    assert len(files) == 165
    assert {f.league for f in files} == {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")


def test_loader_handles_missing_columns_empty_cells_and_bad_rows(tmp_path):
    hdr = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]  # no shots/cards columns at all
    _write_csv(tmp_path / "premier-league" / "season-9394.csv", hdr, [
        ["01/01/94", "A", "B", "2", "1", "H"],
        ["02/01/94", "B", "A", "", "1", "A"],      # empty goals -> 0 (JS +"" === 0)
        ["03/01/94", "A", "", "1", "1", "D"],      # empty AwayTeam -> dropped
        ["04/01/94", "B", "A", "1", "1", ""],      # empty FTR -> dropped
    ])
    m = load_matches(tmp_path)
    assert len(m) == 4 and m["HS"].isna().all()
    ts = agg.team_seasons(m)
    a = ts[ts["Team"] == "A"].iloc[0]
    assert (a["Played"], a["Wins"], a["Points"], a["GoalsFor"], a["GoalsAgainst"]) == (2, 2, 6, 3, 1)
    assert math.isnan(a["AvgShots"])


def test_loader_on_empty_folder(tmp_path):
    (tmp_path / "premier-league").mkdir()
    assert load_matches(tmp_path).empty
    assert agg.team_seasons(load_matches(tmp_path)).empty


def test_loader_missing_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        load_matches(tmp_path / "nope")


# ===========================================================================
# Aggregation (hand-checked synthetic data)
# ===========================================================================
def test_mini_table_by_hand(mini):
    t = mini.ts.set_index("Team")
    assert t.loc["A", ["Played", "Wins", "Draws", "Losses", "Points", "GoalsFor", "GoalsAgainst"]].tolist() == [4, 3, 1, 0, 10, 7, 2]
    assert t.loc["B", ["Points", "GoalsFor", "GoalsAgainst", "GoalDiff"]].tolist() == [4, 3, 4, -1]
    assert t.loc["C", ["Points", "GoalDiff"]].tolist() == [2, -4]


def test_null_mean_skips_missing_matches(mini):
    t = mini.ts.set_index("Team")
    assert t.loc["A", "AvgShots"] == 9.75
    assert t.loc["B", "AvgShots"] == (3 + 5 + 11) / 3      # match with missing shots is ignored, not 0
    assert t.loc["C", "AvgShots"] == (6 + 7 + 4) / 3
    assert t.loc["B", "AvgRedCards"] == 0.25


def test_unexpected_ftr_counts_as_loss_for_both_teams_like_js():
    ds = make_dataset(make_matches([("A", "B", 1, 1, "X")]))
    t = ds.ts.set_index("Team")
    assert t.loc["A", "Losses"] == 1 and t.loc["B", "Losses"] == 1 and t["Points"].sum() == 0


def test_q3_filter_requires_red_cards():
    """The real data never has fouls+yellows without reds, so parity alone cannot catch this."""
    ts = pd.DataFrame([
        {"League": "L", "Season": "2020-21", "Team": "A", "Points": 50, "AvgFouls": 10.0,
         "AvgYellowCards": 2.0, "AvgRedCards": float("nan"), "SeasonStart": 2020},
        {"League": "L", "Season": "2020-21", "Team": "B", "Points": 40, "AvgFouls": 12.0,
         "AvgYellowCards": 2.5, "AvgRedCards": 0.1, "SeasonStart": 2020},
    ])
    assert agg.q3_filter(ts, "AvgFouls")["Team"].tolist() == ["B"]


def test_linear_regression_edge_cases():
    one = pd.DataFrame({"x": [1.0], "y": [2.0]})
    flat_x = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]})
    flat_y = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]})
    exact = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]})
    assert agg.linear_regression(one, "x", "y") is None
    assert agg.linear_regression(flat_x, "x", "y") is None
    assert agg.linear_regression(flat_y, "x", "y")["r2"] == 0
    r = agg.linear_regression(exact, "x", "y")
    assert (r["slope"], r["intercept"], r["r2"]) == (2.0, 1.0, 1.0)


# ===========================================================================
# Realistic questions on the real data (independently known facts)
# ===========================================================================
def test_most_points_in_a_premier_league_season(real):
    r = stats.top_n("Points", n=1, league="Premier League", data=real)
    top = r["results"][0]
    assert (top["Team"], top["Season"], top["Points"]) == ("Man City", "2017-18", 100)


def test_league_table_winner_2004_05(real):
    t = stats.league_table("Premier League", "2004-05", data=real)["table"]
    assert (t[0]["Team"], t[0]["Points"], t[0]["GoalsAgainst"]) == ("Chelsea", 95, 15)
    assert len(t) == 20 and [r["position"] for r in t] == list(range(1, 21))


def test_arsenal_invincibles(real):
    r = stats.team_season_stats("Arsenal", season="2003-04", data=real)["rows"][0]
    assert (r["Played"], r["Wins"], r["Draws"], r["Losses"], r["Points"]) == (38, 26, 12, 0, 90)


def test_record_points_in_other_leagues(real):
    juve = stats.team_season_stats("Juventus", season="2013-14", data=real)["rows"][0]
    barca = stats.team_season_stats("Barcelona", season="2012-13", data=real)["rows"][0]
    assert juve["Points"] == 102 and barca["Points"] == 100


def test_team_average_goals(real):
    r = stats.team_season_stats("Arsenal", season="2003-04", data=real)
    assert r["rows"][0]["GoalsFor"] == 73
    assert r["summary"]["goals_for_per_game"] == pytest.approx(73 / 38)


def test_most_improved_girona_2023_24(real):
    r = stats.most_improved("La Liga", "2022-23", "2023-24", data=real)
    best = r["most_improved"][0]
    assert best["Team"] == "Girona" and best["change"] == best["to_value"] - best["from_value"] > 0


def test_home_win_percentage_definition(real):
    r = stats.home_advantage(league="Premier League", season_start="2004-05", season_end="2004-05", data=real)["overall"]
    assert r["matches"] == 380
    assert r["home_wins"] + r["draws"] + r["away_wins"] == 380
    assert r["home_win_pct"] == pytest.approx(100 * r["home_wins"] / 380)
    assert r["home_win_pct"] + r["draw_pct"] + r["away_win_pct"] == pytest.approx(100)


def test_team_name_aliases_resolve(real):
    assert stats.team_season_stats("Manchester United", season="1998-99", data=real)["team"] == "Man United"
    assert stats.team_season_stats("atlético madrid", season="2013-14", data=real)["team"] == "Ath Madrid"


def test_season_trend(real):
    r = stats.season_trend("Arsenal", "Points", season_start="2003-04", season_end="2005-06", data=real)
    assert [p["Season"] for p in r["series"]] == ["2003-04", "2004-05", "2005-06"]
    assert r["best"] == {"season": "2003-04", "value": 90}
    assert r["change"] == r["last"]["value"] - r["first"]["value"]


def test_team_comparison_and_head_to_head(real):
    r = stats.team_comparison(["Arsenal", "Tottenham"], season_start="2010-11", season_end="2014-15", data=real)
    h = r["head_to_head"][0]
    assert h["matches"] == 10
    assert h["team_a_wins"] + h["team_b_wins"] + h["draws"] == 10
    assert len(r["seasons_in_common"]) == 5


def test_incomplete_season_is_flagged(real):
    r = stats.league_table("Premier League", "2018-19", data=real)
    assert any("incomplete" in w for w in r["warnings"])


def test_spelling_variant_is_flagged(real):
    r = stats.team_season_stats("Villarreal", data=real)
    assert any("Villareal" in w for w in r["warnings"])


def test_shots_vs_points_relationship(real):
    r = correlation.correlation_analysis("AvgShotsOnTarget", "Points", data=real)
    assert r["r"] > 0.5 and r["p_value"] < 1e-10 and r["ci95"][0] < r["r"] < r["ci95"][1]
    assert r["r_squared"] == pytest.approx(r["dashboard_regression"]["r2"], rel=1e-9)


def test_cards_vs_points_relationship_uses_q3_rows(real):
    r = correlation.correlation_analysis("AvgYellowCards", "Points", require_discipline_fields=True, data=real)
    q3 = agg.q3_summary(real.ts, "AvgYellowCards")
    assert r["n"] == q3["n"]
    assert r["dashboard_regression"]["r2_label"] == q3["r2_label"]
    assert r["r"] < 0


def test_regression_matches_dashboard_q2(real):
    r = correlation.regression_analysis("GoalsFor", "Points", data=real)
    q2 = agg.q2_summary(real.ts, "GoalsFor")
    assert r["slope"] == pytest.approx(q2["regression"]["slope"], rel=1e-10)
    assert r["slope_ci95"][0] < r["slope"] < r["slope_ci95"][1]


def test_spearman(real):
    r = correlation.correlation_analysis("AvgShots", method="spearman", league="Serie A", data=real)
    assert -1 <= r["r"] <= 1 and r["method"] == "spearman"


# ===========================================================================
# Edge cases
# ===========================================================================
def test_unknown_team_gives_suggestions(real):
    with pytest.raises(ToolError) as e:
        stats.team_season_stats("Arsenl", data=real)
    assert e.value.code == "unknown_team" and "Arsenal" in e.value.suggestions


def test_team_in_wrong_league(real):
    with pytest.raises(ToolError) as e:
        stats.team_season_stats("Arsenal", league="Serie A", data=real)
    assert e.value.code == "unknown_team"


def test_unknown_league(real):
    with pytest.raises(ToolError) as e:
        stats.league_table("Eredivisie", "2004-05", data=real)
    assert e.value.code == "unknown_league"


@pytest.mark.parametrize("season,code", [("2004-06", "invalid_season"), ("last year", "invalid_season"),
                                         ("1980-81", "unknown_season"), ("2030", "unknown_season")])
def test_invalid_and_unknown_seasons(real, season, code):
    with pytest.raises(ToolError) as e:
        stats.league_table("Premier League", season, data=real)
    assert e.value.code == code


@pytest.mark.parametrize("raw,norm", [("2004-05", "2004-05"), ("2004/05", "2004-05"), ("2004/2005", "2004-05"),
                                      (2004, "2004-05"), ("1999-00", "1999-00"), ("1999/2000", "1999-00")])
def test_season_normalisation(raw, norm):
    assert normalise_season(raw) == norm


def test_season_range_reversed(real):
    with pytest.raises(ToolError) as e:
        stats.season_trend("Arsenal", season_start="2010-11", season_end="2005-06", data=real)
    assert e.value.code == "invalid_season"


def test_unknown_metric(real):
    with pytest.raises(ToolError) as e:
        stats.top_n("xG", data=real)
    assert e.value.code == "unknown_metric"


def test_metric_missing_in_old_seasons(real):
    """Shots were not recorded in the 1990s: the tool must say so rather than return zeros."""
    with pytest.raises(ToolError) as e:
        stats.top_n("AvgShots", league="Premier League", season="1995-96", data=real)
    assert e.value.code == "insufficient_data"


def test_correlation_insufficient_observations():
    ds = make_dataset(make_matches([("A", "B", 1, 0)], stats={"HS": [5], "AS": [3]}))
    with pytest.raises(ToolError) as e:
        correlation.correlation_analysis("AvgShots", data=ds)
    assert e.value.code == "insufficient_data"


def test_empty_dataset():
    ds = Dataset.from_matches(make_matches([("A", "B", 1, 0)]).iloc[0:0])
    with pytest.raises(ToolError):
        stats.top_n("Points", data=ds)
    with pytest.raises(ToolError):
        stats.home_advantage(data=ds)


def test_most_improved_same_season(real):
    with pytest.raises(ToolError):
        stats.most_improved("La Liga", "2020-21", "2020-21", data=real)


# ===========================================================================
# Registry
# ===========================================================================
def test_registry_schemas_hide_injected_data():
    for s in list_tools():
        assert "data" not in s["parameters"].get("properties", {})
        assert s["description"]


def test_registry_success_and_error_envelopes(mini):
    ok = call_tool("league_table", {"league": "Test League", "season": "2020-21"}, data=mini)
    assert ok["ok"] and ok["result"]["table"][0]["Team"] == "A"
    bad = call_tool("league_table", {"league": "Nope", "season": "2020-21"}, data=mini)
    assert not bad["ok"] and bad["error"]["code"] == "unknown_league"
    extra = call_tool("league_table", {"league": "Test League", "season": "2020-21", "foo": 1}, data=mini)
    assert not extra["ok"] and extra["error"]["code"] == "invalid_arguments"
    assert call_tool("nope", {})["error"]["code"] == "unknown_tool"


def test_every_tool_output_is_json_serialisable(real):
    import json
    calls = {
        "team_season_stats": {"team": "Arsenal", "season": "2003-04"},
        "league_table": {"league": "Serie A", "season": "2010-11"},
        "top_n": {"metric": "GoalsFor", "n": 3},
        "season_trend": {"team": "Lyon"},
        "team_comparison": {"teams": ["Real Madrid", "Barcelona"], "season_start": "2015-16"},
        "head_to_head": {"team_a": "Milan", "team_b": "Inter"},
        "most_improved": {"league": "Bundesliga", "season_from": "2020-21", "season_to": "2021-22"},
        "home_advantage": {"team": "Liverpool"},
        "data_coverage": {},
        "correlation_analysis": {"x_metric": "AvgFouls", "require_discipline_fields": True},
        "regression_analysis": {"x_metric": "AvgShotsOnTarget"},
        "home_advantage_test": {"league": "Bundesliga"},
        "hypothesis_test": {"metric": "Points", "group_a": {"league": "La Liga"}, "group_b": {"league": "Serie A"}},
    }
    assert set(calls) == set(TOOLS)
    for name, args in calls.items():
        r = call_tool(name, args, data=real)
        assert r["ok"], (name, r.get("error"))
        json.dumps(r, allow_nan=False)
