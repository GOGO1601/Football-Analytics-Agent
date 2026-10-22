"""
Tests for tools/hypothesis.py.

The statistics are checked against independent calculations (closed-form values,
textbook identities such as z² = χ² for a 2x2 table) and against planted effects in
synthetic data, not just against scipy calling itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from tools import hypothesis
from tools.common import ToolError

from .conftest import make_dataset, make_matches


def _league_period(n_matches: int, home_win_rate: float, season: str, seed: int, draw_rate=0.25):
    """Synthetic matches with an exact number of H/D/A results."""
    rng = np.random.default_rng(seed)
    hw = round(n_matches * home_win_rate)
    dr = round(n_matches * draw_rate)
    results = ["H"] * hw + ["D"] * dr + ["A"] * (n_matches - hw - dr)
    rng.shuffle(results)
    rows = []
    for i, r in enumerate(results):
        hg, ag = {"H": (2, 0), "D": (1, 1), "A": (0, 1)}[r]
        rows.append((f"T{i % 20}", f"T{(i + 1) % 20}", hg, ag))
    return make_matches(rows, league="Synthetic", season=season)


@pytest.fixture
def planted():
    """Home win rate 50% across 2010-11..2011-12, 35% in 2020-21 (a large, known drop)."""
    return make_dataset(_league_period(400, 0.50, "2010-11", 1), _league_period(400, 0.50, "2011-12", 2),
                        _league_period(400, 0.35, "2020-21", 3))


@pytest.fixture
def no_change():
    return make_dataset(_league_period(400, 0.45, "2010-11", 4), _league_period(400, 0.45, "2020-21", 5))


# ---------------------------------------------------------------------------
# home_advantage_test
# ---------------------------------------------------------------------------
def test_planted_drop_is_detected(planted):
    r = hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2011-12",
                                       period_b_start="2020-21", period_b_end="2020-21", data=planted)
    A, B = r["period_a"], r["period_b"]
    assert (A["matches"], A["home_wins"], B["matches"], B["home_wins"]) == (800, 400, 400, 140)
    z = r["tests"]["home_win_rate_two_proportion_z"]
    assert z["difference"] == pytest.approx(0.35 - 0.50)
    assert z["ci95"][0] < -0.15 < z["ci95"][1]
    assert z["p_value"] < 1e-5
    assert "fell" in r["interpretation"] and "significant" in r["interpretation"]


def test_z_statistic_closed_form(planted):
    r = hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2011-12",
                                       period_b_start="2020-21", period_b_end="2020-21", data=planted)
    p1, p2, n1, n2 = 0.5, 0.35, 800, 400
    p = (400 + 140) / 1200
    z = (p2 - p1) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    t = r["tests"]["home_win_rate_two_proportion_z"]
    assert t["statistic"] == pytest.approx(z, rel=1e-12)
    assert t["cohens_h"] == pytest.approx(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)), rel=1e-12)


def test_z_squared_equals_uncorrected_chi_square(planted):
    """Identity for 2x2 tables: two-proportion z² == Pearson χ² without continuity correction."""
    r = hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2011-12",
                                       period_b_start="2020-21", period_b_end="2020-21", data=planted)
    chi2, *_ = sps.chi2_contingency([[400, 400], [140, 260]], correction=False)
    assert r["tests"]["home_win_rate_two_proportion_z"]["statistic"] ** 2 == pytest.approx(chi2, rel=1e-10)


def test_no_change_is_not_significant(no_change):
    r = hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2010-11",
                                       period_b_start="2020-21", period_b_end="2020-21", data=no_change)
    z = r["tests"]["home_win_rate_two_proportion_z"]
    assert z["difference"] == 0 and z["p_value"] == pytest.approx(1.0)
    assert r["tests"]["home_win_rate_fisher_exact"]["p_value"] == pytest.approx(1.0)
    assert z["ci95"][0] < 0 < z["ci95"][1]
    assert "not statistically significant" in r["interpretation"]


def test_covid_default_on_real_data(real):
    r = hypothesis.home_advantage_test(data=real)
    A, B = r["period_a"], r["period_b"]
    assert A["seasons"] == "2010-11 to 2018-19" and B["seasons"] == "2020-21 to 2020-21"
    assert B["home_win_rate"] < A["home_win_rate"]
    t = r["tests"]
    for k in ("home_win_rate_two_proportion_z", "home_win_rate_fisher_exact",
              "result_distribution_chi_square", "home_goal_margin_welch_t"):
        assert 0 <= t[k]["p_value"] <= 1
    # sample sizes are reported and consistent
    assert A["home_wins"] + A["draws"] + A["away_wins"] == A["matches"]


def test_period_b_minus_a_sign_convention(planted):
    fwd = hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2011-12",
                                         period_b_start="2020-21", period_b_end="2020-21", data=planted)
    rev = hypothesis.home_advantage_test(period_a_start="2020-21", period_a_end="2020-21",
                                         period_b_start="2010-11", period_b_end="2011-12", data=planted)
    f = fwd["tests"]["home_win_rate_two_proportion_z"]
    b = rev["tests"]["home_win_rate_two_proportion_z"]
    assert f["difference"] == pytest.approx(-b["difference"])
    assert f["p_value"] == pytest.approx(b["p_value"])


def test_overlapping_periods_rejected(real):
    with pytest.raises(ToolError) as e:
        hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2015-16",
                                       period_b_start="2015-16", period_b_end="2020-21", data=real)
    assert e.value.code == "invalid_argument"


def test_insufficient_matches_rejected():
    ds = make_dataset(_league_period(10, 0.5, "2010-11", 1), _league_period(10, 0.5, "2020-21", 2))
    with pytest.raises(ToolError) as e:
        hypothesis.home_advantage_test(period_a_start="2010-11", period_a_end="2010-11",
                                       period_b_start="2020-21", period_b_end="2020-21", data=ds)
    assert e.value.code == "insufficient_data"


def test_unknown_league_and_bad_season(real):
    with pytest.raises(ToolError) as e:
        hypothesis.home_advantage_test(league="MLS", data=real)
    assert e.value.code == "unknown_league"
    with pytest.raises(ToolError) as e:
        hypothesis.home_advantage_test(period_b_start="2040-41", period_b_end="2040-41", data=real)
    assert e.value.code == "unknown_season"


# ---------------------------------------------------------------------------
# hypothesis_test (two groups of team-seasons)
# ---------------------------------------------------------------------------
def test_welch_against_closed_form(real):
    r = hypothesis.hypothesis_test("Points", {"league": "La Liga", "season_start": "2010-11", "season_end": "2014-15"},
                                   {"league": "Serie A", "season_start": "2010-11", "season_end": "2014-15"}, data=real)
    ts = real.ts
    a = ts[(ts.League == "La Liga") & ts.SeasonStart.between(2010, 2014)]["Points"].to_numpy(float)
    b = ts[(ts.League == "Serie A") & ts.SeasonStart.between(2010, 2014)]["Points"].to_numpy(float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    w = r["tests"]["welch_t"]
    assert r["group_a"]["n"] == len(a) == 100 and r["group_b"]["n"] == len(b) == 100
    assert w["t_statistic"] == pytest.approx((b.mean() - a.mean()) / se, rel=1e-12)
    assert w["ci95"][0] < w["difference"] < w["ci95"][1]


def test_points_per_game_league_comparison_is_symmetric(real):
    a = {"league": "Bundesliga"}
    b = {"league": "Ligue 1"}
    r1 = hypothesis.hypothesis_test("PointsPerGame", a, b, data=real)
    r2 = hypothesis.hypothesis_test("PointsPerGame", b, a, data=real)
    assert r1["tests"]["welch_t"]["difference"] == pytest.approx(-r2["tests"]["welch_t"]["difference"])
    assert r1["tests"]["welch_t"]["p_value"] == pytest.approx(r2["tests"]["welch_t"]["p_value"])


def test_group_of_teams(real):
    r = hypothesis.hypothesis_test("GoalsFor",
                                   {"league": "Premier League", "teams": ["Arsenal", "Chelsea"], "season_end": "2009-10"},
                                   {"league": "Premier League", "teams": ["Arsenal", "Chelsea"], "season_start": "2010-11"},
                                   data=real)
    assert r["group_a"]["teams"] == ["Arsenal", "Chelsea"]
    assert r["group_a"]["n"] > 10 and r["group_b"]["n"] > 10


def test_overlapping_groups_rejected(real):
    with pytest.raises(ToolError) as e:
        hypothesis.hypothesis_test("Points", {"league": "La Liga"}, {"league": "La Liga", "season_start": "2010-11"}, data=real)
    assert e.value.code == "invalid_argument"


def test_group_too_small(real):
    with pytest.raises(ToolError) as e:
        hypothesis.hypothesis_test("Points", {"league": "La Liga", "teams": ["Girona"], "season_start": "2023-24",
                                              "season_end": "2023-24"},
                                   {"league": "Serie A", "season_start": "2023-24", "season_end": "2023-24"}, data=real)
    assert e.value.code == "insufficient_data"


def test_metric_missing_in_period(real):
    """No fouls data in the 1990s Premier League -> insufficient, not a test on zeros."""
    with pytest.raises(ToolError) as e:
        hypothesis.hypothesis_test("AvgFouls", {"league": "Premier League", "season_end": "1997-98"},
                                   {"league": "Premier League", "season_start": "2010-11"}, data=real)
    assert e.value.code == "insufficient_data"


def test_constant_metric_rejected():
    rows = [("A", "B", 1, 1), ("B", "A", 1, 1), ("C", "D", 1, 1), ("D", "C", 1, 1)]
    ds = make_dataset(make_matches(rows[:2], league="L1", season="2010-11"),
                      make_matches(rows[2:], league="L2", season="2010-11"))
    with pytest.raises(ToolError) as e:
        hypothesis.hypothesis_test("Points", {"league": "L1"}, {"league": "L2"}, data=ds)
    assert e.value.code == "insufficient_data"
