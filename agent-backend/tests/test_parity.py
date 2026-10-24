"""
JavaScript <-> Python parity.

The reference values in tests/fixtures/js_aggregates.json were produced by running the
real dashboard JS in a browser (tests/fixtures/generate_js_fixture.py). Nothing in the
fixture is hand-written.

Tolerances
  * counts, wins/draws/losses, points, goals, row keys, display strings: exact
  * team-season averages: exact (bit-for-bit) — the Python port performs the same
    operations in the same order as d3.mean, so any difference is a real logic bug
  * regression slope/intercept/R²: relative 1e-12. These come from long sums over up to
    3,204 points; the port uses the same sequential order as d3.sum so they are expected
    to be exact too, and 1e-12 only allows a few ULPs of noise. A genuine logic error
    (wrong filter, wrong null handling) moves R² by orders of magnitude more.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from data import aggregate as agg

FIXTURE = Path(__file__).parent / "fixtures" / "js_aggregates.json"
FRONTEND_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"
REL_TOL = 1e-12

JS = json.loads(FIXTURE.read_text())
KEY = ("League", "Team", "Season")


@pytest.fixture(scope="module")
def ts() -> pd.DataFrame:
    return agg.team_seasons()


def _norm(v):
    if v in ("__undefined__", "__NaN__"):
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _classify(py, js) -> str:
    if (py is None) != (js is None):
        return "null handling (one side is null)"
    if isinstance(py, (int, float)) and isinstance(js, (int, float)):
        if float(py).is_integer() and float(js).is_integer():
            return "aggregation/filtering (integer values differ)"
        if abs(py - js) < 1e-9 * max(1, abs(js)):
            return "floating-point summation order / rounding"
        return "aggregation (mean/sum differs)"
    return "parsing/type difference"


def _compare_rows(py_df: pd.DataFrame, js_rows: list[dict], label: str):
    py_idx = {tuple(r[k] for k in KEY): r for r in py_df.to_dict("records")}
    js_idx = {tuple(r[k] for k in KEY): r for r in js_rows}
    missing, extra = js_idx.keys() - py_idx.keys(), py_idx.keys() - js_idx.keys()
    assert not missing, f"[{label}] rows in JS but not Python: {sorted(missing)[:5]}"
    assert not extra, f"[{label}] rows in Python but not JS: {sorted(extra)[:5]}"
    problems = []
    for k, jr in js_idx.items():
        pr = py_idx[k]
        for col in agg.TEAM_SEASON_COLUMNS:
            if col in KEY:
                continue
            p, j = _norm(pr[col]), _norm(jr[col])
            if p != j:
                problems.append(f"{col} @ {k[0]} / {k[2]} / {k[1]}: python={p!r} js={j!r} -> {_classify(p, j)}")
    assert not problems, f"[{label}] {len(problems)} mismatches, first 10:\n  " + "\n  ".join(problems[:10])


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=REL_TOL)


def _assert_regression(py: dict | None, js: dict | None, ctx: str):
    assert (py is None) == (js is None), f"{ctx}: regression None mismatch python={py} js={js}"
    if py is None:
        return
    for k in ("slope", "intercept", "r2"):
        assert _close(py[k], js[k]), f"{ctx}: {k} python={py[k]!r} js={js[k]!r} diff={py[k]-js[k]:.3e}"


def _keys_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(agg.row_keys(df)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------
def test_fixture_matches_current_js_source():
    """If the JS changed since the fixture was generated, the fixture is stale."""
    for name, sha in JS["meta"]["source_sha256"].items():
        now = hashlib.sha256((FRONTEND_SRC / name).read_bytes()).hexdigest()[:16]
        assert now == sha, (f"{name} changed since the fixture was generated; rerun "
                            "python tests/fixtures/generate_js_fixture.py")


def test_fixture_has_no_js_runtime_errors():
    assert JS["q1_errors"] == [] and JS["q2_errors"] == [] and JS["q3_errors"] == []


# ---------------------------------------------------------------------------
# Team-season aggregation (data-loader.js and main.js pipelines)
# ---------------------------------------------------------------------------
LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]


@pytest.mark.parametrize("league", LEAGUES)
def test_data_loader_aggregates(ts, league):
    _compare_rows(ts[ts["League"] == league], [r for r in JS["data_loader"] if r["League"] == league],
                  f"data-loader.js {league}")


@pytest.mark.parametrize("league", LEAGUES)
def test_main_js_aggregates(ts, league):
    _compare_rows(ts[ts["League"] == league], [r for r in JS["q1"] if r["League"] == league],
                  f"main.js {league}")


def test_row_order_matches_d3_group(ts):
    """d3.group preserves first-appearance order; the Python output must be in the same order."""
    assert [tuple(r) for r in ts[list(KEY)].itertuples(index=False)] == \
           [tuple(r[k] for k in KEY) for r in JS["data_loader"]]


def test_total_row_count(ts):
    assert len(ts) == len(JS["data_loader"]) == len(JS["q1"])


# ---------------------------------------------------------------------------
# Q1 defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("league", LEAGUES)
def test_q1_team_list_and_default_selection(ts, league):
    ref = JS["q1_defaults"][league]
    assert agg.q1_teams(ts, league) == ref["teams"]
    assert agg.q1_default_selection(ts, league) == ref["selected"]


# ---------------------------------------------------------------------------
# Q2 filters, summary cards and regression
# ---------------------------------------------------------------------------
def _q2_id(c):
    return f"{c['x_metric']}-{c['league']}-{c['season_start']}..{c['season_end']}"


@pytest.mark.parametrize("case", JS["q2_cases"], ids=_q2_id)
def test_q2_case(ts, case):
    ctx = f"Q2 {_q2_id(case)}"
    py = agg.q2_summary(ts, case["x_metric"], case["league"], case["season_start"], case["season_end"])
    f = agg.q2_filter(ts, case["x_metric"], case["league"], case["season_start"], case["season_end"])
    assert py["n"] == case["n"], f"{ctx}: filtered row count python={py['n']} js={case['n']}"
    assert _keys_sha(f) == case["keys_sha256"], f"{ctx}: same count but different rows selected"
    for card in ("card_count", "card_seasons", "card_missing", "r2_label"):
        assert py[card] == case[card], f"{ctx}: {card} python={py[card]!r} js={case[card]!r}"
    _assert_regression(py["regression"], case["regression"], ctx)


# ---------------------------------------------------------------------------
# Q3 filters, summary cards and regression
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", JS["q3_cases"], ids=lambda c: f"{c['x_metric']}-{c['league']}")
def test_q3_case(ts, case):
    ctx = f"Q3 {case['x_metric']} {case['league']}"
    py = agg.q3_summary(ts, case["x_metric"], case["league"])
    f = agg.q3_filter(ts, case["x_metric"], case["league"])
    assert py["n"] == case["n"], f"{ctx}: filtered row count python={py['n']} js={case['n']}"
    assert _keys_sha(f) == case["keys_sha256"], f"{ctx}: same count but different rows selected"
    for card in ("card_count", "card_seasons", "card_fouls", "card_yellow", "r2_label"):
        assert py[card] == case[card], f"{ctx}: {card} python={py[card]!r} js={case[card]!r}"
    assert _close(py["mean_fouls"], case["mean_fouls"]), ctx
    assert _close(py["mean_yellow"], case["mean_yellow"]), ctx
    _assert_regression(py["regression"], case["regression"], ctx)


# ---------------------------------------------------------------------------
# JS display helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x,digits,expected", [
    (0.125, 2, "0.13"),     # exact binary tie: JS rounds half-up, Python format() gives 0.12
    (2.5, 0, "3"),          # Python format(2.5, '.0f') == '2'
    (1.005, 2, "1.00"),     # 1.005 is really 1.00499999999999989... in binary
    (0.7445945, 3, "0.745"),
    (-0.0001, 3, "-0.000"),
    (12.0, 1, "12.0"),
])
def test_js_to_fixed(x, digits, expected):
    assert agg.js_to_fixed(x, digits) == expected
