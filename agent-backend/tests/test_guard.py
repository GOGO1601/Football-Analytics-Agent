"""
Stage 6: the grounding guard. Deterministic checks of a draft against stored tool results.
Tool results come from the real registry on the real data; drafts are written by hand.
No LLM, no API key, no network.
"""
from __future__ import annotations

import inspect
import json
from operator import add
from typing import get_type_hints

import pytest

import tools.registry as registry
from agents import guard as G
from agents import synthesiser as syn
from agents.state import AgentState, GroundingIssue, GroundingResult, new_state


# ---------------------------------------------------------------------------
# Helpers and real tool results
# ---------------------------------------------------------------------------
def result(call_id: str, tool: str, args: dict, agent: str = "trend") -> dict:
    out = registry.call_tool(tool, args)
    return {"call_id": call_id, "agent": agent, "tool": tool, "args": args, "ok": out["ok"],
            "result": out.get("result") if out["ok"] else None, "error": None if out["ok"] else out["error"]}


def run(results, answer, evidence=(), stats=(), explanation="", question="q") -> GroundingResult:
    s = new_state(question)
    s.update(tool_results=list(results), draft_answer={"answer": answer, "evidence": list(evidence),
                                                       "key_stats": list(stats), "explanation": explanation})
    return G.check(s)


def kinds(g: GroundingResult) -> list[str]:
    return [i["kind"] for i in g["issues"]]


def stat(label, value, source):
    return {"label": label, "value": value, "source": source}


@pytest.fixture(scope="module")
def covid():        # Serie A: p = 0.0841, not significant
    return result("trend-1", "home_advantage_test", {"league": "Serie A"})


@pytest.fixture(scope="module")
def covid_all():    # all leagues: significant
    return result("trend-4", "home_advantage_test", {})


@pytest.fixture(scope="module")
def corr():         # r = 0.596685, p = 1.26e-214, n = 2224
    return result("offensive-1", "correlation_analysis", {"x_metric": "AvgShots"}, "offensive")


@pytest.fixture(scope="module")
def reg():          # slope 0.897622, CI [0.879407, 0.915838], R² 0.744595
    return result("offensive-2", "regression_analysis", {"x_metric": "GoalsFor"}, "offensive")


@pytest.fixture(scope="module")
def groups():       # Serie A minus La Liga points: difference -2.278, p = 0.0131, CI [-4.077, -0.479]
    return result("trend-5", "hypothesis_test",
                  {"metric": "Points", "group_a": {"league": "La Liga"}, "group_b": {"league": "Serie A"}})


@pytest.fixture(scope="module")
def table_1819():   # incomplete season: Liverpool top with 54 points from 21 games
    return result("trend-2", "league_table", {"league": "Premier League", "season": "2018-19"})


@pytest.fixture(scope="module")
def arsenal():      # 2003-04: 26 W, 12 D, 0 L, 90 pts, 73 goals
    return result("trend-3", "team_season_stats", {"team": "Arsenal", "season": "2003-04"})


# ---------------------------------------------------------------------------
# 1  correct draft
# ---------------------------------------------------------------------------
def test_correct_draft_passes(covid):
    g = run([covid],
            "No. Serie A's home win rate fell from 45.4% to 40.8% in 2020-21, but the change is not "
            "statistically significant (p = 0.0841).",
            ["Two-proportion z = -1.73; 95% CI for the change -9.9 to +0.6 percentage points.", "Cohen's h = -0.094."],
            [stat("p-value, home win rate change", "0.0841", "trend-1")],
            "Compares 2010-11 to 2018-19 with 2020-21.")
    assert g["passed"], g["feedback"]
    assert g["checked"] >= 8 and g["unsupported"] == [] and g["issues"] == []


# ---------------------------------------------------------------------------
# 2-3  correlation values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("written", ["0.597", "0.5967", "0.596685", "0.60", "0.6"])
def test_rounded_correlation_passes(corr, written):
    assert run([corr], f"Shots and points are associated (r = {written}).")["passed"]


@pytest.mark.parametrize("written", ["0.9", "0.59", "0.7", "0.5"])
def test_unsupported_correlation_fails(corr, written):
    g = run([corr], f"Shots and points are associated (r = {written}).")
    assert kinds(g) == ["unsupported_number"] and g["unsupported"] == [written]
    assert "0.596685" in g["feedback"]                       # tells the model the real value


# ---------------------------------------------------------------------------
# 4-5  percentages
# ---------------------------------------------------------------------------
def test_percentage_conversion_passes(covid):
    assert run([covid], "Home wins fell from 45.4% to 40.8%; not statistically significant.")["passed"]


def test_whole_percent_rounding_passes(covid):
    assert run([covid], "Home wins fell from 45% to 41%; not statistically significant.")["passed"]


def test_incorrectly_rounded_percentage_fails(covid):
    g = run([covid], "Home wins fell from 45.5% to 40.8%; not statistically significant.")
    assert g["unsupported"] == ["45.5%"]
    assert "0.454489" in g["feedback"] and "period_a.home_win_rate" in g["feedback"]


def test_percent_sign_required_for_scaling(covid):
    """0.407895 may be written as 40.79%, but a bare 40.79 is not silently read as 0.407895 x 100."""
    assert run([covid], "The 2020-21 home win rate was 40.79%; not statistically significant.")["passed"]
    assert not run([covid], "The 2020-21 home win rate was 40.79; not statistically significant.")["passed"]


# ---------------------------------------------------------------------------
# 6-10  negatives, scientific notation, p-values, intervals
# ---------------------------------------------------------------------------
def test_supported_negative_coefficient_passes(groups):
    assert run([groups], "Serie A teams average fewer points: difference -2.278 (Cohen's d = -0.138); "
                         "significant at the 5% level (p = 0.0131).")["passed"]


def test_negative_sign_matters(groups):
    g = run([groups], "The difference is +2.278 points; significant (p = 0.0131).")
    assert g["unsupported"] == ["+2.278"]


def test_magnitude_with_direction_word_passes(covid):
    assert run([covid], "The home win rate fell by 4.7 percentage points; not statistically significant.")["passed"]


def test_magnitude_with_wrong_direction_fails(covid):
    g = run([covid], "The home win rate rose by 4.7 percentage points; not statistically significant.")
    assert "direction_contradiction" in kinds(g)


@pytest.mark.parametrize("p", ["1.26e-214", "1.258e-214", "1.26 × 10^-214", "1.26×10⁻²¹⁴", "< 0.001"])
def test_scientific_notation_p_value_passes(corr, p):
    sep = " " if p.startswith("<") else " = "
    assert run([corr], f"Shots are associated with points (r = 0.597, p{sep}{p}).")["passed"]


@pytest.mark.parametrize("p", ["0.01", "0.05", "0.1", "1.26e-200"])
def test_incorrect_p_value_fails(covid, p):
    g = run([covid], f"Not statistically significant (p = {p}).")
    assert "unsupported_number" in kinds(g) and "0.0841462" in g["feedback"]


def test_p_value_must_match_p_value_evidence(corr):
    """0.597 is a real number in this result (r), but not as a p-value."""
    g = run([corr], "The correlation has p = 0.597.")
    assert g["unsupported"] == ["0.597"] and "p-value" in g["issues"][0]["detail"]


def test_supported_confidence_interval_passes(reg):
    assert run([reg], "Each extra goal is associated with +0.898 points (95% CI +0.879 to +0.916); R² = 0.745.")["passed"]


def test_fabricated_confidence_interval_fails(reg):
    g = run([reg], "Each extra goal is associated with +0.898 points (95% CI 0.80 to 0.99).")
    assert g["unsupported"] == ["0.80", "0.99"]
    assert all("confidence-interval" in i["detail"] for i in g["issues"])


def test_percentage_point_interval_passes(covid):
    assert run([covid], "Change -4.7 pp (95% CI -9.9 to +0.6 pp); not statistically significant.")["passed"]


# ---------------------------------------------------------------------------
# 11-13  counts, seasons, numbers from the question
# ---------------------------------------------------------------------------
def test_correct_counts_pass(arsenal):
    assert run([arsenal], "Arsenal went unbeaten in 2003-04: 26 wins, 12 draws, 0 losses, 90 points and 73 goals.")["passed"]


@pytest.mark.parametrize("wrong", ["92 points", "75 goals", "27 wins"])
def test_incorrect_points_or_goals_fail(arsenal, wrong):
    g = run([arsenal], f"Arsenal finished 2003-04 with {wrong}.")
    assert g["unsupported"] == [wrong.split()[0]]


def test_seasons_and_years_are_not_statistics(arsenal):
    g = run([arsenal], "In 2003-04 (and in the 2003/2004 season, 20 years after 1984) Arsenal had 90 points; "
                       "since 2010 they have not repeated it.")
    assert kinds(g) == ["unsupported_number"] and g["unsupported"] == ["20"]   # only the invented '20 years'


def test_question_numbers_are_allowed(arsenal):
    g = run([arsenal], "Here are Arsenal's top 5 figures: 90 points and 26 wins.",
            question="Show me Arsenal's top 5 figures for 2003-04")
    assert g["passed"], g["feedback"]


def test_question_numbers_do_not_cover_typed_claims(corr):
    """A p-value in the question is not evidence for a p-value claim."""
    g = run([corr], "Yes, p = 0.03.", question="Is it true that p = 0.03 for shots vs points?")
    assert g["unsupported"] == ["0.03"]


def test_team_and_league_names_with_digits():
    r = result("trend-6", "team_season_stats", {"team": "Schalke 04", "season": "2010-11"})
    pts = r["result"]["rows"][0]["Points"]
    assert run([r], f"Schalke 04 earned {pts} points in the Bundesliga in 2010-11.")["passed"]
    lg = result("trend-7", "league_table", {"league": "Ligue 1", "season": "2010-11"})
    top = lg["result"]["table"][0]
    assert run([lg], f"{top['Team']} won Ligue 1 in 2010-11 with {top['Points']} points.")["passed"]


# ---------------------------------------------------------------------------
# 14-16  key stats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["trend-99", "", "offensive-1"])
def test_invalid_key_stat_source_fails(arsenal, source):
    g = run([arsenal], "Arsenal had 90 points.", stats=[stat("Points", "90", source)])
    assert kinds(g) == ["invalid_key_stat_source"] and "trend-3" in g["feedback"]


def test_key_stat_from_failed_tool_fails(arsenal):
    failed = result("trend-8", "team_season_stats", {"team": "Arsenl"})
    g = run([arsenal, failed], "Arsenal had 90 points; Arsenl could not be found.",
            stats=[stat("Points", "90", "trend-8")])
    assert kinds(g) == ["invalid_key_stat_source"] and "failed" in g["issues"][0]["detail"]


def test_key_stat_value_not_in_its_source_fails(arsenal, corr):
    """90 is real (Arsenal's points) but it is not in offensive-1."""
    g = run([arsenal, corr], "Arsenal had 90 points; r = 0.597.", stats=[stat("Points", "90", "offensive-1")])
    assert kinds(g) == ["key_stat_value_unsupported"]


def test_key_stat_label_value_mismatch_is_detected(corr):
    g = run([corr], "r = 0.597.", stats=[stat("p-value", "0.597", "offensive-1")])
    assert kinds(g) == ["key_stat_label_mismatch"] and "as r" in g["issues"][0]["detail"]


def test_key_stat_points_label_must_match_points_field(arsenal):
    """73 is Arsenal's goals, not points."""
    g = run([arsenal], "Arsenal scored 73 goals.", stats=[stat("Points", "73", "trend-3")])
    assert kinds(g) == ["key_stat_label_mismatch"]


def test_correct_key_stats_pass(arsenal, covid):
    g = run([arsenal, covid], "Arsenal had 90 points. Serie A home wins fell; not statistically significant.",
            stats=[stat("Arsenal points, 2003-04", "90", "trend-3"), stat("Home win rate, 2020-21", "40.8%", "trend-1"),
                   stat("Change in home win rate", "-4.7 percentage points", "trend-1"), stat("Champion", "Arsenal", "trend-3")])
    assert g["passed"], g["feedback"]


def test_non_numeric_key_stat_must_appear_in_source(arsenal):
    g = run([arsenal], "Arsenal had 90 points.", stats=[stat("Champion", "Chelsea", "trend-3")])
    assert kinds(g) == ["key_stat_value_unsupported"]


# ---------------------------------------------------------------------------
# 17-20  significance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", ["is not statistically significant", "is not significant at the 5% level",
                                    "is statistically insignificant", "did not change significantly",
                                    "is non-significant"])
def test_insignificant_result_described_correctly_passes(covid, phrase):
    assert run([covid], f"The fall from 45.4% to 40.8% {phrase} (p = 0.0841).")["passed"]


@pytest.mark.parametrize("phrase", ["is statistically significant", "is significant at the 5% level",
                                    "fell significantly"])
def test_insignificant_result_called_significant_fails(covid, phrase):
    g = run([covid], f"The home win rate {phrase} (p = 0.0841).")
    assert kinds(g) == ["significance_contradicted"] and "trend-1: not significant" in g["feedback"]


def test_significant_result_described_correctly_passes(covid_all, corr, groups):
    assert run([covid_all], "Home wins fell from 45.9% to 39.9%, a statistically significant change (p < 0.001).")["passed"]
    assert run([corr], "The association (r = 0.597) is statistically significant (p < 0.001).")["passed"]
    assert run([groups], "The difference is significant at the 5% level (p = 0.0131).")["passed"]


def test_significant_result_called_insignificant_fails(covid_all):
    g = run([covid_all], "Home wins fell from 45.9% to 39.9%, but this is not statistically significant.")
    assert kinds(g) == ["significance_contradicted"]


def test_significance_without_any_test_fails(arsenal):
    g = run([arsenal], "Arsenal's 90 points were a statistically significant improvement.")
    assert kinds(g) == ["significance_without_test"]


def test_p_threshold_claim_is_checked(covid, corr):
    assert not run([covid], "Not statistically significant? Actually p < 0.05.")["passed"]
    assert run([corr], "The association is statistically significant (p < 0.05).")["passed"]


# ---------------------------------------------------------------------------
# 21-22  causal language
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Teams that take more shots tend to earn more points (r = 0.597).",
    "Shots are positively associated with points (r = 0.597); correlation does not establish causation.",
    "This does not prove that taking more shots causes more points (r = 0.597).",
    "Shooting more is linked to, but not necessarily the cause of, higher points (r = 0.597).",
])
def test_association_wording_passes(corr, text):
    assert run([corr], text)["passed"]


@pytest.mark.parametrize("text,word", [
    ("Taking more shots causes teams to earn more points (r = 0.597).", "causes"),
    ("More shots lead to more points (r = 0.597).", "lead to"),
    ("Shots directly affect league points (r = 0.597).", "directly affect"),
    ("Each extra shot will earn a team more points (r = 0.597).", "will earn"),
])
def test_causal_wording_on_correlation_fails(corr, text, word):
    g = run([corr], text)
    assert kinds(g) == ["unsupported_causal_claim"] and g["unsupported"] == [word]


def test_causal_check_only_applies_to_correlation_and_regression(arsenal):
    assert run([arsenal], "Arsenal's unbeaten run led to 90 points in 2003-04.")["passed"]


# ---------------------------------------------------------------------------
# 23-24  warnings
# ---------------------------------------------------------------------------
def test_incomplete_season_must_be_acknowledged(table_1819):
    g = run([table_1819], "Liverpool topped the 2018-19 Premier League with 54 points.")
    assert kinds(g) == ["missing_caveat"] and "210 of 380" in g["feedback"]


def test_incomplete_season_acknowledged_passes(table_1819):
    g = run([table_1819], "In the available 2018-19 data, which is incomplete (210 of 380 matches), "
                          "Liverpool lead with 54 points from 21 games.")
    assert g["passed"], g["feedback"]


def test_irrelevant_warning_does_not_reject(corr, covid):
    """Both results carry 2018-19 warnings, but these answers do not rely on that season."""
    assert run([corr], "Shots and points are associated (r = 0.597).")["passed"]
    assert run([covid], "Comparing 2010-11 to 2018-19 with 2020-21, home wins fell from 45.4% to 40.8%; "
                        "not statistically significant.")["passed"]


def test_season_discussed_on_its_own_triggers_caveat(covid):
    g = run([covid], "Home wins in 2018-19 were typical; overall the fall was not statistically significant.")
    assert kinds(g) == ["missing_caveat"]


# ---------------------------------------------------------------------------
# 25-27  failures, overlaps, empties
# ---------------------------------------------------------------------------
def test_failed_tool_is_not_evidence(arsenal):
    poisoned = {"call_id": "trend-9", "agent": "trend", "tool": "top_n", "args": {}, "ok": False,
                "result": {"Points": 999}, "error": {"code": "x", "message": "boom", "suggestions": []}}
    g = run([arsenal, poisoned], "Arsenal had 999 points; the second lookup failed.")
    assert g["unsupported"] == ["999"]


def test_failed_tool_must_be_acknowledged(arsenal):
    failed = result("trend-8", "team_season_stats", {"team": "Arsenl"})
    assert kinds(run([arsenal, failed], "Arsenal had 90 points.")) == ["missing_failure_note"]
    assert run([arsenal, failed], "Arsenal had 90 points; 'Arsenl' could not be found.")["passed"]


def test_overlapping_values_across_tools(arsenal, corr):
    """A value from either tool is accepted; typed claims still need the right type."""
    assert run([arsenal, corr], "Arsenal had 90 points and 26 wins; shots relate to points (r = 0.597).")["passed"]
    assert not run([arsenal, corr], "r = 90.")["passed"]


def test_empty_draft_fails_safely(arsenal):
    s = new_state("q")
    s.update(tool_results=[arsenal], draft_answer=None)
    g = G.check(s)
    assert not g["passed"] and kinds(g) == ["empty_draft"]
    assert not run([arsenal], "   ")["passed"]


def test_no_tool_results_with_statistics_fails():
    g = run([], "Arsenal had 90 points in 2003-04.")
    assert g["unsupported"] == ["90"]


def test_deterministic_no_evidence_and_fallback_drafts_pass(covid, table_1819):
    failed = result("trend-8", "team_season_stats", {"team": "Atletico Madird"})
    s = new_state("How did Atletico Madird do?")
    s.update(tool_results=[failed], draft_answer=syn.no_evidence_draft({"tool_results": [failed], "errors": []}))
    assert G.check(s)["passed"]
    for r in (covid, table_1819):
        s = new_state("q")
        s.update(tool_results=[r])
        s["draft_answer"] = syn.fallback_draft(s, "LLMTimeoutError")
        assert G.check(s)["passed"], G.check(s)["feedback"]


# ---------------------------------------------------------------------------
# 28-31  no LLM, no tools, schema, useful feedback
# ---------------------------------------------------------------------------
def test_guard_takes_no_llm():
    assert inspect.signature(G.make_guard).parameters == {}
    assert "llm" not in G.__dict__ and "LLMClient" not in G.__dict__


def test_guard_never_calls_tools(monkeypatch, covid, corr):
    monkeypatch.setattr(registry, "call_tool", lambda *a, **k: pytest.fail("guard called a tool"))
    for name, tool in list(registry.TOOLS.items()):
        monkeypatch.setitem(registry.TOOLS, name, tool.__class__(tool.name, lambda **k: pytest.fail("fn called"),
                                                                   tool.description, tool.args_model))
    run([covid, corr], "Not statistically significant (p = 0.0841); r = 0.597.")


def test_guard_does_not_modify_tool_results(covid):
    s = new_state("q")
    s.update(tool_results=[covid], draft_answer={"answer": "45.5%", "evidence": [], "key_stats": [], "explanation": ""})
    before = json.dumps(s["tool_results"], sort_keys=True)
    G.make_guard()(s)
    assert json.dumps(s["tool_results"], sort_keys=True) == before


def test_node_update_follows_schema(covid):
    s = new_state("q")
    s.update(tool_results=[covid], draft_answer={"answer": "fell from 45.5%", "evidence": [], "key_stats": [],
                                                 "explanation": ""})
    out = G.make_guard()(s)
    assert set(out) == {"grounding_result"}                  # nothing else: retries belong to Stage 7
    g = out["grounding_result"]
    assert set(g) == set(get_type_hints(GroundingResult))
    assert all(set(i) == set(get_type_hints(GroundingIssue)) for i in g["issues"])
    assert set(out) <= set(get_type_hints(AgentState, include_extras=True))
    json.dumps(out, allow_nan=False)


def test_guard_error_fails_closed(monkeypatch, covid):
    monkeypatch.setattr(G, "check", lambda s: 1 / 0)
    s = new_state("q")
    s.update(tool_results=[covid], draft_answer={"answer": "x", "evidence": [], "key_stats": [], "explanation": ""})
    out = G.make_guard()(s)
    assert out["grounding_result"]["passed"] is False and out["grounding_result"]["issues"][0]["kind"] == "guard_error"
    assert out["errors"][0]["node"] == "guard" and add([], out["errors"])


def test_feedback_is_specific_and_feeds_the_synthesiser(covid):
    g = run([covid], "Home wins fell from 45.5% to 40.8%, which is statistically significant (p = 0.01).",
            stats=[stat("p-value", "0.01", "trend-1")])
    assert set(kinds(g)) == {"unsupported_number", "significance_contradicted", "key_stat_value_unsupported"}
    assert "45.5%" in g["unsupported"] and "0.01" in g["unsupported"]
    for needle in ("0.454489", "0.0841462", "trend-1: not significant", "key_stats[0]"):
        assert needle in g["feedback"]
    # the synthesiser's retry prompt reads exactly these fields
    s = new_state("q")
    s.update(tool_results=[covid], grounding_result=g,
             draft_answer={"answer": "old", "evidence": [], "key_stats": [], "explanation": ""})
    user = syn.build_messages(s, syn.package_evidence(s)[0], "v1")[1].content
    assert "GROUNDING FEEDBACK" in user and "45.5%" in user and "0.0841462" in user


def test_checked_counts_claims(arsenal):
    g = run([arsenal], "26 wins, 12 draws, 90 points.", stats=[stat("Points", "90", "trend-3")])
    assert g["checked"] == 4 and g["passed"]


# ===========================================================================
# Stage 6.1 — evidence integrity: only successful RESULTS validate numbers
# ===========================================================================
@pytest.fixture(scope="module")
def top3():         # args n=3, min_played=37; results 102, 100, 100 (37 is only an argument)
    return result("trend-10", "top_n", {"metric": "Points", "n": 3, "min_played": 37})


@pytest.fixture(scope="module")
def table_0405():   # Chelsea 95, Arsenal 83
    return result("trend-11", "league_table", {"league": "Premier League", "season": "2004-05"})


@pytest.fixture(scope="module")
def arsenal_trend():  # Arsenal points by season: 90 in 2003-04, 83 in 2004-05
    return result("trend-12", "season_trend", {"team": "Arsenal", "metric": "Points"})


FAILED_WITH_NUMBERS = {   # explicit failed result whose error message and arguments contain numbers
    "call_id": "offensive-9", "agent": "offensive", "tool": "correlation_analysis",
    "args": {"x_metric": "AvgShots", "season_start": "1995-96", "season_end": "1996-97"},
    "ok": False, "result": None,
    "error": {"code": "insufficient_data",
              "message": "Only 14 team-seasons have both AvgShots and Points for this selection; at least 3 are needed.",
              "suggestions": []}}


def test_question_only_number_cannot_validate_claim(arsenal):
    g = run([arsenal], "Arsenal scored 100 goals in 2003-04.", question="Did Arsenal score 100 goals in 2003-04?")
    assert g["unsupported"] == ["100"]
    g = run([arsenal], "The figure you asked about is 17.", question="Is Arsenal's number 17?")
    assert g["unsupported"] == ["17"]


def test_argument_only_number_cannot_validate_claim(top3):
    assert 37 in top3["args"].values()
    g = run([top3], "Juventus had 37 wins.")
    assert g["unsupported"] == ["37"] and "args" not in g["feedback"]   # hints only point at results


def test_failed_tool_numbers_cannot_validate_claim(arsenal):
    g = run([arsenal, FAILED_WITH_NUMBERS], "Arsenal averaged 14 shots per game; the correlation could not be run.")
    assert g["unsupported"] == ["14"]


def test_verbatim_quote_of_failed_tool_error_is_allowed(arsenal):
    msg = FAILED_WITH_NUMBERS["error"]["message"]
    g = run([arsenal, FAILED_WITH_NUMBERS], "Arsenal had 90 points in 2003-04.",
            evidence=[f"correlation_analysis failed: {msg}"])
    assert g["passed"], g["feedback"]


def test_key_stat_cannot_use_question_number(arsenal):
    g = run([arsenal], "Arsenal had 90 points.", question="Did Arsenal finish 2003-04 with 100 points?",
            stats=[stat("Points", "100", "trend-3")])
    assert kinds(g) == ["key_stat_value_unsupported"]


def test_key_stat_cannot_use_argument_number(top3):
    g = run([top3], "Juventus top the list with 102 points.", stats=[stat("Minimum games played", "37", "trend-10")])
    assert kinds(g) == ["key_stat_value_unsupported"]


def test_typed_claim_cannot_use_argument_or_question_number(top3):
    assert run([top3], "The p-value is 3.")["unsupported"] == ["3"]
    assert run([top3], "p = 0.04", question="Is p = 0.04 here?")["unsupported"] == ["0.04"]


def test_result_values_still_validate(top3, arsenal):
    assert run([top3], "The top 3 points totals are 102, 100 and 100.")["passed"]          # 3 echoes n=3
    assert run([arsenal], "Arsenal had 90 points and 26 wins in 2003-04.")["passed"]


def test_echo_needs_same_wording_and_no_metric(arsenal):
    assert run([arsenal], "Here are Arsenal's top 5 figures: 90 points.",
               question="Show me Arsenal's top 5 figures for 2003-04")["passed"]
    g = run([arsenal], "Arsenal had 5 wins.", question="Show me Arsenal's top 5 figures for 2003-04")
    assert g["unsupported"] == ["5"]                                                       # '5 wins' is a statistic


def test_context_still_masks_seasons_teams_and_leagues():
    r = result("trend-6", "team_season_stats", {"team": "Schalke 04", "season": "2010-11"})
    pts = r["result"]["rows"][0]["Points"]
    assert run([r], f"Schalke 04 earned {pts} points in the 2010-11 Bundesliga.",
               question="How did Schalke 04 do in 2010-11?")["passed"]
    lg = result("trend-7", "league_table", {"league": "Ligue 1", "season": "2010-11"})
    top = lg["result"]["table"][0]
    assert run([lg], f"{top['Team']} won Ligue 1 in 2010-11 with {top['Points']} points.")["passed"]


# ===========================================================================
# Stage 6.1 — metric / team / season-aware prose checks (conservative)
# ===========================================================================
def test_goals_described_as_points_fails(arsenal):
    g = run([arsenal], "Arsenal finished 2003-04 with 73 points.")
    assert kinds(g) == ["metric_mismatch"]
    assert "GoalsFor" in g["feedback"] and "90" in g["feedback"]           # says what 73 is, and the real points


def test_goals_and_points_correctly_attributed_pass(arsenal):
    assert run([arsenal], "Arsenal scored 73 goals and earned 90 points in 2003-04.")["passed"]


def test_games_played_described_as_points_fails(arsenal):
    assert kinds(run([arsenal], "Arsenal earned 38 points in 2003-04.")) == ["metric_mismatch"]


def test_number_without_metric_word_is_not_metric_checked(arsenal):
    """Conservative: with no metric word, 73 is only checked against all result values."""
    assert run([arsenal], "Arsenal's 2003-04 total was 73.")["passed"]


def test_other_teams_value_fails(table_0405):
    g = run([table_0405], "Arsenal won the 2004-05 league with 95 points.")
    assert kinds(g) == ["entity_mismatch"] and "Chelsea" in g["feedback"] and "83" in g["feedback"]


def test_correct_team_value_passes(table_0405):
    assert run([table_0405], "Chelsea won the 2004-05 league with 95 points.")["passed"]
    assert run([table_0405], "Chelsea won with 95 points, ahead of Arsenal on 83 points.")["passed"]
    assert run([table_0405], "Chelsea (95 points) finished ahead of Arsenal (83 points) in 2004-05.")["passed"]


def test_bare_number_is_not_attributed_to_a_team(table_0405):
    """57 is Chelsea's goal difference; with no metric word after it, it is not pinned on Arsenal."""
    assert run([table_0405], "Arsenal finished behind the champions, whose goal difference was 57.")["passed"]


def test_two_teams_in_one_clause_are_not_attributed(table_0405):
    """Documented limitation: ambiguous attribution is not judged (no false positive, no detection)."""
    assert run([table_0405], "Arsenal trailed Chelsea, who won with 95 points.")["passed"]


def test_other_seasons_value_fails(arsenal_trend):
    g = run([arsenal_trend], "Arsenal had 90 points in 2004-05.")
    assert kinds(g) == ["season_mismatch"] and "2003-04" in g["feedback"]


def test_correct_season_and_aggregates_pass(arsenal_trend):
    mean = arsenal_trend["result"]["mean"]
    assert run([arsenal_trend], "Arsenal had 90 points in 2003-04, their best season.")["passed"]
    assert run([arsenal_trend], f"Since 1993-94 Arsenal have averaged {mean:.1f} points per season.")["passed"]


def test_aggregate_wording_disables_season_attribution(arsenal_trend):
    """'Since 1993-94 ... best season ... 90 points' is about the whole range, not about 1993-94."""
    assert run([arsenal_trend], "Since 1993-94, Arsenal's best season produced 90 points.")["passed"]


def test_alias_team_name_keeps_metric_check():
    """'Manchester United' is not the data's spelling, so the team check is off, but the metric check still works."""
    r = result("trend-13", "team_season_stats", {"team": "Manchester United", "season": "1999-00"})
    row = r["result"]["rows"][0]
    assert run([r], f"Manchester United earned {row['Points']} points in 1999-00.")["passed"]
    assert kinds(run([r], f"Manchester United scored {row['Points']} goals in 1999-00.")) == ["metric_mismatch"]


def test_metric_check_does_not_touch_statistics(corr, covid):
    assert run([corr], "Teams that take more shots earn more points (r = 0.597, n = 2,224).")["passed"]
    assert run([covid], "Over 3,230 matches the home win rate was 45.4%; not statistically significant.")["passed"]