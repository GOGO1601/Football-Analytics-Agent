"""
Generate tests/fixtures/js_aggregates.json by running the REAL dashboard JavaScript
in headless Chromium. Nothing here re-implements the JS: values are read straight
out of the live pages, so the fixture is the ground truth the Python port must match.

What is captured
  data_loader     full output of loadAllLeagues()        (frontend/src/data-loader.js, used by Q2 + Q3)
  q1              aggregatedDataByLeague, all leagues    (frontend/src/main.js, its own copy of the pipeline)
  q1_defaults     team list + default selected teams per league
  q2_cases        getFilteredData() + linearRegression() + summary cards for every
                  x-metric x league x season-range combination
  q3_cases        same for Q3 (x-metric x league)

Run (from agent-backend/):
    python tests/fixtures/generate_js_fixture.py
Needs: pip install playwright && playwright install chromium (or a system Chromium).
d3 is loaded from the CDN as in the real pages; set D3_LOCAL_PATH=/path/to/d3.min.js
to serve a local copy instead (offline / sandboxed environments).
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import socketserver
import threading
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[3]
FRONTEND = REPO / "frontend"
OUT = Path(__file__).resolve().parent / "js_aggregates.json"
D3_LOCAL = os.environ.get("D3_LOCAL_PATH")

# JSON.stringify drops undefined and turns NaN into null; keep them distinguishable.
SERIALISE = """(obj) => JSON.stringify(obj, (k, v) =>
    v === undefined ? "__undefined__" :
    (typeof v === "number" && Number.isNaN(v)) ? "__NaN__" : v)"""

LEAGUES = ["all", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
Q2_METRICS = ["GoalsFor", "AvgShotsOnTarget", "AvgShots"]
Q2_RANGES = [None, ("2000-01", "2009-10"), ("2010-11", "2025-26"), ("2020-21", "2020-21")]
Q3_METRICS = ["AvgFouls", "AvgYellowCards"]


def _serve(root: Path) -> tuple[socketserver.TCPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    handler = functools.partial(Quiet, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _hash_keys(case: dict) -> dict:
    """Replace the sorted 'League|Team|Season' key list with a sha256 so the fixture stays small."""
    keys = case.pop("keys")
    case["keys_sha256"] = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return case


def _page(browser, url: str):
    page = browser.new_page()
    if D3_LOCAL:
        body = Path(D3_LOCAL).read_text()
        page.route("**/d3.v7.min.js", lambda r: r.fulfill(body=body, content_type="application/javascript"))
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    return page, errors


def _grab(page, expr: str):
    return json.loads(page.evaluate(f"({SERIALISE})({expr})"))


def main() -> None:
    httpd, port = _serve(REPO)
    base = f"http://127.0.0.1:{port}/frontend"
    out: dict = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        # ---------------- Q2 page: data-loader.js + Q2 filters/regression ----------------
        page, errs = _page(browser, f"{base}/q2-offensive-vs-success.html")
        page.wait_for_function("allTeamSeasons.length > 0", timeout=120_000)
        d3_version = page.evaluate("d3.version")
        out["data_loader"] = _grab(page, "allTeamSeasons")

        cases = []
        for metric in Q2_METRICS:
            for league in LEAGUES:
                for rng in Q2_RANGES:
                    case = page.evaluate(
                        """([metric, league, rng]) => {
                            xAxisSelect.value = metric; leagueFilter.value = league;
                            const seasons = [...seasonStartSel.options].map(o => o.value);
                            seasonStartSel.value = rng ? rng[0] : seasons[0];
                            seasonEndSel.value   = rng ? rng[1] : seasons[seasons.length - 1];
                            drawScatterplot();
                            const f = getFilteredData();
                            const reg = linearRegression(f, metric, "Points");
                            return JSON.stringify({
                                x_metric: metric, league,
                                season_start: seasonStartSel.value, season_end: seasonEndSel.value,
                                n: f.length,
                                keys: f.map(d => d.League + "|" + d.Team + "|" + d.Season).sort(),
                                regression: reg,
                                card_count: summaryCount.textContent,
                                card_seasons: summarySeasons.textContent,
                                card_missing: summaryMissing.textContent,
                                r2_label: reg ? reg.r2.toFixed(3) : null,
                            });
                        }""",
                        [metric, league, list(rng) if rng else None],
                    )
                    cases.append(_hash_keys(json.loads(case)))
        out["q2_cases"] = cases
        out["q2_errors"] = errs
        page.close()

        # ---------------- Q3 page ----------------
        page, errs = _page(browser, f"{base}/q3-disciplinary.html")
        page.wait_for_function("allTeamSeasons.length > 0", timeout=120_000)
        cases = []
        for metric in Q3_METRICS:
            for league in LEAGUES:
                case = page.evaluate(
                    """([metric, league]) => {
                        xMetricSelect.value = metric; leagueFilter.value = league;
                        drawBubbleChart();
                        const f = getFilteredData();
                        const reg = linearRegression(f, metric, "Points");
                        return JSON.stringify({
                            x_metric: metric, league, n: f.length,
                            keys: f.map(d => d.League + "|" + d.Team + "|" + d.Season).sort(),
                            mean_fouls: f.length ? d3.mean(f, d => d.AvgFouls) : null,
                            mean_yellow: f.length ? d3.mean(f, d => d.AvgYellowCards) : null,
                            regression: reg,
                            card_count: summaryCount.textContent,
                            card_seasons: summarySeasons.textContent,
                            card_fouls: summaryFouls.textContent,
                            card_yellow: summaryYellow.textContent,
                            r2_label: reg ? reg.r2.toFixed(3) : null,
                        });
                    }""",
                    [metric, league],
                )
                cases.append(_hash_keys(json.loads(case)))
        out["q3_cases"] = cases
        out["q3_errors"] = errs
        page.close()

        # ---------------- Q1 page: main.js (separate copy of the pipeline) ----------------
        page, errs = _page(browser, f"{base}/index.html")
        page.wait_for_function(
            "Object.keys(aggregatedDataByLeague).length === 5 && document.querySelector('#teamSelect option')",
            timeout=120_000,
        )
        out["q1"] = _grab(page, "Object.values(aggregatedDataByLeague).flat()")
        defaults = {}
        for league in LEAGUES[1:]:
            defaults[league] = json.loads(page.evaluate(
                """(league) => { leagueSelect.property("value", league);
                    populateTeamMultiSelect(league);
                    return JSON.stringify({
                        teams: [...document.querySelectorAll('#teamSelect option')].map(o => o.value),
                        selected: getSelectedTeams() }); }""",
                league,
            ))
        out["q1_defaults"] = defaults
        out["q1_errors"] = errs
        browser.close()
    httpd.shutdown()

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "d3_version": d3_version,
            "source_sha256": {p: _sha(FRONTEND / "src" / p) for p in
                              ["data-loader.js", "main.js", "q2-chart.js", "q3-chart.js"]},
            "note": "Generated by generate_js_fixture.py from the live JS. Do not edit by hand.",
        },
        **out,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB): "
          f"{len(out['data_loader'])} data-loader rows, {len(out['q1'])} Q1 rows, "
          f"{len(out['q2_cases'])} Q2 cases, {len(out['q3_cases'])} Q3 cases")


if __name__ == "__main__":
    main()
