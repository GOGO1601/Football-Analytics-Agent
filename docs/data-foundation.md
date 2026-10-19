# Data foundation

The Python analytics engine in `agent-backend/data` and `agent-backend/tools` reproduces the dashboard's numbers exactly, and it's what the agents will call later. They never do maths themselves.

## How parity is checked

`tests/fixtures/generate_js_fixture.py` opens the real dashboard pages in headless Chromium and reads values straight out of the running JavaScript: the full output of `loadAllLeagues()` (data-loader.js), `aggregatedDataByLeague` (main.js), the Q1 team lists and default selections, and the filtered rows, summary cards and regression for 72 Q2 and 12 Q3 filter combinations. Nothing in `js_aggregates.json` is written by hand.

`tests/test_parity.py` compares Python against that file. Integers, row sets, row order and display strings must match exactly. Team-season averages also match bit for bit. Regression values allow 1e-12 relative error, though they currently match exactly too. If any of the four JS files changes, a test fails and tells you to regenerate the fixture:

```
cd agent-backend && python tests/fixtures/generate_js_fixture.py
```

## What the JS actually does

- Every league/season CSV is read with `d3.csv` (all strings). A file that fails to load is silently skipped.
- `FTHG`/`FTAG` use unary plus, so an empty cell becomes 0, not missing.
- Shots, fouls and cards become `null` when empty. Averages skip nulls (`nullMean`), and a team-season with no values at all gets `null`.
- Rows with an empty HomeTeam, AwayTeam or FTR are dropped.
- Each match becomes two team rows. A team wins only on its own result letter, so an unexpected FTR value would count as a loss for both teams. That doesn't happen in this data.
- Points are `wins * 3 + draws`. Goals are summed with `d3.sum`.
- data-loader.js and main.js contain two identical copies of this pipeline. Both are tested.
- Q2 keeps rows where x and Points exist, filtered by league and season start year. Its "missing" card counts across all data and ignores the filters.
- Q3 keeps only rows with fouls, yellows and reds all present. It has no season filter.
- No values are rounded until display, which uses `toFixed`. That rounds halves up, unlike Python's `format()`, so Python reproduces it with `js_to_fixed`.

## Things worth knowing about the data

- 2018-19 is incomplete in every league: about half the matches (e.g. 210 of 380 in the Premier League), so that season's totals are partial. 2025-26 is in progress. Ligue 1 2019-20 was cut short by COVID. The tools add a warning whenever a result touches one of these seasons.
- Shots arrive around 2000-01 (2005-06 in Ligue 1). Fouls and cards arrive a bit later, so Q3 has fewer rows.
- Some clubs appear under two spellings and get split histories: M'Gladbach/M'gladbach, Villareal/Villarreal, Dusseldorf/Fortuna Dusseldorf. Leipzig (1993-94) and RB Leipzig are genuinely different clubs, as are AC Ajaccio and GFC Ajaccio. The tools keep the data as it is and warn.
- Q1's default team selection uses names that aren't in the data ("Manchester United", "Atletico Madrid", "Borussia Dortmund", "AC Milan"). Those leagues open with two teams instead of three. This is left unchanged in the JS for now.

## Tools

| Tool | Answers |
|---|---|
| `team_season_stats` | a team's seasons, totals and per-game averages |
| `league_table` | standings for a league and season |
| `top_n` | best or worst team-seasons by any metric |
| `season_trend` | how a metric moved over time, with a linear trend |
| `team_comparison`, `head_to_head` | teams side by side, plus results between them |
| `most_improved` | biggest risers and fallers between two seasons |
| `home_advantage` | home vs away rates, league-wide or for one team |
| `data_coverage` | what's in the data and what's missing |
| `correlation_analysis`, `regression_analysis` | the Q2/Q3 questions with p-values and CIs, plus the dashboard's own R² |
| `home_advantage_test` | did home advantage change between two periods (default: pre-COVID vs 2020-21) |
| `hypothesis_test` | is a metric different between two groups of team-seasons |

Every tool returns plain JSON. Bad input comes back as a structured error with suggestions (`"Atletico Madird"` gets `["Ath Madrid"]`) instead of an exception, via `tools.registry.call_tool`.
