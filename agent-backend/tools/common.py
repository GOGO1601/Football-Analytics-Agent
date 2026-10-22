"""
Shared plumbing for the analytics tools: cached data access, input resolution
(league / team / season), structured errors, data-quality warnings and JSON-safe output.

Tools never return prose answers; they return dicts made of plain Python types so the
agent layer (and the grounding guard) can read every number they produce.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from data.aggregate import expand_to_team_rows, team_seasons
from data.loader import (LEAGUE_FOLDERS, _fingerprint, default_data_dir, discover, load_matches,
                         season_label, season_start_year, valid_matches)

LEAGUES = list(LEAGUE_FOLDERS)

# Team-season columns a tool may be asked about.
NUMERIC_METRICS = ["Played", "Wins", "Draws", "Losses", "Points", "GoalsFor", "GoalsAgainst",
                   "GoalDiff", "AvgShots", "AvgShotsOnTarget", "AvgFouls", "AvgYellowCards",
                   "AvgRedCards", "PointsPerGame", "GoalsForPerGame", "GoalsAgainstPerGame"]

LEAGUE_ALIASES = {
    "premier league": "Premier League", "epl": "Premier League", "pl": "Premier League",
    "english premier league": "Premier League",
    "la liga": "La Liga", "laliga": "La Liga", "primera division": "La Liga",
    "bundesliga": "Bundesliga",
    "serie a": "Serie A", "seriea": "Serie A",
    "ligue 1": "Ligue 1", "ligue1": "Ligue 1",
}

# Common full names -> the spelling used in the football-data.co.uk CSVs.
TEAM_ALIASES = {
    "manchester united": "Man United", "man utd": "Man United", "manchester city": "Man City",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham", "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves", "nottingham forest": "Nott'm Forest",
    "sheffield wednesday": "Sheffield Weds", "west bromwich albion": "West Brom",
    "queens park rangers": "QPR", "newcastle united": "Newcastle", "leicester city": "Leicester",
    "atletico madrid": "Ath Madrid", "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao",
    "real sociedad": "Sociedad", "real betis": "Betis", "espanyol": "Espanol",
    "deportivo la coruna": "La Coruna", "deportivo": "La Coruna", "rayo vallecano": "Vallecano",
    "celta vigo": "Celta", "sporting gijon": "Sp Gijon",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt", "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach", "gladbach": "M'gladbach", "schalke": "Schalke 04",
    "fc cologne": "FC Koln", "cologne": "FC Koln", "koln": "FC Koln", "hertha berlin": "Hertha",
    "vfb stuttgart": "Stuttgart", "vfl wolfsburg": "Wolfsburg", "bayern": "Bayern Munich",
    "ac milan": "Milan", "inter milan": "Inter", "internazionale": "Inter", "as roma": "Roma",
    "ss lazio": "Lazio", "ssc napoli": "Napoli", "hellas verona": "Verona",
    "paris saint-germain": "Paris SG", "paris saint germain": "Paris SG", "psg": "Paris SG",
    "olympique lyonnais": "Lyon", "olympique de marseille": "Marseille", "as monaco": "Monaco",
    "saint-etienne": "St Etienne", "saint etienne": "St Etienne", "losc": "Lille",
}

# The same club recorded under two spellings in the source CSVs (the dashboard treats them
# as separate teams; tools keep that behaviour but warn).
SPELLING_VARIANTS = {
    "M'gladbach": ["M'Gladbach"], "M'Gladbach": ["M'gladbach"],
    "Villarreal": ["Villareal"], "Villareal": ["Villarreal"],
    "Fortuna Dusseldorf": ["Dusseldorf"], "Dusseldorf": ["Fortuna Dusseldorf"],
}


class ToolError(ValueError):
    """Structured, user-facing tool failure (unknown team, insufficient data, ...)."""

    def __init__(self, code: str, message: str, suggestions: list[str] | None = None, **details):
        super().__init__(message)
        self.code, self.message = code, message
        self.suggestions = suggestions or []
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "suggestions": self.suggestions, **jsonable(self.details)}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    matches: pd.DataFrame          # valid match rows (non-empty HomeTeam, AwayTeam, FTR)
    ts: pd.DataFrame               # team-season aggregate (dashboard-identical) + per-game columns
    coverage: pd.DataFrame         # League, Season, matches, teams, expected, complete
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_matches(cls, matches: pd.DataFrame) -> "Dataset":
        m = valid_matches(matches).copy()
        ts = team_seasons(matches) if len(m) else team_seasons(matches.iloc[0:0])
        if len(ts):
            played = ts["Played"].replace(0, np.nan)
            ts["PointsPerGame"] = ts["Points"] / played
            ts["GoalsForPerGame"] = ts["GoalsFor"] / played
            ts["GoalsAgainstPerGame"] = ts["GoalsAgainst"] / played
        else:
            for c in ("PointsPerGame", "GoalsForPerGame", "GoalsAgainstPerGame", "SeasonStart"):
                ts[c] = pd.Series(dtype=float)
        cov = []
        for (lg, s), g in m.groupby(["League", "Season"], sort=False):
            teams = len(set(g["HomeTeam"]) | set(g["AwayTeam"]))
            exp = teams * (teams - 1)
            cov.append({"League": lg, "Season": s, "SeasonStart": season_start_year(s),
                        "matches": len(g), "teams": teams, "expected": exp, "complete": len(g) == exp})
        coverage = pd.DataFrame(cov, columns=["League", "Season", "SeasonStart", "matches", "teams",
                                              "expected", "complete"])
        if len(m):
            m["SeasonStart"] = m["Season"].map(season_start_year)
        return cls(matches=m, ts=ts, coverage=coverage)


@lru_cache(maxsize=2)
def _cached(data_dir: str, fp: tuple) -> Dataset:
    return Dataset.from_matches(load_matches(data_dir))


def get_dataset() -> Dataset:
    """Dataset over /datasets, cached until a CSV changes."""
    root = default_data_dir()
    return _cached(str(root.resolve()), _fingerprint(discover(root)))


def _ds(data: Dataset | None) -> Dataset:
    return data if data is not None else get_dataset()


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.strip().lower())


def resolve_league(name: str | None, data: Dataset) -> str | None:
    if name is None or str(name).strip().lower() in ("", "all", "all leagues"):
        return None
    key = _norm(str(name))
    known = list(dict.fromkeys(LEAGUES + list(data.ts["League"].unique())))  # any discovered league folder
    league = LEAGUE_ALIASES.get(key) or next((l for l in known if _norm(l) == key), None)
    if league is None:
        raise ToolError("unknown_league", f"Unknown league '{name}'.",
                        suggestions=[l for l in known if (data.ts["League"] == l).any()])
    if not (data.ts["League"] == league).any():
        raise ToolError("no_data", f"No data loaded for league '{league}'.")
    return league


def resolve_team(name: str, data: Dataset, league: str | None = None) -> str:
    if not name or not str(name).strip():
        raise ToolError("unknown_team", "Team name is empty.")
    pool = data.ts if league is None else data.ts[data.ts["League"] == league]
    teams = sorted(set(pool["Team"]))
    by_norm = {_norm(t): t for t in teams}
    key = _norm(str(name))
    if key in by_norm:
        return by_norm[key]
    alias = TEAM_ALIASES.get(key)
    if alias and alias in teams:
        return alias
    close = difflib.get_close_matches(key, list(by_norm), n=5, cutoff=0.6)
    close += [k for k in by_norm if key in k and k not in close]
    where = f" in {league}" if league else ""
    raise ToolError("unknown_team", f"Unknown team '{name}'{where}.",
                    suggestions=[by_norm[k] for k in close][:5])


_SEASON_RE = re.compile(r"^\s*(\d{4})\s*[-/]\s*(\d{2}|\d{4})\s*$")


def normalise_season(value: str | int) -> str:
    """Accepts '2004-05', '2004/05', '2004/2005', 2004 (start year) -> '2004-05'."""
    s = str(value).strip()
    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return f"{y}-{str(y + 1)[-2:]}"
    m = _SEASON_RE.match(s)
    if not m:
        raise ToolError("invalid_season", f"'{value}' is not a season. Use the form '2004-05'.")
    a, b = int(m.group(1)), m.group(2)
    if int(b[-2:]) != (a + 1) % 100:
        raise ToolError("invalid_season", f"'{value}' is not a valid season (years must be consecutive).")
    return f"{a}-{b[-2:]}"


def resolve_season(value: str | int, data: Dataset, league: str | None = None) -> str:
    s = normalise_season(value)
    pool = data.ts if league is None else data.ts[data.ts["League"] == league]
    seasons = set(pool["Season"])
    if s not in seasons:
        rng = sorted(seasons, key=season_start_year)
        span = f"{rng[0]} to {rng[-1]}" if rng else "none"
        raise ToolError("unknown_season", f"No data for season {s}" + (f" in {league}" if league else "")
                        + f". Available: {span}.", suggestions=rng[-5:] if rng else [])
    return s


def season_bounds(start, end, data: Dataset, league: str | None = None) -> tuple[int, int]:
    lo = season_start_year(resolve_season(start, data, league)) if start is not None else -10**9
    hi = season_start_year(resolve_season(end, data, league)) if end is not None else 10**9
    if lo > hi:
        raise ToolError("invalid_season", f"season_start {start} is after season_end {end}.")
    return lo, hi


def check_metric(metric: str) -> str:
    if metric not in NUMERIC_METRICS:
        close = difflib.get_close_matches(metric, NUMERIC_METRICS, n=3, cutoff=0.4)
        raise ToolError("unknown_metric", f"Unknown metric '{metric}'.", suggestions=close or NUMERIC_METRICS)
    return metric


# ---------------------------------------------------------------------------
# Data-quality notes
# ---------------------------------------------------------------------------
def incomplete_seasons(data: Dataset, rows: pd.DataFrame) -> list[dict]:
    """Which (league, season) pairs among `rows` are not complete double round-robins in the CSVs."""
    if rows.empty or data.coverage.empty:
        return []
    pairs = rows[["League", "Season"]].drop_duplicates()
    cov = data.coverage.merge(pairs, on=["League", "Season"])
    bad = cov[~cov["complete"]]
    return [{"league": r.League, "season": r.Season, "matches": int(r.matches), "expected": int(r.expected)}
            for r in bad.itertuples()]


def quality_warnings(data: Dataset, rows: pd.DataFrame, teams: list[str] | None = None,
                     metrics: list[str] | None = None) -> list[str]:
    w = []
    for b in incomplete_seasons(data, rows):
        w.append(f"{b['league']} {b['season']} is incomplete in the source data "
                 f"({b['matches']} of {b['expected']} matches); season totals are partial.")
    for t in teams or []:
        for v in SPELLING_VARIANTS.get(t, []):
            if (data.ts["Team"] == v).any():
                seasons = ", ".join(data.ts.loc[data.ts["Team"] == v, "Season"])
                w.append(f"'{t}' also appears as '{v}' ({seasons}); those seasons are not included.")
    for m in metrics or []:
        if m in rows and rows[m].isna().any():
            w.append(f"{m} is missing for {int(rows[m].isna().sum())} of {len(rows)} team-seasons "
                     "(not recorded in older seasons); those rows are excluded.")
    return w


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def jsonable(obj):
    """Convert numpy/pandas scalars, NaN and tuples into plain JSON-compatible Python."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


INT_LIKE = {"Played", "Wins", "Draws", "Losses", "Points", "GoalsFor", "GoalsAgainst", "GoalDiff",
            "SeasonStart", "position"}


def records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for r in df[cols].to_dict("records"):
        row = {}
        for k, v in r.items():
            v = jsonable(v)
            if k in INT_LIKE and isinstance(v, float):
                v = int(v)
            row[k] = v
        out.append(row)
    return out


def effect_label(value: float, kind: str) -> str:
    a = abs(value)
    if kind == "r":
        return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


__all__ = ["ToolError", "Dataset", "get_dataset", "LEAGUES", "NUMERIC_METRICS", "season_label",
           "expand_to_team_rows"]
