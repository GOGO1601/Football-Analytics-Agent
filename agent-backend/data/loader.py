"""
Load the football match CSVs from the shared /datasets folder.

Mirrors how the dashboard reads the data (frontend/src/data-loader.js `loadSeasonCSV`
and frontend/src/main.js `loadSeason`):

  * one CSV per league per season: datasets/<league-folder>/season-XXYY.csv
  * every cell is read as a string (d3.csv does the same)
  * FTHG / FTAG are coerced with JS unary plus:   ""  -> 0,  "abc" -> NaN
  * HS AS HST AST HF AF HY AY HR AR are "nullable":  "" or absent -> null,
    anything else -> Number(value)
  * a file that fails to load contributes no rows (the JS `.catch(() => [])`)

In pandas both JS `null` and JS `NaN` become NaN. That is safe for every calculation
the dashboard does: d3.sum and d3.mean skip both, and nullMean filters nulls before
d3.mean. The one difference (all-NaN column -> d3.mean returns `undefined` instead of
`null`) cannot happen in these datasets and both are treated as "missing" downstream.

The source of truth stays in /datasets; nothing is copied. Loaded data is cached in
memory and invalidated automatically when any CSV's size or mtime changes.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Same league names and folders as LEAGUE_CONFIGS / leagueConfigs in the JS.
LEAGUE_FOLDERS: dict[str, str] = {
    "Premier League": "premier-league",
    "La Liga": "la-liga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
}
FOLDER_TO_LEAGUE = {v: k for k, v in LEAGUE_FOLDERS.items()}

# Columns the JS treats as nullable numbers, in the JS order.
NULLABLE_FIELDS = ["HS", "AS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR"]
GOAL_FIELDS = ["FTHG", "FTAG"]

_SEASON_FILE = re.compile(r"^season-(\d{2})(\d{2})\.csv$")
# What JS Number() accepts for a trimmed, non-empty decimal string.
_JS_DECIMAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class DataError(ValueError):
    """Raised when the dataset folder is missing or unusable."""


# ---------------------------------------------------------------------------
# JS number coercion
# ---------------------------------------------------------------------------
def js_number(value: str | None) -> float:
    """Python equivalent of JavaScript `+value` / `Number(value)` for CSV strings."""
    if value is None:
        return np.nan  # +undefined -> NaN
    s = value.strip()
    if s == "":
        return 0.0  # +"" and +"   " -> 0
    if _JS_DECIMAL.match(s):
        return float(s)
    if re.fullmatch(r"[+-]?Infinity", s):
        return float("-inf") if s.startswith("-") else float("inf")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return float(int(s, 16))
    if re.fullmatch(r"0[bB][01]+", s):
        return float(int(s, 2))
    if re.fullmatch(r"0[oO][0-7]+", s):
        return float(int(s, 8))
    return np.nan


def js_nullable_number(value: str | None) -> float:
    """`(v === "" || v == null) ? null : +v` — null is represented as NaN."""
    if value is None or value == "":
        return np.nan
    return js_number(value)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def default_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[1] / p).resolve()
        return p
    return Path(__file__).resolve().parents[2] / "datasets"


def season_label(code: str) -> str:
    """'9394' -> '1993-94', '0001' -> '2000-01', '2526' -> '2025-26' (same labels as ALL_SEASONS)."""
    m = re.fullmatch(r"(\d{2})(\d{2})", code)
    if not m:
        raise DataError(f"Not a season code: {code!r}")
    a, b = m.groups()
    century = "19" if int(a) >= 90 else "20"
    return f"{century}{a}-{b}"


def season_start_year(label: str) -> int:
    """JS getSeasonSortKey / getSeasonSortValue: parseInt(label.split('-')[0])."""
    return int(label.split("-")[0])


@dataclass(frozen=True)
class SeasonFile:
    league: str
    season: str
    path: Path


def discover(data_dir: Path | None = None) -> list[SeasonFile]:
    """Find every datasets/<league>/season-XXYY.csv, ordered like the JS (league config order, then season)."""
    root = Path(data_dir) if data_dir else default_data_dir()
    if not root.is_dir():
        raise DataError(f"Dataset folder not found: {root}")
    found: list[SeasonFile] = []
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    order = {f: i for i, f in enumerate(LEAGUE_FOLDERS.values())}
    folders.sort(key=lambda p: (order.get(p.name, len(order)), p.name))
    for folder in folders:
        league = FOLDER_TO_LEAGUE.get(folder.name, folder.name.replace("-", " ").title())
        for f in folder.iterdir():
            m = _SEASON_FILE.match(f.name)
            if m:
                found.append(SeasonFile(league, season_label(m.group(1) + m.group(2)), f))
    found.sort(key=lambda s: (order.get(LEAGUE_FOLDERS.get(s.league, ""), len(order)), s.league,
                              season_start_year(s.season)))
    return found


def _fingerprint(files: list[SeasonFile]) -> tuple:
    out = []
    for f in files:
        st = f.path.stat()
        out.append((str(f.path), st.st_size, st.st_mtime_ns))
    return tuple(out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_one(sf: SeasonFile) -> pd.DataFrame:
    try:
        raw = pd.read_csv(sf.path, dtype=str, keep_default_na=False, na_filter=False,
                          encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()  # JS: .catch(() => [])
    df = pd.DataFrame({"League": sf.league, "Season": sf.season}, index=raw.index)
    for col in ("Date", "HomeTeam", "AwayTeam", "FTR", "Referee"):
        df[col] = raw[col].astype(object) if col in raw.columns else None
    for col in GOAL_FIELDS:
        df[col] = [js_number(v) for v in raw[col]] if col in raw.columns else np.nan
    for col in NULLABLE_FIELDS:
        df[col] = [js_nullable_number(v) for v in raw[col]] if col in raw.columns else np.nan
    df["SourceFile"] = str(sf.path.relative_to(sf.path.parents[1]))
    return df


@lru_cache(maxsize=4)
def _load_cached(data_dir: str, fingerprint: tuple) -> pd.DataFrame:
    files = discover(Path(data_dir))
    frames = [_read_one(f) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_matches()
    df = pd.concat(frames, ignore_index=True)
    df["SeasonStart"] = df["Season"].map(season_start_year).astype(int)
    for col in GOAL_FIELDS + NULLABLE_FIELDS:
        df[col] = df[col].astype(float)
    return df


def _empty_matches() -> pd.DataFrame:
    cols = (["League", "Season", "Date", "HomeTeam", "AwayTeam", "FTR", "Referee"]
            + GOAL_FIELDS + NULLABLE_FIELDS + ["SourceFile", "SeasonStart"])
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def load_matches(data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    All match rows as loaded by the JS, BEFORE the HomeTeam/AwayTeam/FTR filter.
    Columns: League, Season, SeasonStart, Date, HomeTeam, AwayTeam, FTR, Referee,
             FTHG, FTAG, HS, AS, HST, AST, HF, AF, HY, AY, HR, AR, SourceFile.
    Returns a copy, so callers can modify it freely.
    """
    root = Path(data_dir) if data_dir else default_data_dir()
    files = discover(root)
    return _load_cached(str(root.resolve()), _fingerprint(files)).copy()


def valid_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """JS: `if (!d.HomeTeam || !d.AwayTeam || !d.FTR) return;` — drop rows with an empty team or result."""
    if matches.empty:
        return matches
    ok = (matches["HomeTeam"].fillna("").astype(str) != "") & \
         (matches["AwayTeam"].fillna("").astype(str) != "") & \
         (matches["FTR"].fillna("").astype(str) != "")
    return matches[ok]


def inventory(data_dir: Path | str | None = None) -> dict:
    """What is on disk: leagues, seasons per league, rows per file, and incomplete seasons."""
    m = valid_matches(load_matches(data_dir))
    out: dict = {"leagues": {}, "total_matches": int(len(m))}
    for (league, season), g in m.groupby(["League", "Season"], sort=False):
        teams = set(g["HomeTeam"]) | set(g["AwayTeam"])
        expected = len(teams) * (len(teams) - 1)
        out["leagues"].setdefault(league, []).append({
            "season": season, "matches": int(len(g)), "teams": len(teams),
            "expected_matches": expected, "complete": len(g) == expected,
        })
    return out


def clear_cache() -> None:
    _load_cached.cache_clear()
