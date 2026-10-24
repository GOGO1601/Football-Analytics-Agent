"""Shared fixtures: the real dataset (cached) and a builder for small synthetic datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import GOAL_FIELDS, NULLABLE_FIELDS
from tools.common import Dataset, get_dataset


@pytest.fixture(scope="session")
def real() -> Dataset:
    return get_dataset()


def make_matches(rows: list[tuple], league: str = "Test League", season: str = "2020-21",
                 stats: dict | None = None) -> pd.DataFrame:
    """
    rows: (home, away, home_goals, away_goals[, FTR]) — FTR derived from goals if omitted.
    stats: optional {column: [values per row]} for HS/AS/HF/... (None = missing).
    Produces a frame in the same shape as data.loader.load_matches().
    """
    recs = []
    for i, r in enumerate(rows):
        h, a, hg, ag = r[:4]
        ftr = r[4] if len(r) > 4 else ("H" if hg > ag else "A" if ag > hg else "D")
        rec = {"League": league, "Season": season, "Date": f"{i + 1:02d}/01/21", "HomeTeam": h,
               "AwayTeam": a, "FTR": ftr, "Referee": None, "FTHG": float(hg), "FTAG": float(ag)}
        for c in NULLABLE_FIELDS:
            v = (stats or {}).get(c, [None] * len(rows))[i]
            rec[c] = np.nan if v is None else float(v)
        rec["SourceFile"] = "synthetic"
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["SeasonStart"] = df["Season"].str[:4].astype(int)
    for c in GOAL_FIELDS + NULLABLE_FIELDS:
        df[c] = df[c].astype(float)
    return df


def make_dataset(*frames: pd.DataFrame) -> Dataset:
    return Dataset.from_matches(pd.concat(frames, ignore_index=True))


@pytest.fixture
def mini() -> Dataset:
    """
    Three-team double round robin in one season. Worked out by hand:
      A: W W W D  P4 3-1-0  10 pts  GF 7 GA 2 GD +5   AvgShots (10+12+9+8)/4 = 9.75
      B: L L D W  P4 1-1-2   4 pts  GF 3 GA 4 GD -1   AvgShots (3+5+11)/3   (one match missing)
      C: L D D L  P4 0-2-2   2 pts  GF 3 GA 7 GD -4   AvgShots (6+7+4)/3    (one match missing)
    Home results: 2 H, 2 D, 2 A.
    """
    rows = [
        ("A", "B", 2, 0), ("A", "C", 3, 1), ("B", "A", 0, 1),
        ("B", "C", 1, 1), ("C", "A", 1, 1), ("C", "B", 0, 2),
    ]
    stats = {
        "HS": [10, 12, 5, None, 7, 4], "AS": [3, 6, 9, None, 8, 11],
        "HST": [5, 6, 2, None, 3, 1], "AST": [1, 2, 4, None, 3, 5],
        "HF": [10, 11, 14, 12, 9, 13], "AF": [12, 9, 10, 8, 11, 7],
        "HY": [1, 2, 3, 2, 1, 0], "AY": [2, 1, 1, 3, 2, 1],
        "HR": [0, 0, 1, 0, 0, 0], "AR": [0, 1, 0, 0, 0, 0],
    }
    return make_dataset(make_matches(rows, stats=stats))
