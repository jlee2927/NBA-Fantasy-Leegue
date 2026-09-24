"""Compare the three NBA data sources before switching projection inputs.

Sources:
  hoopR    ESPN feed, 2002-2026, keyed on athlete_id
  Kaggle   NBA feed, 2002-2026, keyed on personId
  BM       Basketball Monster exports, 2022-2026, top 234 players, names only

hoopR and Kaggle are matched on identity through the name+birthdate crosswalk,
so that pair can be compared across the whole population.

Basketball Monster carries neither an id nor a birthdate, so it can only be
matched on name. It does carry age, on the same Feb-1 convention, so every
name match is verified against it: a match whose ages differ by more than
AGE_TOLERANCE is rejected rather than compared, that being the signature of
two players sharing a name.

Why both comparisons are needed: the BM overlap is the higher-confidence
sample but also a best case, since its 234 players a season are established
regulars and a player only appears if they ranked that highly, which
conditions the sample on having stayed healthy. The full-population
hoopR/Kaggle comparison covers the fringe players the BM sample cannot see.

With three sources, disagreements become attributable: when two agree and one
differs, the odd source out is the one carrying the error.

Usage:
    python -m puntfit.parity --out docs/parity-check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import marcel
from .crosswalk import normalize_name

AGE_TOLERANCE = 0.5      # years, comfortably wider than any rounding
MIN_MINUTES = 500        # only compare players with a real workload
MINUTE_TOLERANCE = 1.0   # minutes are stored at different precision per source

# fields all three sources carry (Basketball Monster has no turnovers)
SHARED = ["g", "min", "pts", "tpm", "reb", "ast", "stl", "blk",
          "fgm", "fga", "ftm", "fta"]
COUNTING = [f for f in SHARED if f not in ("min",)]

# What counts as a regular season game in the Kaggle feed.
#
# Defined by exclusion, because the in-season tournament's gameType label
# changes every year ("NBA Emirates Cup" in 2024, "Emirates NBA Cup" in 2026,
# also "NBA Cup" and "in-season-knockout"). Its group and knockout games do
# count toward regular season stats; only the championship final does not,
# and that one game is reliably marked with a "Championship" sub-label.
#
# Filtering on gameType == "Regular Season" instead drops 66 real games from
# 2024, which collapses agreement with the other two sources to 4%.
#
# The final is marked "Championship" in 2026 but carries no sub-label at all
# in 2024, so it is identified as a Cup game with no round sub-label: every
# game that does count names its round ("East Group A", "West Semifinal").
NOT_REGULAR_SEASON = {"Preseason", "Pre Season", "Playoffs", "All-Star Game",
                      "Play-in Tournament"}


def _cup_final(d: pd.DataFrame) -> pd.Series:
    cup = d.gameType.str.contains("Cup", case=False, na=False)
    return cup & (d.gameSubLabel.isna() | (d.gameSubLabel == "Championship"))

KAGGLE_STATS = {
    "points": "pts", "threePointersMade": "tpm", "reboundsTotal": "reb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
    "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
    "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
}


def kaggle_seasons(kaggle_dir: Path, first: int = 2002, last: int = 2026) -> pd.DataFrame:
    cols = ["personId", "gameDate", "gameType", "gameSubLabel", "numMinutes",
            *KAGGLE_STATS]
    d = pd.read_csv(kaggle_dir / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    d["gameDate"] = pd.to_datetime(d.gameDate, errors="coerce")
    d["season"] = np.where(d.gameDate.dt.month >= 10,
                           d.gameDate.dt.year + 1, d.gameDate.dt.year)
    for c in ["numMinutes", *KAGGLE_STATS]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[~d.gameType.isin(NOT_REGULAR_SEASON)
          & ~_cup_final(d)
          & (d.numMinutes.fillna(0) > 0)
          & d.season.between(first, last)]
    return d.groupby(["personId", "season"]).agg(
        g=("gameDate", "size"),
        **{"min": ("numMinutes", "sum")},
        **{new: (old, "sum") for old, new in KAGGLE_STATS.items()},
    ).reset_index()


def assemble(hoopr: pd.DataFrame, kag: pd.DataFrame, cw: pd.DataFrame,
             bm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per player-season with each source's figures side by side.

    Returns (frame, rejected_bm_matches).
    """
    # rename explicitly: a merge only suffixes columns present on both sides,
    # so relying on suffixes leaves fields like `age` silently unlabelled
    tagged = SHARED + ["tov", "age"]
    h = (hoopr.merge(cw[["espn_athlete_id", "nba_person_id"]],
                     left_on="athlete_id", right_on="espn_athlete_id")
              .rename(columns={f: f + "_h" for f in tagged}))
    k = kag.rename(columns={f: f + "_k" for f in tagged if f in kag.columns})
    m = h.merge(k, left_on=["nba_person_id", "season"],
                right_on=["personId", "season"])
    m["key"] = m.name.map(normalize_name)

    bm = bm.copy()
    bm["key"] = bm.name.map(normalize_name)
    bm = bm[["key", "season", "age", *SHARED]].add_suffix("_bm")
    joined = m.merge(bm, left_on=["key", "season"],
                     right_on=["key_bm", "season_bm"], how="left")

    has_bm = joined.age_bm.notna()
    mismatch = has_bm & ((joined.age_h - joined.age_bm).abs() > AGE_TOLERANCE)
    rejected = joined[mismatch].copy()
    for f in SHARED:                      # drop the unverified BM figures
        joined.loc[mismatch, f + "_bm"] = np.nan
    return joined[joined["min_h"] >= MIN_MINUTES], rejected


def compare(df: pd.DataFrame, left: str, right: str, fields=SHARED) -> pd.DataFrame:
    rows = []
    for f in fields:
        pair = df[[f"{f}_{left}", f"{f}_{right}"]].dropna()
        if pair.empty:
            continue
        x, y = pair.iloc[:, 0].astype(float), pair.iloc[:, 1].astype(float)
        diff = x - y
        rel = (diff.abs() / y.replace(0, np.nan)).dropna()
        close = MINUTE_TOLERANCE if f == "min" else 0
        rows.append({"field": f, "n": len(pair),
                     "agree_pct": 100 * (diff.abs() <= close).mean(),
                     "within_1pct": 100 * (rel <= 0.01).mean(),
                     "median_abs_diff": diff.abs().median(),
                     "mean_rel_diff_pct": 100 * rel.mean()})
    return pd.DataFrame(rows)


def odd_one_out(df: pd.DataFrame, fields=COUNTING) -> pd.DataFrame:
    """Where all three sources have a figure, which one stands alone?"""
    rows = []
    for f in fields:
        t = df[[f"{f}_h", f"{f}_k", f"{f}_bm"]].dropna()
        if t.empty:
            continue
        h, k, b = (t.iloc[:, i].astype(float).round(0) for i in range(3))
        rows.append({
            "field": f, "n": len(t),
            "all_agree_pct": 100 * ((h == k) & (k == b)).mean(),
            "hoopR_odd": int(((k == b) & (h != k)).sum()),
            "kaggle_odd": int(((h == b) & (k != h)).sum()),
            "bm_odd": int(((h == k) & (b != h)).sum()),
            "no_majority": int(((h != k) & (k != b) & (h != b)).sum()),
        })
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    out = [head, sep]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            cells.append(f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"
                         if isinstance(v, (int, np.integer)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoopr", default="data/nba_seasons.parquet", type=Path)
    ap.add_argument("--kaggle", default="data/Kaggle Data Set", type=Path)
    ap.add_argument("--seasons", default="data/seasons", type=Path)
    ap.add_argument("--crosswalk", default="data/crosswalk.csv", type=Path)
    ap.add_argument("--out", default="docs/parity-check.md", type=Path)
    a = ap.parse_args()

    hoopr = pd.read_parquet(a.hoopr)
    frame, rejected = assemble(hoopr, kaggle_seasons(a.kaggle),
                               pd.read_csv(a.crosswalk), marcel.load_all(a.seasons))
    triple = frame[frame.age_bm.notna()]

    hk = compare(frame, "h", "k")
    bh = compare(triple, "bm", "h")
    bk = compare(triple, "bm", "k")
    odd = odd_one_out(triple)

    lines = [
        "# Three-way data parity check", "",
        "Run before switching the projection input away from Basketball Monster.",
        f"Players with {MIN_MINUTES}+ minutes in a season.", "",
        "## 1. hoopR vs Kaggle, full population",
        "", f"{len(frame):,} player-seasons, 2002-2026, matched on name+birthdate.", "",
        _table(hk), "",
        "## 2. Basketball Monster vs hoopR",
        "", f"{len(triple):,} player-seasons, 2022-2026, top 234 only.", "",
        _table(bh), "",
        "## 3. Basketball Monster vs Kaggle", "",
        _table(bk), "",
        "## 4. Who is wrong when they disagree", "",
        "Counts over the rows where all three sources have a figure. When two "
        "agree and one differs, the one standing alone carries the error.", "",
        _table(odd), "",
        f"Name matches rejected by the age check: {len(rejected)}.", "",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")

    print(f"full population: {len(frame):,} player-seasons | triple overlap: {len(triple):,}")
    print(f"BM name matches rejected by age check: {len(rejected)}")
    print("\n1. hoopR vs Kaggle (all players)\n", hk.to_string(index=False))
    print("\n2. BM vs hoopR\n", bh.to_string(index=False))
    print("\n3. BM vs Kaggle\n", bk.to_string(index=False))
    print("\n4. odd one out\n", odd.to_string(index=False))
    print(f"\nwrote {a.out}")
