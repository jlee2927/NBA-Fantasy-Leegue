"""How much of each category's ranking is real.

A z-score treats every category as equally knowable. They are not. Rebounds
project well; steals barely do. Ranking a player on the plain average of nine
z-scores therefore spends as much of his value on a number that is mostly
noise as on one that is mostly signal.

The fix is the standard one for a score measured with error: shrink it toward
zero by how reliably it is measured. Reliability here is the correlation
between what the engine projected and what actually happened, so a category
the engine cannot predict contributes less to the total.

Measured walk-forward, like everything else: the reliability used for a
target season comes only from seasons before it.

A warning about the obvious measure. Error relative to a category's mean says
how close the number is, not whether the ranking holds up, and the two
disagree sharply. Blocks have the worst relative error of any category and
one of the best reliabilities, because the gap between a rim protector and a
guard is so wide that even a poor projection sorts them correctly. Steals are
the reverse: almost everyone lands between half a steal and one and a half, so
projection noise swamps the real differences. Ranking is what a draft board
does, so ranking is what is measured.

Usage:
    python -m puntfit.reliability --out data/reliability.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import marcel as M
from .categories import RATIO

MIN_MINUTES = 500


def _wcorr(x, y, w) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(cov / (sx * sy)) if sx and sy else 0.0


def season_reliability(df: pd.DataFrame, target: int, proj: pd.DataFrame,
                       stats) -> dict:
    """Correlation between projection and outcome, per category, for one season."""
    act = df[(df.season == target) & (df["min"] >= MIN_MINUTES)].set_index(M.PLAYER)
    names = act.index.intersection(proj.index)
    a, p = act.loc[names], proj.loc[names]
    if len(names) < 50:
        return {}

    out = {}
    for s in stats:
        out[s] = _wcorr((p[s + "_rate"] * 36).values,
                        (a[s] / a["min"] * 36).values, a["min"].values)
    for cat, (made, att) in RATIO.items():
        made, att = made.replace("_pg", ""), att.replace("_pg", "")
        out[cat] = _wcorr((p[made + "_rate"] / p[att + "_rate"]).values,
                          (a[made] / a[att]).values, a[att].values)
    return out


def measure(df: pd.DataFrame, through: int, first: int = 2010,
            k: int = 2500, min_base: float = 500,
            game_base: float = 0.25) -> dict:
    """Reliability per category, averaged over every season up to `through`."""
    stats = M.stats_in(df)
    rows = []
    for t in range(first, through + 1):
        curve = M.age_curve(df, t - 1)
        proj = M.marcel(df, t, {s: k for s in stats}, min_base, game_base,
                        aging="delta", curve=curve)
        got = season_reliability(df, t, proj, stats)
        if got:
            rows.append(got)
    if not rows:
        return {}
    return {c: float(np.mean([r[c] for r in rows if c in r]))
            for c in rows[-1]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="data/nba_seasons.parquet", type=Path)
    ap.add_argument("--out", default="data/reliability.json", type=Path)
    ap.add_argument("--through", type=int, default=2025)
    a = ap.parse_args()

    df = M.load_hoopr(a.seasons)
    rel = measure(df, a.through)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"through": a.through, "reliability": rel},
                                indent=2, sort_keys=True))
    print(f"measured through {a.through} -> {a.out}")
    for c, v in sorted(rel.items(), key=lambda kv: kv[1]):
        print(f"  {c:5} {v:.3f}")
