"""Category-league valuation: z-scores, volume-weighted percentages, punt fits.

Reads the Marcel projection set and ranks players by punt-adjusted value.

Usage:
    python -m puntfit.valuation --projections data/marcel_projections.json \
        --out docs/ranking-baseline.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from .categories import NEGATIVE, NINE_CAT, RATIO

DEFAULT_POOL = 156   # 12 teams x 13 roster slots
MAX_PASSES = 50


def load_projections(path: Path) -> tuple[pd.DataFrame, dict]:
    with open(path) as f:
        blob = json.load(f)
    return pd.DataFrame(blob["players"]).set_index("name"), blob.get("params", {})


def available_categories(df: pd.DataFrame, cats=NINE_CAT) -> tuple[str, ...]:
    """Categories the projection set can actually supply.

    Turnovers drop out here until the Basketball Monster sheets are
    re-exported with a to/g column, which is what makes 9-cat degrade to
    8-cat rather than producing a column of zeros.
    """
    ok = []
    for c in cats:
        if c in RATIO:
            if all(col in df.columns for col in RATIO[c]):
                ok.append(c)
        elif f"{c}_pg" in df.columns:
            ok.append(c)
    return tuple(ok)


def category_values(df: pd.DataFrame, cats, pool: pd.Index) -> pd.DataFrame:
    """Raw per-category contribution, per game.

    Ratio categories use volume-weighted impact: a percentage on its own says
    nothing about how far it moves a team total, so credit scales with
    attempts. Written as `made - league_rate * attempted` rather than
    `(pct - league_pct) * attempted` because the two are algebraically equal
    and this form leaves a player with zero attempts at zero impact instead
    of dividing by zero.
    """
    ref = df.loc[pool]
    out = pd.DataFrame(index=df.index)
    for c in cats:
        if c in RATIO:
            made, att = RATIO[c]
            out[c] = df[made] - (ref[made].sum() / ref[att].sum()) * df[att]
        else:
            out[c] = df[f"{c}_pg"]
    return out


def z_scores(values: pd.DataFrame, cats, pool: pd.Index) -> pd.DataFrame:
    """Standardise each category against the draftable pool, not the full set."""
    ref = values.loc[pool]
    z = pd.DataFrame(index=values.index)
    for c in cats:
        sd = ref[c].std(ddof=0)
        z[c] = 0.0 if sd == 0 else (values[c] - ref[c].mean()) / sd
        if c in NEGATIVE:
            z[c] = -z[c]
    return z


def value_players(df: pd.DataFrame, cats=NINE_CAT,
                  pool_size: int = DEFAULT_POOL) -> tuple[pd.DataFrame, pd.Index]:
    """Z-scores over a self-consistent pool.

    Means and standard deviations should come from draftable players only,
    but which players are draftable is what the z-scores decide. Seed the
    pool by minutes, re-rank, repeat until it stops moving. A pool that
    oscillates between two states settles on the later one rather than
    spinning until the pass limit.
    """
    cats = available_categories(df, cats)
    n = min(pool_size, len(df))
    pool = df.nlargest(n, "proj_mpg").index
    seen: set[frozenset] = set()
    for _ in range(MAX_PASSES):
        z = z_scores(category_values(df, cats, pool), cats, pool)
        nxt = z[list(cats)].sum(axis=1).nlargest(n).index
        if frozenset(nxt) == frozenset(pool) or frozenset(nxt) in seen:
            pool = nxt
            break
        seen.add(frozenset(pool))
        pool = nxt

    z = z_scores(category_values(df, cats, pool), cats, pool)
    z["total"] = z[list(cats)].sum(axis=1)
    return z.sort_values("total", ascending=False), pool


def punt_value(df: pd.DataFrame, punt, cats=NINE_CAT,
               pool_size: int = DEFAULT_POOL) -> tuple[pd.DataFrame, pd.Index]:
    """Value with `punt` categories dropped.

    The pool is re-derived under the reduced category set, because who counts
    as draftable changes once a category stops being worth anything.
    """
    punted = set(punt)
    kept = tuple(c for c in available_categories(df, cats) if c not in punted)
    return value_players(df, kept, pool_size)


def suggest_builds(df: pd.DataFrame, player: str, n_punts: int, cats=NINE_CAT,
                   pool_size: int = DEFAULT_POOL, top: int = 5) -> pd.DataFrame:
    """Rank every punt build of size `n_punts` by how well `player` fits it.

    Builds with the same punt count keep the same number of categories, so
    their totals are directly comparable.
    """
    cats = available_categories(df, cats)
    rows = []
    for punt in itertools.combinations(cats, n_punts):
        z, _ = punt_value(df, punt, cats, pool_size)
        rows.append({"punt": "+".join(punt),
                     "value": z.loc[player, "total"],
                     "rank": z.index.get_loc(player) + 1})
    return pd.DataFrame(rows).sort_values("value", ascending=False).head(top)


def write_report(df: pd.DataFrame, z: pd.DataFrame, cats, params: dict,
                 out: Path, limit: int = 150) -> None:
    fmt = "9-cat" if "tov" in cats else "8-cat (no turnovers in the projection set)"
    lines = [
        "# Ranking baseline",
        "",
        f"- Projection set: `{params.get('target_season', 'unknown')}`",
        f"- Format: {fmt}",
        f"- Categories: {', '.join(cats)}",
        f"- Pool: top {DEFAULT_POOL} by punt-adjusted value, iterated to convergence",
        "",
        "| # | Player | Team | Pos | Total | " + " | ".join(cats) + " |",
        "|---|---|---|---|---|" + "---|" * len(cats),
    ]
    for i, (name, row) in enumerate(z.head(limit).iterrows(), 1):
        meta = df.loc[name]
        cells = " | ".join(f"{row[c]:+.2f}" for c in cats)
        lines.append(f"| {i} | {name} | {meta['team']} | {meta['pos']} | "
                     f"{row['total']:+.2f} | {cells} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--projections", default="data/marcel_projections.json", type=Path)
    ap.add_argument("--out", default="docs/ranking-baseline.md", type=Path)
    ap.add_argument("--pool", default=DEFAULT_POOL, type=int)
    a = ap.parse_args()

    df, params = load_projections(a.projections)
    cats = available_categories(df)
    z, pool = value_players(df, cats, a.pool)
    write_report(df, z, cats, params, a.out)
    print(json.dumps({"players": len(df), "categories": list(cats),
                      "pool": len(pool), "report": str(a.out)}, indent=1))
