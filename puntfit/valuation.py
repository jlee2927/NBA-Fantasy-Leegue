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

import numpy as np
import pandas as pd

from .categories import NEGATIVE, NINE_CAT, RATIO, STAT_VIEW

DEFAULT_POOL = 156   # 12 teams x 13 roster slots
MAX_PASSES = 50


def load_projections(path: Path, prospects: Path | None = None
                     ) -> tuple[pd.DataFrame, dict]:
    """The projection set, with translated rookies folded in when present.

    Rookies have no NBA seasons for Marcel to read, so they come from a
    separate translation of their college production. They are valued on the
    same footing as everyone else but keep a projection_source of "translated"
    and a confidence, because they are the least certain players on the board.
    """
    with open(path) as f:
        blob = json.load(f)
    players = pd.DataFrame(blob["players"])
    params = blob.get("params", {})
    players["projection_source"] = "marcel"
    players["confidence"] = "high"

    if prospects is not None and Path(prospects).exists():
        with open(prospects) as f:
            extra = json.load(f)
        rookies = pd.DataFrame(extra["players"])
        rookies = rookies[~rookies.name.isin(players.name)]
        # Line the frames up first. An entirely empty column (no birthdate is
        # published for this draft class yet) carries no dtype of its own, and
        # concatenating one changes the dtype of the column it lands in.
        rookies = rookies.dropna(axis=1, how="all")
        columns = players.columns.union(rookies.columns, sort=False)
        players = pd.concat([players.reindex(columns=columns),
                             rookies.reindex(columns=columns)], ignore_index=True)
        params["prospects"] = extra.get("params", {})
        params["rookies_added"] = len(rookies)

    return players.set_index("name"), params


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

    # The same arithmetic as category_values and z_scores, on arrays.
    #
    # Those build about eighteen small Series per pass, and the pool takes
    # several passes to settle, and the build chooser asks for this once per
    # combination of categories - 126 of them for a four-category punt. The
    # pandas overhead, not the arithmetic, was most of a second.
    #
    # Pool selection stays on pandas: nlargest decides ties in a particular
    # way and the chosen pool determines every number that follows, so it is
    # not worth reimplementing to save one call a pass.
    index = df.index
    columns = {}
    for c in cats:
        if c in RATIO:
            made, att = RATIO[c]
            columns[c] = (df[made].to_numpy(float), df[att].to_numpy(float))
        else:
            # A counting category does not depend on the pool, so its
            # contribution is the same on every pass.
            columns[c] = df[f"{c}_pg"].to_numpy(float)

    def z_matrix(positions):
        out = np.empty((len(index), len(cats)))
        for j, c in enumerate(cats):
            held = columns[c]
            if isinstance(held, tuple):
                made, att = held
                v = made - (made[positions].sum() / att[positions].sum()) * att
            else:
                v = held
            ref = v[positions]
            sd = ref.std()                      # ddof=0, as before
            if sd == 0:
                out[:, j] = 0.0
                continue
            col = (v - ref.mean()) / sd
            out[:, j] = -col if c in NEGATIVE else col
        return out

    # Positions throughout rather than labels. nlargest over a positional
    # series breaks ties the same way - first occurrence wins, and the rows
    # are in the same order either way - without translating five hundred
    # names back to offsets on every pass.
    positions = index.get_indexer(df.nlargest(n, "proj_mpg").index)
    seen: set[frozenset] = set()
    matrix, settled = None, False
    for _ in range(MAX_PASSES):
        matrix = z_matrix(positions)
        nxt = pd.Series(matrix.sum(axis=1)).nlargest(n).index.to_numpy()
        same = frozenset(nxt) == frozenset(positions)
        if same or frozenset(nxt) in seen:
            # When the pool has stopped moving, the matrix in hand was built
            # on it already and does not need building again.
            settled = same
            positions = nxt
            break
        seen.add(frozenset(positions))
        positions = nxt

    if not settled:
        matrix = z_matrix(positions)
    pool = index[positions]
    z = pd.DataFrame(matrix, index=index, columns=list(cats))
    # Mean rather than sum, matching Basketball Monster's "Value" column. The
    # ordering is identical either way, but a mean stays comparable across
    # formats: summing gives 9-cat scores an extra category of headroom over
    # 8-cat ones, and punting 3 categories would deflate every score.
    z["total"] = matrix.mean(axis=1)
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


# Basketball Monster's column names for the per-category values, so the board
# reads the same way to anyone who already uses their sheet.
def stat_table(df: pd.DataFrame, cats) -> pd.DataFrame:
    """The same players as the value board, shown as projected statistics.

    A z-score says how a player ranks; it does not say what he is expected to
    do. Both are worth seeing, and a manager who knows the categories often
    reads the raw line faster than the standardised one.

    Returns formatted strings rather than numbers so the template does not
    have to know that percentages want three decimals and counting stats want
    one. A projection that is missing renders as a dash rather than "nan".
    """
    out = pd.DataFrame(index=df.index)
    for c in cats:
        column, _, places = STAT_VIEW[c]
        values = df[column] if column in df.columns else pd.Series(index=df.index)
        out[c] = [f"{v:.{places}f}" if pd.notna(v) else "\u2014" for v in values]
    out["g"] = [f"{v:.0f}" if pd.notna(v) else "\u2014" for v in df["proj_g"]]
    out["m/g"] = [f"{v:.1f}" if pd.notna(v) else "\u2014" for v in df["proj_mpg"]]
    return out


def last_season(seasons: Path, players: pd.DataFrame,
                season: int) -> pd.DataFrame | None:
    """What each player actually did last year, shaped like a projection.

    A projection is easier to trust beside the season it came from. Seeing
    that a player is forecast 25.5 points off 27.7 last year says more about
    the forecast than the forecast alone does.

    The frame deliberately carries the same column names as the projection
    set, so the formatter that renders one renders the other with no special
    case. Joined on athlete_id rather than name, because two players have
    shared a name before and will again.
    """
    if not Path(seasons).exists():
        return None
    df = pd.read_parquet(seasons)
    past = df[df.season == season]
    if past.empty:
        return None

    past = past.set_index("athlete_id")
    games = past.g.replace(0, pd.NA)
    out = pd.DataFrame(index=past.index)
    for stat in ("pts", "reb", "ast", "stl", "blk", "tov", "tpm",
                 "fgm", "fga", "ftm", "fta"):
        out[stat + "_pg"] = past[stat] / games
    out["fg_pct"] = past.fgm / past.fga.replace(0, pd.NA)
    out["ft_pct"] = past.ftm / past.fta.replace(0, pd.NA)
    out["proj_g"] = past.g
    out["proj_mpg"] = past["min"] / games

    # Back onto the projection's own index, so a player who did not play last
    # season simply has no row and renders as dashes.
    ids = players.get("player")
    if ids is None:
        return None
    aligned = out.reindex(ids.values)
    aligned.index = players.index
    return aligned


def stat_headings(cats) -> list[str]:
    return [STAT_VIEW[c][1] for c in cats]


VALUE_LABEL = {"pts": "pV", "tpm": "3V", "reb": "rV", "ast": "aV", "stl": "sV",
               "blk": "bV", "tov": "toV", "fg": "fg%V", "ft": "ft%V"}
RATE_LABEL = {"pts": "p/g", "tpm": "3/g", "reb": "r/g", "ast": "a/g",
              "stl": "s/g", "blk": "b/g", "tov": "to/g"}


def player_table(df: pd.DataFrame, z: pd.DataFrame, cats) -> pd.DataFrame:
    """Per-game production beside each category's value, then the scalar.

    Laid out like the Basketball Monster export: the rate columns first, then
    one value column per category, then Value.
    """
    out = pd.DataFrame(index=z.index)
    out["Team"] = df.team
    out["Pos"] = df.pos
    out["Age"] = df.age.round(1)
    out["g"] = df.proj_g.round(0)
    out["m/g"] = df.proj_mpg.round(1)
    for c in cats:
        if c in RATE_LABEL:
            out[RATE_LABEL[c]] = df[f"{c}_pg"].round(1)
    out["fg%"] = df.fg_pct.round(3)
    out["ft%"] = df.ft_pct.round(3)
    for c in cats:
        out[VALUE_LABEL[c]] = z[c].round(2)
    out["Value"] = z["total"].round(2)
    return out


# Above-average and below-average poles, with white as the neutral midpoint.
#
# "classic" matches Basketball Monster. It is not colour-blind safe: measured in
# OKLab against a deuteranope, its green and red are 4.6 apart, under the 6.0
# floor and far under the target of 8, so roughly 6% of men cannot separate
# them. Fantasy basketball skews male enough that this is not a rounding error.
#
# "cvd" keeps red for below-average and swaps the green for blue, which is the
# standard diverging pair. Worst case across protanopia, deuteranopia and
# tritanopia is 23.8, comfortably clear. Red vs green is the unreadable
# pairing; red vs blue is fine.
#
# Either way the number sits in the cell, so colour reinforces rather than
# carries the meaning.
PALETTES = {
    "cvd": {"high": (42, 120, 214), "low": (208, 59, 59)},
    "classic": {"high": (34, 150, 83), "low": (200, 52, 52)},
}


def _shade(v: float, palette: dict, cap: float = 3.0) -> str:
    if pd.isna(v):
        return ""
    weight = min(abs(v) / cap, 1.0) * 0.75
    r, g, b = palette["high"] if v >= 0 else palette["low"]
    return f"background-color: rgba({r},{g},{b},{weight:.2f})"


def write_board(table: pd.DataFrame, out: Path, cats, title: str,
                palette: str = "cvd") -> None:
    shades = PALETTES[palette]
    value_cols = [VALUE_LABEL[c] for c in cats] + ["Value"]
    head = "".join(f"<th>{c}</th>" for c in ["#", "Player", *table.columns])
    rows = []
    for i, (name, r) in enumerate(table.iterrows(), 1):
        cells = [f"<td class=r>{i}</td>", f"<td class=name>{name}</td>"]
        for col in table.columns:
            style = f' style="{_shade(r[col], shades)}"' if col in value_cols else ""
            cls = " class=r" if col in value_cols or isinstance(r[col], float) else ""
            cells.append(f"<td{cls}{style}>{r[col]}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><meta charset=utf-8>
<title>{title}</title>
<style>
 body{{font:13px/1.4 -apple-system,Segoe UI,sans-serif;margin:24px 16px;color:#111;background:#fff}}
 h1{{font-size:17px;margin:0 0 4px}} p{{color:#666;margin:0 0 16px}}
 /* On a phone the table is far wider than the screen. Scrolling it sideways
    keeps the text readable; letting it shrink to fit does not. */
 .scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
 table{{border-collapse:collapse}} th,td{{padding:3px 7px;border-bottom:1px solid #eee;white-space:nowrap}}
 th{{position:sticky;top:0;background:#fafafa;text-align:left;font-weight:600;border-bottom:2px solid #ddd}}
 td.r{{text-align:right;font-variant-numeric:tabular-nums}}
 td.name{{font-weight:500}} tr:hover td{{background:#f6f9ff}}
</style>
<h1>{title}</h1>
<p>Each value column is standard deviations from the draft-pool average, so
every category carries equal weight. Value is their mean.
{"Blue is above average, red below." if palette == "cvd" else
 "Green is above average, red below - not colour-blind safe."}</p>
<div class=scroll>
<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>
</div>
""")


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
    ap.add_argument("--prospects", default="data/prospect_projections.json", type=Path)
    ap.add_argument("--out", default="docs/ranking-baseline.md", type=Path)
    ap.add_argument("--pool", default=DEFAULT_POOL, type=int)
    ap.add_argument("--values", default="data/player_values.csv", type=Path)
    ap.add_argument("--board", default="docs/draft-board.html", type=Path)
    ap.add_argument("--limit", default=200, type=int)
    ap.add_argument("--palette", choices=tuple(PALETTES), default="cvd",
                    help="cvd: blue/red, readable with colour blindness. "
                         "classic: Basketball Monster's green/red.")
    a = ap.parse_args()

    df, params = load_projections(a.projections, a.prospects)
    cats = available_categories(df)
    z, pool = value_players(df, cats, a.pool)
    write_report(df, z, cats, params, a.out)

    table = player_table(df, z, cats)
    table.to_csv(a.values)
    fmt = "9-cat" if "tov" in cats else "8-cat"
    write_board(table.head(a.limit), a.board, cats,
                f"NBA Fantasy Leegue draft board - {params.get('target_season', '')} ({fmt})",
                a.palette)
    print(json.dumps({"players": len(df), "categories": list(cats),
                      "pool": len(pool), "report": str(a.out),
                      "values": str(a.values), "board": str(a.board)}, indent=1))
