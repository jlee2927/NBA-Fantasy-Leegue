"""Fantasy positions and roster slots.

ESPN publishes only G, F and C - its roster API returns `abbreviation='G'`
for every guard alive - but a fantasy roster has PG, SG, SF, PF and C slots.
The missing half is the split inside each group, and that is the part the
box score can actually answer: what separates a point guard from a shooting
guard is how much he passes, and what separates a power forward from a small
forward is how much of the floor he occupies.

So ESPN's coarse call is kept, because it is reliable, and only the split is
inferred. Nothing here overrides ESPN on whether somebody is a guard.

Players near a boundary get both positions, which is not a hedge: real
fantasy platforms award dual eligibility for exactly those players, and a
roster template that insists every guard is one or the other would be more
confident than the evidence and less useful than the real thing.

Overrides: data/positions_manual.csv wins over everything derived here, for
the same reason data/injuries_manual.csv wins over the injury feed. Position
eligibility is set by the platform a league runs on - Yahoo and ESPN disagree
with each other and with real life - so the authoritative answer lives in the
user's league, not in any feed. The file is keyed on athlete_id and takes a
comma-separated list:

    athlete_id,positions
    3945274,PG,SG
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ALL = ("PG", "SG", "SF", "PF", "C")

# Which positions can start in which slot.
FILLS = {
    "PG": {"PG"}, "SG": {"SG"}, "SF": {"SF"}, "PF": {"PF"}, "C": {"C"},
    "G": {"PG", "SG"},
    "F": {"SF", "PF"},
    "UTIL": set(ALL),
    "BE": set(ALL),
}

# The starting shape, then bench. Matches the common Yahoo category league.
STARTERS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL")
BENCH = "BE"

# Players between these percentiles of their group get both positions. The
# band is wide because the boundary is genuinely blurred - a combo guard is a
# real thing, not a measurement failure.
LOWER, UPPER = 0.40, 0.60


def slots_for(rounds: int) -> list[str]:
    """The roster template for a draft of this length.

    Short drafts lose bench spots first and then the flex, because a league
    that only rosters six players still needs them spread across the floor.
    """
    if rounds >= len(STARTERS):
        return list(STARTERS) + [BENCH] * (rounds - len(STARTERS))
    return list(STARTERS[:rounds])


def derive(df: pd.DataFrame, pool_size: int = 156) -> pd.Series:
    """Positions per player, as a tuple, from ESPN's group plus the split.

    Thresholds are percentiles of the draftable pool rather than fixed
    numbers, so they do not drift as the league's pace changes.
    """
    pool = df.nlargest(min(pool_size, len(df)), "proj_mpg")
    per36 = lambda frame, col: frame[col] / frame.proj_mpg.replace(0, pd.NA) * 36

    # Guards split on passing, forwards on the rebounding and rim protection
    # that distinguishes a four from a three.
    guard_lo, guard_hi = per36(pool[pool.pos == "G"], "ast_pg").quantile([LOWER, UPPER])
    big = per36(pool[pool.pos == "F"], "reb_pg") + 2 * per36(pool[pool.pos == "F"], "blk_pg")
    fwd_lo, fwd_hi = big.quantile([LOWER, UPPER])

    out = {}
    for name, row in df.iterrows():
        minutes = row.proj_mpg or 0
        if row.pos == "C":
            out[name] = ("C",)
            continue
        if not minutes:
            # No projected minutes means no basis for a split; keep it coarse.
            out[name] = ("PG", "SG") if row.pos == "G" else ("SF", "PF")
            continue
        if row.pos == "G":
            ast = row.ast_pg / minutes * 36
            out[name] = ("PG",) if ast >= guard_hi else ("SG",) if ast <= guard_lo \
                else ("PG", "SG")
        else:
            size = (row.reb_pg + 2 * row.blk_pg) / minutes * 36
            out[name] = ("PF",) if size >= fwd_hi else ("SF",) if size <= fwd_lo \
                else ("SF", "PF")
    return pd.Series(out, name="positions")


def apply_overrides(positions: pd.Series, df: pd.DataFrame, path: Path) -> pd.Series:
    """Hand-maintained eligibility replaces anything derived."""
    if not path.exists():
        return positions
    manual = pd.read_csv(path, header=None, names=["athlete_id", "positions"],
                         skiprows=1, dtype=str)
    by_id = {float(i): tuple(p.split(",")) for i, p in
             zip(manual.athlete_id, manual.positions) if pd.notna(i)}
    out = positions.copy()
    for name, row in df.iterrows():
        wanted = by_id.get(row.get("player"))
        if wanted:
            out[name] = tuple(p.strip().upper() for p in wanted)
    return out


def assign(players: list[tuple[str, tuple]], slots: list[str]) -> list:
    """Fit players into slots, filling as many as possible.

    Deliberately a maximum matching rather than filling greedily in draft
    order. Greedy gets this wrong in a way that matters: a dual-eligible
    guard taken first lands in PG, and then a pure point guard drafted later
    has nowhere to go while SG sits empty. Matching reshuffles the earlier
    player instead, which is what a real lineup editor would let you do.
    """
    filled = [None] * len(slots)

    def place(p, seen):
        for i, slot in enumerate(slots):
            if seen[i] or not (FILLS[slot] & set(players[p][1])):
                continue
            seen[i] = True
            if filled[i] is None or place(filled[i], seen):
                filled[i] = p
                return True
        return False

    for p in range(len(players)):
        place(p, [False] * len(slots))
    return filled


def roster_view(roster: list[str], positions: pd.Series, rounds: int) -> dict:
    """The sidebar's picture of a roster: every slot, and who is in it."""
    slots = slots_for(rounds)
    players = [(name, positions.get(name, ALL)) for name in roster]
    filled = assign(players, slots)

    rows = [{"slot": slot,
             "player": players[filled[i]][0] if filled[i] is not None else None,
             "positions": "/".join(players[filled[i]][1]) if filled[i] is not None else ""}
            for i, slot in enumerate(slots)]

    counts = {p: 0 for p in ALL}
    for _, eligible in players:
        for p in eligible:
            counts[p] += 1

    starters = [r for r in rows if r["slot"] != BENCH]
    return {
        "rows": rows,
        "counts": counts,
        "filled": len(roster),
        "total": len(slots),
        "starters_open": sum(1 for r in starters if r["player"] is None),
        "unplaced": len(roster) - sum(1 for f in filled if f is not None),
    }
