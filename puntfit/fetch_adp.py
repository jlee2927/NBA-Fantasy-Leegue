"""Where the market is taking players, against where this engine ranks them.

A projection says how good a player is. Average draft position says when he
will actually be gone. They answer different questions, and the gap between
them is the only place a draft is won: a player the room undervalues is worth
waiting on, and one it overvalues is worth letting go.

They are kept apart on purpose. ADP is never folded into a player's value,
because blending the two would make these rankings a partial copy of the
consensus they exist to beat, and would leave neither number meaning
anything.

Sources, in the order they are trusted:

  * data/adp_manual.csv, hand-entered. Yahoo publishes ADP but only behind
    an OAuth application, which is not something this can set up on a user's
    behalf, so Yahoo numbers are typed in. Whatever a manager's own league
    platform says is the authority anyway.
  * ESPN's fantasy API, which publishes a draft rank per player. Note it
    publishes for a season only once that season's rankings are out: a
    projection built for 2026-27 may find ESPN still carrying 2025-26.

Keyed on athlete_id. ESPN's fantasy player id is the same id the rest of the
pipeline uses - checked against the projection set, 473 of 473 matching - so
no crosswalk is involved.

Usage:
    python -m puntfit.fetch_adp --season 2027 --out data/adp.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

ESPN = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/players"

# ESPN's lineup slot ids. Only the five real positions matter here; the rest
# are the flex and bench slots a roster template supplies itself.
SLOTS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}

COLUMNS = ["athlete_id", "name", "adp", "espn_rank", "owned", "espn_slots", "source"]


def fetch(season: int, limit: int = 600, timeout: int = 60) -> pd.DataFrame:
    """Draft rank, ownership and fantasy eligibility, straight from ESPN."""
    # Sorted by draft rank rather than ownership on purpose. ESPN returns the
    # rank field only when it is the sort key - ask for ownership order and
    # every draftRanksByRankType comes back null, which reads exactly like a
    # season whose rankings are not out yet.
    flt = {"players": {"limit": limit,
                       "sortDraftRanks": {"sortPriority": 1, "sortAsc": True,
                                          "value": "STANDARD"}}}
    r = requests.get(ESPN.format(season=season),
                     headers={"User-Agent": "Mozilla/5.0",
                              "x-fantasy-filter": json.dumps(flt)},
                     # kona_player_info, not players_wl: the lightweight view
                     # omits draftRanksByRankType entirely, which looks
                     # identical to a season with no rankings published.
                     params={"scoringPeriodId": 0, "view": "kona_player_info"},
                     timeout=timeout)
    r.raise_for_status()

    rows = []
    for p in r.json():
        ranks = (p.get("draftRanksByRankType") or {}).get("STANDARD") or {}
        slots = [SLOTS[s] for s in (p.get("eligibleSlots") or []) if s in SLOTS]
        rows.append({
            "athlete_id": p.get("id"),
            "name": p.get("fullName"),
            "espn_rank": ranks.get("rank"),
            "owned": (p.get("ownership") or {}).get("percentOwned"),
            "espn_slots": ",".join(slots),
        })
    df = pd.DataFrame(rows).dropna(subset=["athlete_id", "name"])
    return df.astype({"athlete_id": "int64"})


def apply_overrides(feed: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Hand-entered ADP wins, because a manager's own platform is the one
    that decides when a player actually goes."""
    out = feed.copy()
    out["adp"] = out["espn_rank"]
    out["source"] = out.adp.map(lambda v: "espn" if pd.notna(v) else "")
    if not path.exists():
        return out.reindex(columns=COLUMNS)

    manual = pd.read_csv(path).astype({"athlete_id": "int64"})
    by_id = dict(zip(manual.athlete_id, manual.adp))
    source = dict(zip(manual.athlete_id,
                      manual.get("source", pd.Series(["manual"] * len(manual)))))
    hit = out.athlete_id.isin(by_id)
    out.loc[hit, "adp"] = out.loc[hit, "athlete_id"].map(by_id)
    out.loc[hit, "source"] = out.loc[hit, "athlete_id"].map(source).fillna("manual")
    return out.reindex(columns=COLUMNS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--out", default="data/adp.csv", type=Path)
    ap.add_argument("--overrides", default="data/adp_manual.csv", type=Path)
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()

    df = apply_overrides(fetch(a.season, a.limit), a.overrides)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    have = int(df.adp.notna().sum())
    print(f"{len(df)} players -> {a.out}")
    print(f"  with an ADP: {have}")
    if not have:
        print(f"  ESPN has not published draft ranks for season {a.season} yet.")
        print(f"  Type Yahoo's into {a.overrides} as athlete_id,adp,source.")
    print(f"  with ESPN fantasy eligibility: {int((df.espn_slots != '').sum())}")
