"""College and draft data, for players with no NBA history to project from.

Marcel needs prior NBA seasons, so every incoming rookie is missing from the
board entirely - including the top of the draft. This module supplies what a
translation model needs instead: what a player did in college, and where they
were drafted.

The join that usually makes this painful does not exist here. ESPN uses the
same athlete id for a player in college and in the NBA, verified against all
60 picks of the 2026 draft, so a college season and an NBA career line up on
an id rather than on a name. That also means the training set - every player
since 2003 who played college ball and then in the NBA - can be built by an id
join over data we already hold.

Draft position is carried because it predicts NBA minutes far better than
college production does. A second pick plays thirty minutes on a rebuilding
roster whatever his college line; a fiftieth pick plays eight. Minutes are
what per-game fantasy value is mostly made of.

Usage:
    python -m puntfit.fetch_prospects --out data/college_seasons.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests

from .fetch_nba import STAT_COLUMNS, _read_parquet

MBB_REPO = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
NBA_REPO = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
FIRST_SEASON, LAST_SEASON = 2003, 2026
REGULAR_SEASON = 2
ESPN_ID = re.compile(r"a:(\d+)")


def _cached(url: str, path: Path, refresh: bool = False) -> Path:
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def college_box(cache: Path, season: int, refresh: bool = False) -> pd.DataFrame:
    path = cache / "mbb" / f"player_box_{season}.parquet"
    return _read_parquet(_cached(f"{MBB_REPO}/player_box/parquet/{path.name}",
                                 path, refresh))


def season_totals(box: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per college player, same column names the NBA side uses."""
    box = box.copy()
    for col in ["minutes", "athlete_id", *STAT_COLUMNS]:
        box[col] = pd.to_numeric(box[col], errors="coerce")

    played = box[(box.season_type == REGULAR_SEASON) & (box.minutes.fillna(0) > 0)]
    out = played.groupby("athlete_id").agg(
        name=("athlete_display_name", "last"),
        g=("game_id", "nunique"),
        **{"min": ("minutes", "sum")},
        **{new: (old, "sum") for old, new in STAT_COLUMNS.items()},
    )
    out["team"] = played.groupby("athlete_id").team_short_display_name.last()
    out["season"] = season
    out.index = out.index.astype("int64")
    out.index.name = "athlete_id"
    return out


def college_seasons(cache: Path, first: int = FIRST_SEASON,
                    last: int = LAST_SEASON, refresh: bool = False) -> pd.DataFrame:
    frames = [season_totals(college_box(cache, s, refresh), s)
              for s in range(first, last + 1)]
    return pd.concat(frames).reset_index()


def draft_class(cache: Path, season: int, refresh: bool = False) -> pd.DataFrame:
    """Draft picks for a season, keyed on the player's ESPN athlete id.

    The file's own athlete_id column is a draft-entry id, not the player's;
    the real one is embedded in athlete_uid as "s:40~l:46~a:<id>".
    """
    path = cache / "draft" / f"draft_{season}.parquet"
    raw = _read_parquet(_cached(f"{NBA_REPO}/draft/parquet/{path.name}", path, refresh))
    ids = raw.athlete_uid.str.extract(ESPN_ID, expand=False)
    out = pd.DataFrame({
        "athlete_id": pd.to_numeric(ids, errors="coerce"),
        "name": raw.athlete_display_name.values,
        "pick": pd.to_numeric(raw.overall_pick, errors="coerce"),
        "round": pd.to_numeric(raw["round"], errors="coerce"),
        "draft_season": season,
    })
    return out.dropna(subset=["athlete_id"]).astype({"athlete_id": "int64"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/hoopr", type=Path)
    ap.add_argument("--out", default="data/college_seasons.parquet", type=Path)
    ap.add_argument("--draft-out", default="data/draft_picks.parquet", type=Path)
    ap.add_argument("--first", default=FIRST_SEASON, type=int)
    ap.add_argument("--last", default=LAST_SEASON, type=int)
    # Only the most recent file carries athlete_uid, which is the only place
    # the player's own ESPN id appears. Draft position for earlier classes
    # comes from the crosswalk, which already has it for 2,392 players.
    ap.add_argument("--draft-seasons", default="2027")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    college = college_seasons(a.cache, a.first, a.last, a.refresh)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    college.to_parquet(a.out, index=False)

    # hoopR does not publish a draft file for every season, so take what is
    # there. Older classes come from the crosswalk's draft columns instead.
    frames, absent = [], []
    for s in a.draft_seasons.split(","):
        try:
            frames.append(draft_class(a.cache, int(s), a.refresh))
        except requests.HTTPError:
            absent.append(s)
    picks = pd.concat(frames, ignore_index=True)
    if absent:
        print(f"no draft file published for: {', '.join(absent)}")
    picks.to_parquet(a.draft_out, index=False)

    print(f"{len(college):,} college player-seasons {a.first}-{a.last} -> {a.out}")
    print(f"{len(picks)} draft picks -> {a.draft_out}")
    print(picks.groupby("draft_season").size().to_string())
