"""hoopR (ESPN) data layer for PuntFit.

Downloads NBA player box scores and player bios from the sportsdataverse
hoopR-nba-data repo, caches them under data/hoopr/, and aggregates them into
one row per player-season carrying the columns the projection code expects.

This is the only module that knows where NBA data comes from.

Feed choice: hoopR publishes two NBA feeds. This uses only hoopR-nba-data,
the ESPN feed, keyed on `athlete_id`. The NBA Stats feed
(hoopR-nba-stats-data) uses a different player ID space, so the two must
never be joined.

Season labels are the season's ENDING year, matching this project: season
2026 covers games from October 2025 to June 2026.

Data notes:
  * Games played counts only games the player actually appeared in
    (minutes > 0). DNP rows carry null minutes and are excluded.
  * `g_share` is games divided by the games the player's main team played
    that season, so the lockout (2012, 66 games), the shortened 2020 season
    and the 72-game 2021 season do not drag down a games baseline fitted
    across eras.
  * Birthdates come from the player bio files, unioned across every season
    and de-duplicated, because a player's bio row only appears in seasons
    they were active. Height and weight are deliberately NOT taken from
    there: ESPN overwrites those with current values, so historical rows
    would be wrong. A birthdate cannot change, so it is safe.
  * Exhibition squads (All-Star, Rising Stars) appear as extra team ids
    playing one or two games. Teams below MIN_TEAM_GAMES are ignored when
    working out how long a season was.

Usage:
    python -m puntfit.fetch_nba --out data/nba_seasons.parquet
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

REPO = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
FIRST_SEASON, LAST_SEASON = 2002, 2026
REGULAR_SEASON = 2      # ESPN season_type code
MIN_TEAM_GAMES = 50     # below this a "team" is an exhibition squad, not a club

# box score column -> the name the projection code uses
STAT_COLUMNS = {
    "points": "pts",
    "three_point_field_goals_made": "tpm",
    "rebounds": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
}


def _read_parquet(path: Path) -> pd.DataFrame:
    """pyarrow 19 raises "Repetition level histogram size mismatch" on the
    2002-2020 files, which fastparquet reads without complaint."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def cached(kind: str, season: int, cache: Path, refresh: bool = False) -> Path:
    path = cache / kind / f"{kind}_{season}.parquet"
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{REPO}/{kind}/parquet/{kind}_{season}.parquet", timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def birthdates(cache: Path, seasons: range, refresh: bool = False) -> pd.DataFrame:
    """One row per player: birthdate only, unioned over every season file."""
    frames = []
    for s in seasons:
        core = _read_parquet(cached("player_core", s, cache, refresh))
        frames.append(core[["athlete_id", "date_of_birth"]])
    both = pd.concat(frames).dropna(subset=["date_of_birth"])
    both["dob"] = pd.to_datetime(both["date_of_birth"], format="mixed",
                                 utc=True).dt.tz_localize(None)
    return both.drop_duplicates("athlete_id").set_index("athlete_id")[["dob"]]


def season_totals(box: pd.DataFrame, season: int) -> pd.DataFrame:
    """Aggregate one season of box scores to one row per player."""
    regular = box[box.season_type == REGULAR_SEASON]
    schedule = regular.groupby("team_id").game_id.nunique()
    schedule = schedule[schedule >= MIN_TEAM_GAMES]

    # The All-Star game is tagged as regular season in this feed, under its own
    # one-game "Conf All-Stars" team ids. Counting it would add a game and a
    # night's stats to every All-Star's season, so club teams only.
    played = regular[(regular.minutes.fillna(0) > 0)
                     & (regular.team_id.isin(schedule.index))]
    out = played.groupby("athlete_id").agg(
        name=("athlete_display_name", "last"),
        g=("game_id", "nunique"),
        **{"min": ("minutes", "sum")},
        **{new: (old, "sum") for old, new in STAT_COLUMNS.items()},
    )

    appearances = played.groupby(["athlete_id", "team_id"]).game_id.nunique()
    main = appearances.sort_values().groupby("athlete_id").tail(1).reset_index("team_id")
    out["team_games"] = main.team_id.map(schedule)
    out["team"] = played.groupby("athlete_id").team_abbreviation.last()
    out["pos"] = played.groupby("athlete_id").athlete_position_abbreviation.last()
    out["g_share"] = out.g / out.team_games
    out["season"] = season
    return out


def build(cache: Path, first: int = FIRST_SEASON, last: int = LAST_SEASON,
          refresh: bool = False) -> pd.DataFrame:
    seasons = range(first, last + 1)
    dob = birthdates(cache, seasons, refresh)
    frames = []
    for s in seasons:
        box = _read_parquet(cached("player_box", s, cache, refresh))
        frames.append(season_totals(box, s))
    df = pd.concat(frames).reset_index()

    df["dob"] = df.athlete_id.map(dob.dob)
    feb1 = pd.to_datetime(dict(year=df.season, month=2, day=1))
    df["age"] = (feb1 - df.dob).dt.days / 365.25
    df["inj"] = None  # Basketball Monster carried this; ESPN box scores do not
    return df.drop(columns=["dob"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/hoopr", type=Path)
    ap.add_argument("--out", default="data/nba_seasons.parquet", type=Path)
    ap.add_argument("--first", default=FIRST_SEASON, type=int)
    ap.add_argument("--last", default=LAST_SEASON, type=int)
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    a = ap.parse_args()

    df = build(a.cache, a.first, a.last, a.refresh)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} player-seasons, {a.first}-{a.last} -> {a.out}")
    print(df.groupby("season").agg(players=("name", "size"),
                                   team_games=("team_games", "median")).to_string())
