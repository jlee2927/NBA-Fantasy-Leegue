"""Match players across data sources by name and birthdate.

hoopR keys players on an ESPN `athlete_id`; the Kaggle NBA dump keys them on
an NBA `personId`. Neither carries the other's id, so comparing the two means
matching players.

Matching on name alone is not good enough: two different Marcus Williamses
were in the league at the same time, and joining them produced a 62-game
player being compared against a 9-game one. Name plus birthdate is effectively
unique for a person, and both sources carry a birthdate for 100% of the
players active in our window.

Any name+birthdate that still lands on more than one id on either side is
dropped rather than guessed at, and written out so the discards can be
inspected.

Validated against hoopR's own published crosswalk (which is exact-name
matched, covers only current players, and leaves 149 of 544 unmatched): this
matcher reproduces 389 of its 395 pairs and disagrees with none of them, while
covering 2,435 players across 2002-2026.

Usage:
    python -m puntfit.crosswalk --out data/crosswalk.csv
"""
from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from .fetch_nba import _read_parquet

CROSSWALK_URL = ("https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data"
                 "/main/nba/crosswalk/parquet/nba_player_crosswalk_{season}.parquet")
SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Strip accents, punctuation, generational suffixes and casing."""
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    plain = SUFFIXES.sub("", plain.lower())
    return " ".join(re.sub(r"[^a-z ]", "", plain).split())


def hoopr_players(cache: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(cache / "player_core" / "*.parquet")))
    cols = ["athlete_id", "display_name", "date_of_birth"]
    both = pd.concat([_read_parquet(f)[cols] for f in files])
    both = both.dropna(subset=["date_of_birth"]).drop_duplicates("athlete_id")
    both["dob"] = pd.to_datetime(both.date_of_birth, format="mixed",
                                 utc=True).dt.tz_localize(None).dt.date
    both["key"] = both.display_name.map(normalize_name)
    return both[["athlete_id", "display_name", "dob", "key"]]


def kaggle_players(kaggle_dir: Path) -> pd.DataFrame:
    p = pd.read_csv(kaggle_dir / "Players.csv").dropna(subset=["birthDate"])
    p["dob"] = pd.to_datetime(p.birthDate, errors="coerce").dt.date
    p["key"] = (p.firstName.astype(str) + " " + p.lastName.astype(str)).map(normalize_name)
    return p[["personId", "dob", "key", "draftYear", "draftRound", "draftNumber"]]


def build(cache: Path, kaggle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (crosswalk, discarded). Discarded rows are ambiguous matches."""
    matched = hoopr_players(cache).merge(kaggle_players(kaggle_dir), on=["key", "dob"])
    ambiguous = (matched.duplicated("athlete_id", keep=False)
                 | matched.duplicated("personId", keep=False))
    clean = matched[~ambiguous].rename(columns={"athlete_id": "espn_athlete_id",
                                                "personId": "nba_person_id",
                                                "display_name": "name"})
    return clean.reset_index(drop=True), matched[ambiguous].reset_index(drop=True)


def validate(crosswalk: pd.DataFrame, season: int = 2026) -> dict:
    """Check against hoopR's published crosswalk for one season."""
    r = requests.get(CROSSWALK_URL.format(season=season), timeout=60)
    r.raise_for_status()
    tmp = Path("/tmp") / f"hoopr_crosswalk_{season}.parquet"
    tmp.write_bytes(r.content)
    official = _read_parquet(tmp).dropna(subset=["nba_player_id"])
    truth = official[["espn_athlete_id", "nba_player_id"]].astype("int64").drop_duplicates()

    ours = crosswalk[["espn_athlete_id", "nba_person_id"]].astype("int64")
    j = truth.merge(ours, on="espn_athlete_id", how="left")
    covered = j.nba_person_id.notna()
    agree = covered & (j.nba_person_id == j.nba_player_id)
    return {"official_pairs": len(truth), "covered": int(covered.sum()),
            "agree": int(agree.sum()),
            "disagree": int((covered & ~agree).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/hoopr", type=Path)
    ap.add_argument("--kaggle", default="data/Kaggle Data Set", type=Path)
    ap.add_argument("--out", default="data/crosswalk.csv", type=Path)
    a = ap.parse_args()

    cw, dropped = build(a.cache, a.kaggle)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cw.to_csv(a.out, index=False)
    dropped.to_csv(a.out.with_name(a.out.stem + "_ambiguous.csv"), index=False)
    print(f"{len(cw)} players matched -> {a.out}")
    print(f"{len(dropped)} ambiguous rows dropped")
    print("validation vs hoopR's own crosswalk:", validate(cw))
