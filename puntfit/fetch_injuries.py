"""Current injury status for NBA Fantasy Leegue.

Pulls ESPN's live injury feed and writes data/injuries.csv, keyed on the same
athlete_id as the rest of the pipeline. ESPN's own player ids are used, so no
crosswalk is needed.

This is a data layer, not an engine change. It deliberately does NOT alter
anybody's projection.

The reason is the one that motivates freezing the engine at all: a projection
that moves for a reason a user cannot see reads as a broken tool. An injury
feed is live, third-party, and wrong often enough to matter, so wiring it
straight into valuation would let an unverified outside source silently
reorder a draft board mid-draft. Instead the status is attached as a label,
the user sees "Out - torn ACL, expected back July", and a human decides. That
keeps a frozen engine frozen while still surfacing the one input that actually
changes during draft season.

Statistical stats barely move in October: most drafts happen before or within
days of opening night, so there is almost no new box-score information. Injury
and role news is the live signal that matters in that window.

Overrides: data/injuries_manual.csv, if present, wins over the feed. Injury
news moves faster than any aggregator, and on draft day being able to correct
a status by hand matters more than automation.

Usage:
    python -m puntfit.fetch_injuries --out data/injuries.csv
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ESPN_INJURIES = ("https://site.api.espn.com/apis/site/v2/sports/basketball"
                 "/nba/injuries")
PLAYER_ID = re.compile(r"/id/(\d+)")

# Higher means more likely to miss games. Used for sorting and for deciding
# what deserves a warning in the UI, never for changing a projection.
SEVERITY = {"Out": 4, "Doubtful": 3, "Questionable": 2,
            "Game Time Decision": 2, "Day-To-Day": 1}

# The badge sits beside a player's name in a table that is already wider than
# a phone, and "Game Time Decision" is wider than most names. The full text
# stays on the element's title, so nothing is lost by shortening it.
SHORT = {"Out": "OUT", "Doubtful": "DTF", "Questionable": "Q",
         "Game Time Decision": "GTD", "Day-To-Day": "DTD"}

COLUMNS = ["athlete_id", "name", "team", "status", "severity", "body_part",
           "detail", "return_date", "reported", "comment", "source"]


def _athlete_id(athlete: dict) -> int | None:
    for link in athlete.get("links") or []:
        found = PLAYER_ID.search(link.get("href") or "")
        if found:
            return int(found.group(1))
    return None


def parse(payload: dict) -> pd.DataFrame:
    rows = []
    for team in payload.get("injuries") or []:
        for entry in team.get("injuries") or []:
            athlete = entry.get("athlete") or {}
            details = entry.get("details") or {}
            status = entry.get("status")
            rows.append({
                "athlete_id": _athlete_id(athlete),
                "name": athlete.get("displayName"),
                "team": (athlete.get("team") or {}).get("abbreviation"),
                "status": status,
                "severity": SEVERITY.get(status, 0),
                "body_part": details.get("type"),
                "detail": details.get("detail"),
                "return_date": details.get("returnDate"),
                "reported": entry.get("date"),
                "comment": entry.get("shortComment"),
                "source": "espn",
            })
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.dropna(subset=["athlete_id"]).astype({"athlete_id": "int64"})


def fetch(url: str = ESPN_INJURIES, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return parse(r.json())


def apply_overrides(feed: pd.DataFrame, overrides: Path) -> pd.DataFrame:
    """Hand-maintained rows replace the feed for the same player."""
    if not overrides.exists():
        return feed
    manual = pd.read_csv(overrides)
    manual["source"] = "manual"
    manual["severity"] = manual.status.map(SEVERITY).fillna(0).astype(int)
    keep = feed[~feed.athlete_id.isin(manual.athlete_id)]
    return pd.concat([manual, keep], ignore_index=True).reindex(columns=COLUMNS)


def label(projections: pd.DataFrame, injuries: pd.DataFrame,
          key: str = "player") -> pd.DataFrame:
    """Attach status to a projection frame without touching its numbers."""
    status = injuries.set_index("athlete_id")
    out = projections.copy()
    ids = out[key] if key in out.columns else out.index
    out["injury_status"] = ids.map(status.status)
    out["injury_detail"] = ids.map(status.detail)
    out["injury_return"] = ids.map(status.return_date)
    out["injury_severity"] = ids.map(status.severity).fillna(0).astype(int)
    out["injury_short"] = out.injury_status.map(
        lambda v: SHORT.get(v, v) if isinstance(v, str) else v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/injuries.csv", type=Path)
    ap.add_argument("--overrides", default="data/injuries_manual.csv", type=Path)
    ap.add_argument("--url", default=ESPN_INJURIES)
    a = ap.parse_args()

    df = apply_overrides(fetch(a.url), a.overrides)
    df = df.sort_values(["severity", "name"], ascending=[False, True])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    pulled = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"{len(df)} injury entries -> {a.out}  ({pulled})")
    print(df.status.value_counts().to_string())
    if (df.source == "manual").any():
        print(f"{int((df.source == 'manual').sum())} from {a.overrides}")
