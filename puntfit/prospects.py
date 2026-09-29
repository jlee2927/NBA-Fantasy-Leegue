"""Project incoming rookies, who have no NBA seasons for Marcel to read.

Two fitted pieces, both derived from history rather than assumed:

  Translation. What a per-40-minute college rate becomes in the NBA, one
  factor per category, fitted on every player since 2015 who played college
  ball and then an NBA season. ESPN uses the same athlete id in both, so the
  pairs are joined on identity, not on names.

  Playing time. Minutes and games from draft slot. This is the important half:
  per-game fantasy value is mostly made of minutes, and draft position
  predicts a rookie's minutes while college production barely does - college
  minutes per game correlate 0.04 with rookie minutes per game. Teams know
  things about tools, defence and medicals that a box score does not, and the
  draft order carries it.

Every projection is flagged `translated` and carries a confidence, because
these are the least certain numbers in the product and a user deciding whether
to reach for a rookie deserves to know that.

Usage:
    python -m puntfit.prospects --out data/prospect_projections.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ["pts", "tpm", "reb", "ast", "stl", "blk", "tov",
         "fgm", "fga", "ftm", "fta"]
MIN_COLLEGE_MINUTES = 400   # a real college workload, not a bench cameo
MIN_NBA_MINUTES = 500       # enough rookie minutes for the rate to mean something
FULL_SEASON = 82


def transitions(college: pd.DataFrame, nba: pd.DataFrame) -> pd.DataFrame:
    """Each player's last college season beside their first NBA season."""
    last = college.sort_values("season").groupby("athlete_id").last()
    first = nba.sort_values("season").groupby("athlete_id").first()
    both = last.join(first, lsuffix="_c", rsuffix="_n", how="inner")
    return both[(both.season_n > both.season_c) & (both.min_c >= MIN_COLLEGE_MINUTES)]


def fit_translation(pairs: pd.DataFrame) -> dict[str, float]:
    """One multiplier per category, taking per-40 college rates to per-40 NBA.

    A ratio of minute-weighted totals rather than a mean of per-player ratios,
    so a player with a handful of NBA minutes cannot swing a category, and a
    zero college rate cannot divide by zero.
    """
    usable = pairs[pairs.min_n >= MIN_NBA_MINUTES]
    out = {}
    for stat in STATS:
        college = (usable[f"{stat}_c"] / usable.min_c * 40)
        nba = (usable[f"{stat}_n"] / usable.min_n * 40)
        weight = usable.min_n
        out[stat] = float(np.average(nba, weights=weight)
                          / np.average(college, weights=weight))
    return out


# Games are fitted against a floored pick. The curve turns over below pick 3,
# which would hand the first pick fewer games than the third; only two picks a
# draft fall below it, and that gap sits well inside the fit's own ~19-game
# error. Minutes are linear in log(pick) and need no such floor.
GAMES_PICK_FLOOR = 3.0


def fit_playing_time(pairs: pd.DataFrame) -> dict[str, list[float]]:
    """Rookie minutes per game and games played, as a function of draft slot.

    Both against log(pick), since the gap between the 1st and 10th pick is
    worth far more playing time than the gap between the 41st and 50th. Games
    take a squared term because the drop-off steepens at the end of the second
    round, where picks barely play at all; a straight line there predicts 32
    games against an observed 22.
    """
    d = pairs.dropna(subset=["pick"])
    d = d[d.pick > 0]
    mpg = (d.min_n / d.g_n).astype(float)
    return {"mpg": list(np.polyfit(np.log(d.pick.astype(float)), mpg, 1)),
            "games": list(np.polyfit(np.log(d.pick.clip(lower=GAMES_PICK_FLOOR)
                                             .astype(float)), d.g_n.astype(float), 2))}


def playing_time(pick: float, model: dict) -> tuple[float, float]:
    pick = max(float(pick), 1.0)
    mpg = float(np.clip(np.polyval(model["mpg"], np.log(pick)), 6.0, 36.0))
    games = float(np.clip(
        np.polyval(model["games"], np.log(max(pick, GAMES_PICK_FLOOR))),
        10.0, FULL_SEASON))
    return mpg, games


def confidence(pick: float, college_minutes: float) -> str:
    """How much to trust one of these. Lottery picks with a full college
    season are the best case; late picks with little college data the worst."""
    if pick <= 14 and college_minutes >= 600:
        return "medium"
    if pick <= 30 and college_minutes >= 400:
        return "low"
    return "very low"


def project(college: pd.DataFrame, picks: pd.DataFrame, factors: dict,
            model: dict, roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-game projections for a draft class, from their last college season."""
    last = college.sort_values("season").groupby("athlete_id").last()
    joined = picks.join(last, on="athlete_id", rsuffix="_c", how="inner")

    rows = []
    for r in joined.itertuples():
        mpg, games = playing_time(r.pick, model)
        seat = roster.loc[r.athlete_id] if (
            roster is not None and r.athlete_id in roster.index) else None
        row = {"player": int(r.athlete_id), "name": r.name,
               "team": seat.team if seat is not None else None,
               "pos": seat.pos if seat is not None else None,
               "age": None,          # no birthdate published for this class yet
               "proj_mpg": round(mpg, 2), "proj_g": round(games, 1),
               "pick": int(r.pick), "college": r.team,
               "projection_source": "translated",
               "confidence": confidence(r.pick, r.min)}
        for stat in STATS:
            per40 = getattr(r, stat) / r.min * 40 if r.min else 0.0
            row[f"{stat}_pg"] = round(per40 * factors[stat] * mpg / 40, 4)
        row["fg_pct"] = round(row["fgm_pg"] / row["fga_pg"], 4) if row["fga_pg"] else 0.0
        row["ft_pct"] = round(row["ftm_pg"] / row["fta_pg"], 4) if row["fta_pg"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pick")


def backtest(college: pd.DataFrame, nba: pd.DataFrame, pairs: pd.DataFrame,
             seasons: list[int]) -> pd.DataFrame:
    """Fit on everything before a rookie class, then score that class.

    Error is mean absolute error per game, beside the spread of what rookies
    actually do, so a category's error can be read against its own scale.
    """
    rows = []
    for season in seasons:
        train = pairs[pairs.season_n < season]
        test = pairs[pairs.season_n == season].dropna(subset=["pick"])
        test = test[(test.pick > 0) & (test.min_n >= MIN_NBA_MINUTES)]
        if len(train) < 50 or test.empty:
            continue
        factors = fit_translation(train)
        model = fit_playing_time(train)

        for stat in STATS:
            predicted, actual = [], []
            for r in test.itertuples():
                mpg, games = playing_time(r.pick, model)
                per40 = getattr(r, f"{stat}_c") / r.min_c * 40
                predicted.append(per40 * factors[stat] * mpg / 40)
                actual.append(getattr(r, f"{stat}_n") / r.g_n)
            predicted, actual = np.array(predicted), np.array(actual)
            rows.append({"season": season, "stat": stat, "n": len(test),
                         "mae_per_game": float(np.mean(np.abs(predicted - actual))),
                         "actual_mean": float(actual.mean()),
                         "actual_sd": float(actual.std(ddof=0))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--college", default="data/college_seasons.parquet", type=Path)
    ap.add_argument("--nba", default="data/nba_seasons.parquet", type=Path)
    ap.add_argument("--picks", default="data/draft_picks.parquet", type=Path)
    ap.add_argument("--crosswalk", default="data/crosswalk.csv", type=Path)
    ap.add_argument("--out", default="data/prospect_projections.json", type=Path)
    ap.add_argument("--report", default="docs/rookie-backtest.md", type=Path)
    a = ap.parse_args()

    college = pd.read_parquet(a.college)
    nba = pd.read_parquet(a.nba)
    pairs = transitions(college, nba)
    cw = pd.read_csv(a.crosswalk).set_index("espn_athlete_id")
    pairs["pick"] = cw.draftNumber.reindex(pairs.index)

    factors = fit_translation(pairs)
    model = fit_playing_time(pairs)
    picks = pd.read_parquet(a.picks)
    from .fetch_nba import current_roster
    roster = current_roster(Path("data/hoopr"), 2027)
    out = project(college, picks, factors, model, roster)
    out = out[out.team.notna()]      # only players who made a roster

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"params": {"source": "college translation",
                    "training_pairs": int((pairs.min_n >= MIN_NBA_MINUTES).sum()),
                    "factors": {k: round(v, 4) for k, v in factors.items()},
                    "playing_time": model},
         "players": json.loads(out.to_json(orient="records"))}, indent=1))

    scores = backtest(college, nba, pairs, [2024, 2025, 2026])
    if not scores.empty:
        agg = scores.groupby("stat").agg(
            mae=("mae_per_game", "mean"), actual_mean=("actual_mean", "mean"),
            actual_sd=("actual_sd", "mean")).round(2)
        agg["mae_vs_sd"] = (agg.mae / agg.actual_sd).round(2)
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(
            "# Rookie translation backtest\n\n"
            "Fitted only on rookie classes before each test season.\n"
            "`mae_vs_sd` under 1.0 means the projection beats guessing the "
            "class average for that category.\n\n"
            + agg.to_markdown() + "\n")
        print(agg.to_string())
        print()
    print(f"{len(out)} rookies projected -> {a.out}")
    print(out.head(8)[["pick", "name", "proj_mpg", "proj_g", "pts_pg",
                       "reb_pg", "ast_pg", "confidence"]].to_string(index=False))
