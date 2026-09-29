import json
from pathlib import Path

import pandas as pd
import pytest

import puntfit.prospects as P

DATA = Path(__file__).parent / "data"
COLLEGE = DATA / "college_seasons.parquet"
PICKS = DATA / "prospect_projections.json"
needs_college = pytest.mark.skipif(
    not COLLEGE.exists(), reason="run `python -m puntfit.fetch_prospects` first")
needs_picks = pytest.mark.skipif(
    not PICKS.exists(), reason="run `python -m puntfit.prospects` first")

MODEL = {"mpg": [-6.8, 36.8], "games": [2.0, -22.0, 95.0]}


# ------------------------------------------------------------ playing time
def test_earlier_picks_get_more_minutes():
    minutes = [P.playing_time(p, MODEL)[0] for p in (1, 5, 15, 30, 60)]
    assert minutes == sorted(minutes, reverse=True)


def test_the_first_pick_is_not_handed_fewer_games_than_the_third():
    """The games curve turns over below pick 3, which without a floor would
    rank the top pick behind the third."""
    games = [P.playing_time(p, MODEL)[1] for p in (1, 2, 3)]
    assert games[0] == games[1] == games[2]


def test_playing_time_stays_inside_a_season():
    for pick in (1, 30, 60, 200):
        mpg, games = P.playing_time(pick, MODEL)
        assert 0 < mpg <= 36
        assert 0 < games <= P.FULL_SEASON


# --------------------------------------------------------------- confidence
def test_confidence_falls_away_with_draft_slot():
    assert P.confidence(3, 900) == "medium"
    assert P.confidence(25, 700) == "low"
    assert P.confidence(55, 700) == "very low"
    assert P.confidence(3, 100) == "very low"      # barely played in college


# -------------------------------------------------------------- translation
@needs_college
def test_translation_factors_point_the_way_the_plan_expects():
    college = pd.read_parquet(COLLEGE)
    nba = pd.read_parquet(DATA / "nba_seasons.parquet")
    factors = P.fit_translation(P.transitions(college, nba))

    assert factors["pts"] < 0.9          # scoring drops substantially
    assert factors["blk"] < 0.8          # blocks drop
    assert factors["ast"] > factors["pts"]   # assists travel better than scoring
    assert all(0.3 < v < 1.3 for v in factors.values())


@needs_college
def test_transitions_put_college_before_the_nba():
    college = pd.read_parquet(COLLEGE)
    nba = pd.read_parquet(DATA / "nba_seasons.parquet")
    pairs = P.transitions(college, nba)
    assert len(pairs) > 200
    assert (pairs.season_n > pairs.season_c).all()


# ------------------------------------------------------------- the projections
@needs_picks
def test_every_projected_rookie_is_labelled_and_rostered():
    blob = json.loads(PICKS.read_text())
    assert blob["params"]["training_pairs"] > 200
    for player in blob["players"]:
        assert player["projection_source"] == "translated"
        assert player["confidence"] in {"medium", "low", "very low"}
        assert player["team"], player["name"]
        assert player["proj_mpg"] > 0 and player["proj_g"] > 0


@needs_picks
def test_the_first_pick_outprojects_a_second_rounder():
    players = json.loads(PICKS.read_text())["players"]
    by_pick = {p["pick"]: p for p in players}
    top = min(by_pick)
    late = max(by_pick)
    assert by_pick[top]["proj_mpg"] > by_pick[late]["proj_mpg"]


@needs_picks
def test_rookies_join_the_board_without_displacing_anyone():
    import puntfit.valuation as V
    veterans, _ = V.load_projections(DATA / "marcel_projections.json")
    both, params = V.load_projections(DATA / "marcel_projections.json", PICKS)
    assert params["rookies_added"] > 0
    assert len(both) == len(veterans) + params["rookies_added"]
    assert (both.projection_source == "translated").sum() == params["rookies_added"]
    assert both.tov_pg.notna().all()      # every player still valuable in 9-cat
