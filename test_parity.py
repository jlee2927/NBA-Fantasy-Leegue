from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puntfit.parity as P

KAGGLE = Path(__file__).parent / "data" / "Kaggle Data Set"
needs_kaggle = pytest.mark.skipif(not KAGGLE.exists(), reason="Kaggle dump not present")


def _games(**over):
    row = {"gameType": "Regular Season", "gameSubLabel": None}
    row.update(over)
    return pd.DataFrame([row])


def test_cup_final_detected_when_marked_championship():
    d = _games(gameType="Emirates NBA Cup", gameSubLabel="Championship")
    assert P._cup_final(d).iloc[0]


def test_cup_final_detected_when_sub_label_is_missing():
    """In 2024 the final carries no sub-label at all, unlike 2026."""
    d = _games(gameType="NBA Cup", gameSubLabel=None)
    assert P._cup_final(d).iloc[0]


def test_cup_group_and_knockout_games_are_kept():
    for sub in ("East Group A", "West Quarterfinal", "East Semifinal"):
        d = _games(gameType="NBA Emirates Cup", gameSubLabel=sub)
        assert not P._cup_final(d).iloc[0], sub


def test_ordinary_regular_season_game_is_not_a_cup_final():
    assert not P._cup_final(_games()).iloc[0]


def test_playoffs_and_preseason_are_not_regular_season():
    for t in ("Playoffs", "Preseason", "All-Star Game", "Play-in Tournament"):
        assert t in P.NOT_REGULAR_SEASON
    assert "Regular Season" not in P.NOT_REGULAR_SEASON


def test_compare_reports_exact_agreement():
    df = pd.DataFrame({"pts_a": [100.0, 200.0, 300.0], "pts_b": [100.0, 200.0, 301.0]})
    out = P.compare(df, "a", "b", fields=["pts"]).iloc[0]
    assert out.n == 3
    assert out.agree_pct == pytest.approx(200 / 3)
    assert out.median_abs_diff == 0.0


def test_odd_one_out_attributes_the_disagreement():
    df = pd.DataFrame({
        "pts_h": [10.0, 10.0, 10.0, 10.0],
        "pts_k": [10.0, 99.0, 10.0, 20.0],
        "pts_bm": [10.0, 10.0, 99.0, 30.0],
    })
    r = P.odd_one_out(df, fields=["pts"]).iloc[0]
    assert r.all_agree_pct == pytest.approx(25.0)
    assert r.kaggle_odd == 1
    assert r.bm_odd == 1
    assert r.no_majority == 1
    assert r.hoopR_odd == 0


@needs_kaggle
def test_no_player_exceeds_the_season_length():
    """Counting the Cup final would push traded players past the schedule."""
    k = P.kaggle_seasons(KAGGLE, first=2022, last=2026)
    longest = k.groupby("season").g.max()
    assert longest[2026] <= 82
    assert longest[2023] <= 82
    assert longest[2024] <= 84   # traded players can exceed 82 legitimately
