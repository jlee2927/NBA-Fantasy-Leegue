import numpy as np
import pandas as pd
import pytest

import puntfit.marcel as M


def _season(season, rows):
    df = pd.DataFrame(rows)
    df["season"] = season
    for s in M.STATS:
        if s not in df:
            df[s] = 0.0
    df[M.PLAYER] = df.name          # both loaders set the key column
    df["team"], df["pos"], df["inj"] = "X", "C", None
    return df


def test_age_back_computed_from_export_date():
    # 31.6 at export (Sep 19 2026) -> ~31.97 on Feb 1 2027, ~27.97 on Feb 1 2023
    a = pd.Series([31.6])
    assert M.age_on_feb1(a, 2027).iloc[0] == pytest.approx(31.97, abs=0.01)
    assert M.age_on_feb1(a, 2023).iloc[0] == pytest.approx(27.97, abs=0.01)


def test_rate_matches_hand_calc_with_regression():
    rows = lambda pts: [{"name": "A", "g": 50, "min": 1000, "pts": pts, "age": 27},
                        {"name": "B", "g": 50, "min": 1000, "pts": 500, "age": 27}]
    df = pd.concat([_season(2021, rows(300)), _season(2022, rows(400)), _season(2023, rows(600))])
    k = {s: 1000 for s in M.STATS}
    p = M.marcel(df, 2024, k, 0, 0, aging="none")
    lg = (5 * 1100 / 2000 + 4 * 900 / 2000 + 3 * 800 / 2000) / 12
    expected = (5 * 600 + 4 * 400 + 3 * 300 + 1000 * lg) / (12 * 1000 + 1000)
    assert p.loc["A", "pts_rate"] == pytest.approx(expected)


def test_missing_season_carries_no_weight_in_rate():
    rows = [{"name": "A", "g": 50, "min": 1000, "pts": 500, "age": 27}]
    df = pd.concat([_season(2022, rows), _season(2023, rows)])  # absent in 2021
    p = M.marcel(df, 2024, {s: 0 for s in M.STATS}, 0, 0, aging="none")
    assert p.loc["A", "pts_rate"] == pytest.approx(0.5)


def test_minutes_projection_formula():
    rows = lambda m: [{"name": "A", "g": 60, "min": m, "age": 27}]
    df = pd.concat([_season(2021, rows(1000)), _season(2022, rows(2000)), _season(2023, rows(2500))])
    p = M.marcel(df, 2024, {s: 0 for s in M.STATS}, 300, 0, aging="none")
    assert p.loc["A", "proj_min"] == pytest.approx(0.5 * 2500 + 0.1 * 2000 + 300)


def test_two_players_sharing_a_name_are_kept_apart():
    """hoopR has 12 names covering two different people. Keying on name
    merged Chris Johnson's 2013 (29 games) with another Chris Johnson's
    (8 games) into a 37-game player who never existed."""
    def rows(season):
        return _season(season, [{"name": "Chris Johnson", "g": 29, "min": 285,
                                 "pts": 117, "age": 27},
                                {"name": "Chris Johnson", "g": 8, "min": 101,
                                 "pts": 29, "age": 23}])
    df = pd.concat([rows(2021), rows(2022), rows(2023)])
    df[M.PLAYER] = list(range(2)) * 3          # same name, different ids

    p = M.marcel(df, 2024, {s: 0 for s in M.STATS}, 0, 0, aging="none")
    assert len(p) == 2
    assert p["pts_rate"].round(6).tolist() == [round(117 / 285, 6), round(29 / 101, 6)]


def test_stats_are_read_off_the_data_not_a_global():
    """Basketball Monster exports have no turnovers; hoopR always does.
    Loading one must not change what the other sees."""
    with_tov = _season(2023, [{"name": "A", "g": 1, "min": 10, "age": 27}])
    assert "tov" in M.stats_in(with_tov)
    assert "tov" not in M.stats_in(with_tov.drop(columns=["tov"]))
    assert "tov" in M.stats_in(with_tov)        # unchanged by the call above


def test_tango_age_factor():
    assert M.tango_factor(23, "pts") == pytest.approx(1.036)
    assert M.tango_factor(33, "pts") == pytest.approx(0.988)
    assert M.tango_factor(23, "tov") == 1.0
