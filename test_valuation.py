from pathlib import Path

import pandas as pd
import pytest

import puntfit.valuation as V
from puntfit.categories import EIGHT_CAT, NINE_CAT

PROJECTIONS = Path(__file__).parent / "data" / "marcel_projections.json"

DEFAULTS = {"pts_pg": 15.0, "tpm_pg": 1.5, "reb_pg": 5.0, "ast_pg": 3.0,
            "stl_pg": 1.0, "blk_pg": 0.5, "tov_pg": 2.0, "fgm_pg": 6.0,
            "fga_pg": 13.0, "ftm_pg": 3.0, "fta_pg": 4.0, "proj_mpg": 30.0}


def make_df(rows):
    """Projection frame carrying turnovers, so the 9-cat path is exercised
    even while the real export set is still 8-cat."""
    df = pd.DataFrame(rows).set_index("name")
    for col, default in DEFAULTS.items():
        df[col] = default if col not in df.columns else df[col].fillna(default)
    return df


def spread(n, **fixed):
    """n filler players, points varied so ranks do not tie."""
    return [{"name": f"P{i}", "pts_pg": 10.0 + 0.5 * i, **fixed} for i in range(n)]


# ------------------------------------------------------------ ratio categories
def test_ratio_impact_scales_with_volume():
    df = make_df([{"name": "HighVol", "fgm_pg": 12.0, "fga_pg": 20.0},   # 60%
                  {"name": "LowVol", "fgm_pg": 3.0, "fga_pg": 5.0},      # 60%
                  {"name": "Filler", "fgm_pg": 5.0, "fga_pg": 13.0}])
    v = V.category_values(df, ["fg"], df.index)
    assert v.loc["HighVol", "fg"] > v.loc["LowVol", "fg"] > 0


def test_shooter_at_league_rate_has_no_impact():
    df = make_df([{"name": f"P{i}", "fgm_pg": 5.0, "fga_pg": 10.0} for i in range(5)])
    assert V.category_values(df, ["fg"], df.index)["fg"].abs().max() == pytest.approx(0.0)


def test_zero_attempts_is_zero_impact_not_nan():
    df = make_df([{"name": "Shooter", "ftm_pg": 4.0, "fta_pg": 5.0},
                  {"name": "NeverShoots", "ftm_pg": 0.0, "fta_pg": 0.0}])
    v = V.category_values(df, ["ft"], df.index)
    assert v.loc["NeverShoots", "ft"] == 0.0
    assert v["ft"].notna().all()


def test_ratio_uses_made_and_attempted_not_a_stored_percentage():
    df = make_df([{"name": "A", "fgm_pg": 6.0, "fga_pg": 10.0},
                  {"name": "B", "fgm_pg": 4.0, "fga_pg": 10.0}])
    df["fg_pct"] = [0.99, 0.01]  # a regressed percentage must not be consulted
    v = V.category_values(df, ["fg"], df.index)
    assert v.loc["A", "fg"] == pytest.approx(1.0)
    assert v.loc["B", "fg"] == pytest.approx(-1.0)


# ------------------------------------------------------------------ turnovers
def test_turnover_z_is_inverted():
    df = make_df([{"name": "Careful", "tov_pg": 1.0},
                  {"name": "Mid", "tov_pg": 3.0},
                  {"name": "Careless", "tov_pg": 5.0}])
    z = V.z_scores(V.category_values(df, ["tov"], df.index), ["tov"], df.index)
    assert z.loc["Careful", "tov"] > z.loc["Mid", "tov"] > z.loc["Careless", "tov"]
    assert z.loc["Careless", "tov"] < 0 < z.loc["Careful", "tov"]


def test_turnovers_drop_out_when_the_projection_set_lacks_them():
    df = make_df([{"name": "A"}, {"name": "B"}])
    assert V.available_categories(df) == NINE_CAT
    assert V.available_categories(df.drop(columns=["tov_pg"])) == EIGHT_CAT


# ----------------------------------------------------------------- punt builds
def test_punting_a_category_removes_it_and_demotes_its_specialist():
    df = make_df(spread(20, blk_pg=0.5) + [{"name": "Blocker", "blk_pg": 4.0, "pts_pg": 15.0}])
    full, _ = V.value_players(df, NINE_CAT, pool_size=len(df))
    punt, _ = V.punt_value(df, ("blk",), NINE_CAT, pool_size=len(df))
    assert "blk" in full.columns and "blk" not in punt.columns
    assert full.index.get_loc("Blocker") < punt.index.get_loc("Blocker")


def test_suggest_builds_ranks_by_fit():
    df = make_df(spread(20, blk_pg=0.5) + [{"name": "Blocker", "blk_pg": 4.0, "pts_pg": 15.0}])
    out = V.suggest_builds(df, "Blocker", 1, pool_size=len(df), top=3)
    assert len(out) == 3
    assert out["value"].is_monotonic_decreasing
    assert out.iloc[0]["punt"] != "blk"  # never punt your own best category


# -------------------------------------------------------------- real data path
def test_real_projection_set_is_eight_cat_until_the_turnover_export_lands():
    df, _ = V.load_projections(PROJECTIONS)
    assert V.available_categories(df) == EIGHT_CAT


def test_every_projected_player_gets_a_value():
    df, _ = V.load_projections(PROJECTIONS)
    z, pool = V.value_players(df)
    assert len(z) == len(df)
    assert z["total"].notna().all()
    assert len(pool) == V.DEFAULT_POOL


def test_converged_pool_reproduces_itself():
    df, _ = V.load_projections(PROJECTIONS)
    cats = V.available_categories(df)
    _, pool = V.value_players(df, cats)
    again = (V.z_scores(V.category_values(df, cats, pool), cats, pool)[list(cats)]
             .sum(axis=1).nlargest(len(pool)).index)
    assert frozenset(again) == frozenset(pool)
