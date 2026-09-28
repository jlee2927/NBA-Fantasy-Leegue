from pathlib import Path

import numpy as np
import pytest

import puntfit.backtest as B

CONFIG = Path(__file__).parent / "config.toml"
TARGETS = list(range(2005, 2027))


def test_config_loads_every_window_from_the_file():
    cfg = B.Config.load(CONFIG)
    assert cfg.holdout == 2026
    assert cfg.regime_window == 5
    assert cfg.structural_window == 0        # 0 means every prior season
    assert cfg.early_through < cfg.late_from


def test_a_target_never_learns_from_itself_or_later():
    """The no-leakage guarantee."""
    for target in TARGETS:
        window = B.tuning_window(TARGETS, target, holdout=2026)
        assert all(t < target for t in window), target


def test_the_holdout_is_excluded_from_every_window():
    for target in TARGETS:
        assert 2026 not in B.tuning_window(TARGETS, target, holdout=2026)


def test_an_earlier_holdout_is_also_excluded():
    window = B.tuning_window(TARGETS, 2020, holdout=2015)
    assert 2015 not in window
    assert 2014 in window and 2016 in window


def test_the_earliest_target_has_nothing_to_learn_from():
    assert B.tuning_window(TARGETS, 2005, holdout=2026) == []


def test_flat_weights_when_no_half_life_is_set():
    w = B.weights_for([2020, 2021, 2022], 2023, halflife=0)
    assert (w == 1.0).all()


def test_recency_weighting_decays_by_half_life():
    w = B.weights_for([2019, 2021, 2023], 2023, halflife=2)
    assert w[2] == pytest.approx(1.0)      # the target's own prior season
    assert w[1] == pytest.approx(0.5)      # two seasons back
    assert w[0] == pytest.approx(0.25)     # four seasons back
    assert (np.diff(w) > 0).all()          # older seasons always count less


def test_season_error_is_scale_free():
    """Categories on different units must contribute equally, so each is
    divided by its own mean before averaging."""
    rows = [{"marcel": 2.0, "last_season": 4.0, "mean_per36": 20.0},
            {"marcel": 0.1, "last_season": 0.2, "mean_per36": 1.0}]
    assert B.season_error(rows) == pytest.approx((0.1 + 0.1) / 2)
    assert B.naive_error(rows) == pytest.approx((0.2 + 0.2) / 2)
