from pathlib import Path

import pandas as pd
import pytest

import puntfit.valuation as V
from puntfit.draft import DraftState, League

PROJECTIONS = Path(__file__).parent / "data" / "marcel_projections.json"


@pytest.fixture(scope="module")
def projections():
    df, _ = V.load_projections(PROJECTIONS)
    return df


@pytest.fixture
def draft(projections):
    return DraftState(projections, League(teams=12, rounds=13))


# ------------------------------------------------------------- snake order
def test_first_round_runs_forwards():
    league = League(teams=12)
    assert [league.slot(p) for p in range(12)] == list(range(12))


def test_second_round_runs_backwards():
    league = League(teams=12)
    assert [league.slot(p) for p in range(12, 24)] == list(reversed(range(12)))


def test_the_turn_gets_back_to_back_picks():
    league = League(teams=12)
    assert league.slot(11) == 11 and league.slot(12) == 11


def test_every_team_gets_one_pick_per_round():
    league = League(teams=10, rounds=13)
    for rnd in range(13):
        picks = [league.slot(p) for p in range(rnd * 10, (rnd + 1) * 10)]
        assert sorted(picks) == list(range(10)), rnd


# ------------------------------------------------------------ making picks
def test_a_pick_goes_to_whoever_is_on_the_clock(draft):
    team = draft.on_the_clock
    pick = draft.make_pick(draft.recommend().index[0])
    assert pick.team == team
    assert draft.roster(team) == [pick.player]


def test_a_drafted_player_leaves_the_pool(draft):
    taken = draft.recommend().index[0]
    draft.make_pick(taken)
    assert taken not in draft.available
    assert taken not in draft.recommend().index


def test_the_same_player_cannot_go_twice(draft):
    taken = draft.recommend().index[0]
    draft.make_pick(taken)
    with pytest.raises(ValueError, match="already drafted"):
        draft.make_pick(taken)


def test_an_unknown_player_is_refused(draft):
    with pytest.raises(ValueError, match="not in the projection set"):
        draft.make_pick("Nobody At All")


def test_the_draft_ends_after_the_last_pick(projections):
    d = DraftState(projections, League(teams=4, rounds=2))
    for _ in range(8):
        assert not d.complete
        d.make_pick(d.recommend().index[0])
    assert d.complete
    with pytest.raises(ValueError, match="draft is over"):
        d.make_pick(d.available[0])


def test_rounds_advance_with_the_picks(projections):
    d = DraftState(projections, League(teams=4, rounds=3))
    assert d.current_round == 1
    for _ in range(4):
        d.make_pick(d.recommend().index[0])
    assert d.current_round == 2


# ------------------------------------------------------------ mock draft
def test_a_full_mock_draft_gives_everyone_a_full_roster(projections):
    league = League(teams=12, rounds=13)
    d = DraftState(projections, league)
    while not d.complete:
        d.make_pick(d.recommend().index[0])
    assert len(d.picks) == 156
    for team in range(12):
        assert len(d.roster(team)) == 13
    assert len({p.player for p in d.picks}) == 156      # nobody drafted twice


# ---------------------------------------------------------------- punting
def test_a_punt_build_changes_who_is_recommended(draft):
    plain = draft.recommend(team=0, limit=20).index
    draft.set_punt(0, ("tov",))
    punted = draft.recommend(team=0, limit=20).index
    assert list(plain) != list(punted)


def test_punting_drops_the_category_from_the_values(draft):
    draft.set_punt(0, ("blk",))
    assert "blk" not in draft.recommend(team=0).columns


def test_each_team_keeps_its_own_build(draft):
    draft.set_punt(0, ("tov",))
    draft.set_punt(1, ("ft",))
    assert list(draft.recommend(team=0).index) != list(draft.recommend(team=1).index)


def test_an_unknown_category_is_refused(draft):
    with pytest.raises(ValueError, match="not categories"):
        draft.set_punt(0, ("dunks",))


def test_suggested_builds_are_ordered_by_fit(draft):
    out = draft.suggest_punts("Luka Doncic", 1, limit=4)
    assert len(out) == 4
    assert out.value.is_monotonic_decreasing
    # the high-usage guard's best single punt is turnovers
    assert out.iloc[0].label == "tov"


def test_giannis_still_wants_to_punt_free_throws(draft):
    out = draft.suggest_punts("Giannis Antetokounmpo", 1, limit=1)
    assert out.iloc[0].label == "ft"
