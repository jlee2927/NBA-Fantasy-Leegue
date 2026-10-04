from pathlib import Path

import pandas as pd
import pytest

import puntfit.valuation as V
from puntfit.draft import DraftState, League, roster_slots

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


# ------------------------------------------------------------- punt limits
def test_you_must_keep_enough_categories_to_win_a_week():
    """A week goes to whoever takes a majority, so punting five of nine caps
    you at four and loses every week by construction."""
    from puntfit.categories import EIGHT_CAT, NINE_CAT
    assert League(categories=NINE_CAT).max_punts == 4
    assert League(categories=EIGHT_CAT).max_punts == 3
    for cats in (NINE_CAT, EIGHT_CAT):
        league = League(categories=cats)
        kept = len(cats) - league.max_punts
        assert kept > len(cats) / 2


# --------------------------------------------------------------- auto punt
def test_auto_punt_takes_the_players_worst_categories(draft):
    z = draft.values(())
    for n in (1, 2, 4):
        punted = draft.auto_punt("Luka Doncic", n)
        assert len(punted) == n
        kept = [c for c in draft.league.categories if c not in punted]
        assert z.loc["Luka Doncic", list(punted)].max() <= z.loc["Luka Doncic", kept].min()


def test_auto_punt_gives_giannis_free_throws(draft):
    assert draft.auto_punt("Giannis Antetokounmpo", 1) == ("ft",)


def test_auto_punt_nests_as_it_grows(draft):
    one = draft.auto_punt("Luka Doncic", 1)
    assert set(one) < set(draft.auto_punt("Luka Doncic", 3))


# ------------------------------------------------- contested categories
@pytest.fixture
def blocks_heavy(projections):
    """A roster that has locked up blocks and nothing else."""
    import puntfit.valuation as V
    d = DraftState(projections, League(teams=1, rounds=13))
    z, _ = V.value_players(projections)
    for name in z.nlargest(5, "blk").index:
        d.make_pick(name)
    return d


def test_a_locked_category_stops_being_chased(blocks_heavy):
    w = blocks_heavy.category_weights(0)
    assert w["blk"] < 0.05
    assert w["blk"] == w.min()


def test_contested_categories_carry_the_most_weight(blocks_heavy):
    z = blocks_heavy.values(())
    cats = [c for c in z.columns if c != "total"]
    totals = z.loc[z.index.intersection(blocks_heavy.roster(0), sort=False), cats].sum()
    w = blocks_heavy.category_weights(0)
    nearest = totals.abs().idxmin()
    assert w[nearest] == pytest.approx(w.max(), abs=0.05)


def test_a_category_far_behind_is_still_worth_more_than_one_locked(blocks_heavy):
    """Behind is recoverable; already won is not worth padding."""
    w = blocks_heavy.category_weights(0)
    assert w["ast"] > w["blk"]


def test_weighting_reorders_the_board(blocks_heavy):
    plain = list(blocks_heavy.recommend(0, limit=6, contest=False).index)
    smart = list(blocks_heavy.recommend(0, limit=6, contest=True).index)
    assert plain != smart


def test_emphasis_scales_the_automatic_weight(blocks_heavy):
    auto = blocks_heavy.category_weights(0)
    blocks_heavy.set_emphasis(0, "stl", 2.0)
    w = blocks_heavy.weights(0)
    assert w["stl"] == pytest.approx(auto["stl"] * 2.0)
    assert w["ast"] == pytest.approx(auto["ast"])      # untouched


def test_zero_emphasis_takes_a_category_out_of_the_ranking(blocks_heavy):
    blocks_heavy.set_emphasis(0, "ast", 0.0)
    assert blocks_heavy.weights(0)["ast"] == 0.0


def test_emphasis_multiplies_rather_than_replaces(blocks_heavy):
    """Dialling a category up must not stop it tapering once it is won, or
    'I want more steals' becomes 'chase steals forever'."""
    blocks_heavy.set_emphasis(0, "blk", 2.0)
    w = blocks_heavy.weights(0)
    assert w["blk"] < 0.1          # still near zero: blocks are long since won
    assert w["blk"] < w["ast"]


def test_emphasis_is_capped(blocks_heavy):
    from puntfit.draft import MAX_EMPHASIS
    blocks_heavy.set_emphasis(0, "stl", 99.0)
    assert blocks_heavy.emphasis[0]["stl"] == MAX_EMPHASIS
    blocks_heavy.set_emphasis(0, "stl", -5.0)
    assert blocks_heavy.emphasis[0]["stl"] == 0.0


def test_an_unknown_category_cannot_be_emphasised(draft):
    with pytest.raises(ValueError, match="not a category"):
        draft.set_emphasis(0, "dunks", 1.5)


def test_clearing_emphasis_restores_the_automatic_weights(blocks_heavy):
    auto = blocks_heavy.category_weights(0).copy()
    blocks_heavy.set_emphasis(0, "stl", 0.0)
    assert blocks_heavy.weights(0)["stl"] != auto["stl"]
    blocks_heavy.clear_emphasis(0)
    pd.testing.assert_series_equal(blocks_heavy.weights(0), auto)


def test_each_team_keeps_its_own_emphasis(draft):
    draft.make_pick(draft.recommend().index[0])
    draft.set_emphasis(0, "stl", 2.0)
    assert 1 not in draft.emphasis


def test_emphasis_reorders_the_board(blocks_heavy):
    before = list(blocks_heavy.recommend(0, limit=8).index)
    blocks_heavy.set_emphasis(0, "stl", 2.0)
    blocks_heavy.set_emphasis(0, "blk", 0.0)
    assert list(blocks_heavy.recommend(0, limit=8).index) != before


def test_an_empty_roster_has_nothing_to_lean_on(draft):
    w = draft.category_weights(0)
    assert (w == 1.0).all()
    assert list(draft.recommend(0, limit=5, contest=True).index) == \
           list(draft.recommend(0, limit=5, contest=False).index)


# ------------------------------------------------------------ roster shape
def test_an_empty_roster_needs_every_position():
    slots = roster_slots([], 13)
    assert slots["short"] == 8
    assert slots["flex_total"] == 5


def test_a_fourth_guard_stops_counting_toward_the_guard_requirement():
    """Five guards is a legal roster and a bad one. The template has to say
    so: the extras are playable, they just stop filling the G requirement."""
    slots = roster_slots(["G"] * 5, 13)
    guards = next(r for r in slots["rows"] if r["pos"] == "G")
    assert guards["filled"] == guards["need"] == 3
    assert slots["flex_filled"] == 2
    assert slots["short"] == 5          # still owes 3 F and 2 C


def test_a_short_draft_scales_the_minimums_down():
    """A five-round league cannot be told it is eight players short."""
    slots = roster_slots([], 5)
    assert sum(r["need"] for r in slots["rows"]) <= 5
    assert all(r["need"] >= 1 for r in slots["rows"])


def test_an_unknown_position_counts_as_flex():
    slots = roster_slots([None, "PF"], 13)
    assert slots["flex_filled"] == 2
