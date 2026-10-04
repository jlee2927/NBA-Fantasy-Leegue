import pandas as pd
import pytest

from puntfit.positions import (ALL, assign, derive, disagreements,
                               roster_view, slots_for)


def test_the_template_matches_a_standard_category_league():
    slots = slots_for(13)
    assert slots[:7] == ["PG", "SG", "SF", "PF", "C", "G", "F"]
    assert slots.count("UTIL") == 3
    assert slots.count("BE") == 3


def test_a_short_draft_keeps_the_floor_spread():
    """Six roster spots should still cover the court rather than being
    three guards and a bench."""
    slots = slots_for(6)
    assert slots == ["PG", "SG", "SF", "PF", "C", "G"]
    assert "BE" not in slots


def test_a_long_draft_adds_bench_not_starters():
    slots = slots_for(16)
    assert slots.count("BE") == 6
    assert slots.count("UTIL") == 3


def test_a_guard_can_fill_the_flex_guard_slot():
    filled = assign([("A", ("SG",))], ["PG", "G"])
    assert filled[1] == 0            # no PG claim, lands in G


def test_matching_reshuffles_rather_than_blocking():
    """Greedy would put the dual-eligible guard in PG and leave the pure
    point guard homeless. Maximum matching moves the first one instead."""
    players = [("Combo", ("PG", "SG")), ("Pure", ("PG",))]
    filled = assign(players, ["PG", "SG"])
    assert sorted(f for f in filled if f is not None) == [0, 1]


def test_a_player_with_nowhere_to_start_is_reported():
    pos = pd.Series({"A": ("C",), "B": ("C",), "C": ("C",)})
    view = roster_view(["A", "B", "C"], pos, 5)   # one C slot in a 5-man roster
    assert view["unplaced"] == 2


def test_eligibility_counts_every_slot_a_player_can_fill():
    pos = pd.Series({"A": ("PG", "SG")})
    view = roster_view(["A"], pos, 13)
    assert view["counts"]["PG"] == 1 and view["counts"]["SG"] == 1


def test_an_unknown_player_is_eligible_everywhere():
    """Better to let a missing position fill any slot than to strand a real
    player over a gap in the data."""
    view = roster_view(["Nobody"], pd.Series(dtype=object), 13)
    assert view["unplaced"] == 0


def test_derive_keeps_espns_coarse_call(projections):
    """The split is inferred; the group is not. No centre becomes a guard."""
    pos = derive(projections)
    centres = projections[projections.pos == "C"].index
    assert all(pos[n] == ("C",) for n in centres)
    guards = projections[projections.pos == "G"].index
    assert all(set(pos[n]) <= {"PG", "SG"} for n in guards)


def test_derive_gives_some_players_dual_eligibility(projections):
    pos = derive(projections)
    assert sum(1 for p in pos if len(p) == 2) > 0
    assert all(set(p) <= set(ALL) for p in pos)


def test_disagreements_flag_a_misfiled_big(projections):
    """ESPN files Wembanyama under F, so the derivation can only ever reach
    PF. The review list is the one place that can say otherwise."""
    pos = derive(projections)
    ranks = pd.Series(range(1, len(projections) + 1), index=projections.index)
    rows = disagreements(projections, pos, ranks)
    flagged = {r["player"]: r for r in rows}
    assert "Victor Wembanyama" in flagged
    assert flagged["Victor Wembanyama"]["suggests"] == "C"


def test_disagreements_come_back_in_draft_order(projections):
    """A wrong position in the first round matters; one in the three
    hundredth does not."""
    pos = derive(projections)
    ranks = pd.Series(range(1, len(projections) + 1), index=projections.index)
    rows = disagreements(projections, pos, ranks)
    assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)


def test_disagreements_decide_nothing(projections):
    """The derived position is reported alongside the suggestion, never
    replaced by it - the league's platform is the authority."""
    pos = derive(projections)
    ranks = pd.Series(range(1, len(projections) + 1), index=projections.index)
    for r in disagreements(projections, pos, ranks):
        assert r["derived"] == "/".join(pos[r["player"]])
