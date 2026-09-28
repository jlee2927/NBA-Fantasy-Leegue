"""Simulated managers for the mock draft.

They exist to make a mock feel like a real room. If every seat took the best
available player the draft would run in ranking order, the user would learn
nothing about how a board actually falls, and every mock would be identical.

Three behaviours, assigned at draft setup:

  best_available  takes the top of the board. Roughly how a first-timer drafts.
  punt_build      commits to a build in the early rounds and then drafts to it,
                  which is what makes categories run dry unevenly.
  category_need   chases whatever its roster is currently worst at, which is
                  how a lot of people actually draft and why runs happen.

All three pick through DraftState.make_pick like a person does.
"""
from __future__ import annotations

import random

import pandas as pd

BEST_AVAILABLE = "best_available"
PUNT_BUILD = "punt_build"
CATEGORY_NEED = "category_need"
STRATEGIES = (BEST_AVAILABLE, PUNT_BUILD, CATEGORY_NEED)

# How far down the board a manager will reach. Nobody drafts the consensus
# board exactly, and a little noise is what stops every mock being identical.
REACH = 3


def assign(teams: int, user_slot: int, seed: int | None = None) -> dict[int, str]:
    """A strategy per seat. Roughly half chase a build, the rest split."""
    rng = random.Random(seed)
    out = {}
    for team in range(teams):
        if team == user_slot:
            continue
        out[team] = rng.choices(STRATEGIES, weights=[3, 4, 3])[0]
    return out


def _commit_to_a_build(state, team, rng) -> None:
    """After a couple of picks, adopt whichever build the roster already fits."""
    roster = state.roster(team)
    if not roster or team in state.punts:
        return
    if len(roster) < 2:
        return
    fits = state.suggest_punts(roster[0], 1, limit=3)
    state.set_punt(team, tuple(fits.iloc[rng.randrange(len(fits))].punt))


def _weakest_categories(state, team, n=2) -> tuple[str, ...]:
    roster = state.roster(team)
    if not roster:
        return ()
    z = state.values(state.punts.get(team, ()))
    cats = [c for c in z.columns if c != "total"]
    totals = z.loc[z.index.intersection(roster, sort=False), cats].sum()
    return tuple(totals.nsmallest(n).index)


def choose(state, team: int, strategy: str, rng: random.Random) -> str:
    """The player this manager takes, given the board as it stands."""
    if strategy == PUNT_BUILD:
        _commit_to_a_build(state, team, rng)

    board = state.recommend(team=team, limit=REACH * 4)
    if board.empty:
        return state.available[0]

    if strategy == CATEGORY_NEED:
        need = _weakest_categories(state, team)
        if need:
            # among the players in reach, take whoever helps the gaps most
            reach = board.head(REACH * 3)
            return reach[list(need)].sum(axis=1).idxmax()

    return board.index[min(rng.randrange(REACH), len(board) - 1)]


def autopick(state, strategies: dict[int, str], rng: random.Random) -> list:
    """Advance the draft until a seat with no strategy is on the clock."""
    made = []
    while not state.complete and state.on_the_clock in strategies:
        team = state.on_the_clock
        made.append(state.make_pick(choose(state, team, strategies[team], rng)))
    return made
