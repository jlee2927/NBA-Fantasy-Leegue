"""The draft state machine.

One state machine, three drivers. A solo mock draft, the live assistant with
manually entered picks, and a shared league room all advance the same object
through make_pick(). Simulated managers and real managers take the same path;
if either needed its own, the design would be wrong.

Player values are computed once per punt build and then filtered to whoever is
still available, rather than restandardised after every pick. Recomputing
would let the exchange rate between categories drift as the pool drains, so a
player's value would change because other people got drafted rather than
because anything about him did. It is also what keeps a recommendation cheap.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from .categories import NINE_CAT
from .valuation import DEFAULT_POOL, punt_value, value_players


@dataclass(frozen=True)
class League:
    teams: int = 12
    rounds: int = 13
    categories: tuple[str, ...] = NINE_CAT

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    def slot(self, pick: int) -> int:
        """Whose turn it is at a zero-based overall pick, snake order."""
        rnd, within = divmod(pick, self.teams)
        return within if rnd % 2 == 0 else self.teams - 1 - within


@dataclass(frozen=True)
class Pick:
    number: int          # zero-based, overall
    team: int
    player: str


@dataclass
class DraftState:
    projections: pd.DataFrame
    league: League = field(default_factory=League)
    pool_size: int = DEFAULT_POOL
    picks: list[Pick] = field(default_factory=list)
    punts: dict[int, tuple[str, ...]] = field(default_factory=dict)
    _values: dict[tuple[str, ...], pd.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------- position
    @property
    def complete(self) -> bool:
        return len(self.picks) >= self.league.total_picks

    @property
    def on_the_clock(self) -> int:
        return self.league.slot(len(self.picks))

    @property
    def current_round(self) -> int:
        return len(self.picks) // self.league.teams + 1

    @property
    def drafted(self) -> set[str]:
        return {p.player for p in self.picks}

    @property
    def available(self) -> pd.Index:
        return self.projections.index.difference(self.drafted, sort=False)

    def roster(self, team: int) -> list[str]:
        return [p.player for p in self.picks if p.team == team]

    # --------------------------------------------------------------- values
    def values(self, punt: tuple[str, ...] = ()) -> pd.DataFrame:
        """Z-scores under a punt build, cached because every pick reuses them."""
        key = tuple(sorted(punt))
        if key not in self._values:
            if key:
                z, _ = punt_value(self.projections, key, self.league.categories,
                                  self.pool_size)
            else:
                z, _ = value_players(self.projections, self.league.categories,
                                     self.pool_size)
            self._values[key] = z
        return self._values[key]

    # ----------------------------------------------------------- the action
    def make_pick(self, player: str) -> Pick:
        if self.complete:
            raise ValueError("draft is over")
        if player in self.drafted:
            raise ValueError(f"{player} is already drafted")
        if player not in self.projections.index:
            raise ValueError(f"{player} is not in the projection set")
        pick = Pick(number=len(self.picks), team=self.on_the_clock, player=player)
        self.picks.append(pick)
        return pick

    def recommend(self, team: int | None = None, limit: int = 10) -> pd.DataFrame:
        """Best available under that team's build, best first."""
        team = self.on_the_clock if team is None else team
        z = self.values(self.punts.get(team, ()))
        free = z.index.intersection(self.available, sort=False)
        return z.loc[free].head(limit)

    # ----------------------------------------------------------- punt fits
    def suggest_punts(self, player: str, n_punts: int, limit: int = 5) -> pd.DataFrame:
        """Which builds of this size suit a player, best fit first.

        Builds with the same punt count keep the same number of categories, so
        their values compare directly.
        """
        rows = []
        for punt in itertools.combinations(self.league.categories, n_punts):
            z = self.values(punt)
            rows.append({"punt": punt, "label": "+".join(punt),
                         "value": z.loc[player, "total"],
                         "rank": z.index.get_loc(player) + 1})
        return (pd.DataFrame(rows).sort_values("value", ascending=False)
                .head(limit).reset_index(drop=True))

    def set_punt(self, team: int, punt: tuple[str, ...]) -> None:
        unknown = set(punt) - set(self.league.categories)
        if unknown:
            raise ValueError(f"not categories in this league: {sorted(unknown)}")
        self.punts[team] = tuple(punt)
