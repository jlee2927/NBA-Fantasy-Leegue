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
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .categories import NINE_CAT
from .valuation import DEFAULT_POOL, punt_value, value_players

# How far from parity a category has to be before it stops being worth
# chasing, measured in standard deviations of a team's season total.
CONTEST_WIDTH = 1.0

# A manager's own emphasis multiplies the automatic weighting. 1.0 leaves it
# alone, 0 ignores the category, and the cap stops one category swamping the
# board the way an unbounded multiplier would.
NEUTRAL_EMPHASIS = 1.0
MAX_EMPHASIS = 2.0


@dataclass(frozen=True)
class League:
    teams: int = 12
    rounds: int = 13
    categories: tuple[str, ...] = NINE_CAT

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    @property
    def max_punts(self) -> int:
        """Punting more than this cannot win a week.

        A week goes to whoever takes a majority of the categories, so a roster
        has to keep more than half of them. Punt five of nine and the best
        available result is four, which loses every week by construction.
        """
        return math.ceil(len(self.categories) / 2) - 1

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
    emphasis: dict[int, dict[str, float]] = field(default_factory=dict)
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

    def category_weights(self, team: int) -> pd.Series:
        """How much a marginal player could still change each category.

        A category league is won by taking a majority of categories each week,
        not by piling up totals: winning blocks 60-12 scores exactly what
        winning 31-30 does. So the value of adding to a category is the chance
        it flips the result, which is highest where the team sits near parity.
        A category already lost is not worth buying into, and one already won
        is not worth padding. Both tail off; the contested middle does not.

        Weighting by raw strength instead - chase whatever the roster is best
        at - gets the lost half right and the won half exactly backwards, and
        keeps stacking a category the team locked up rounds ago.

        Values are standardised over the draftable pool, so an average team's
        season total is zero in every category and the spread of team totals is
        about the square root of the roster size. Unfilled slots are treated as
        league-average, which is why an almost-empty roster sits near parity
        everywhere and the board stays close to plain best-available until the
        team has taken some shape.
        """
        z = self.values(self.punts.get(team, ()))
        cats = [c for c in z.columns if c != "total"]
        roster = self.roster(team)
        if not roster:
            return pd.Series(1.0, index=cats)

        totals = z.loc[z.index.intersection(roster, sort=False), cats].sum()
        standing = totals / math.sqrt(self.league.rounds)
        return np.exp(-0.5 * (standing / CONTEST_WIDTH) ** 2)

    def weights(self, team: int) -> pd.Series:
        """The automatic weighting with the user's own emphasis applied.

        Emphasis multiplies rather than replaces, so a category a manager has
        dialled up still tapers off once it is safely won. Replacing the
        automatic value outright would turn "I want more steals" into "chase
        steals forever", which is the behaviour this weighting exists to avoid.
        """
        auto = self.category_weights(team)
        manual = self.emphasis.get(team)
        if not manual:
            return auto
        return auto * pd.Series({c: manual.get(c, NEUTRAL_EMPHASIS)
                                 for c in auto.index})

    def set_emphasis(self, team: int, category: str, value: float) -> None:
        if category not in self.league.categories:
            raise ValueError(f"not a category in this league: {category}")
        value = float(np.clip(value, 0.0, MAX_EMPHASIS))
        self.emphasis.setdefault(team, {})[category] = value

    def clear_emphasis(self, team: int) -> None:
        self.emphasis.pop(team, None)

    def recommend(self, team: int | None = None, limit: int = 10,
                  contest: bool = True) -> pd.DataFrame:
        """Best available under that team's build, best first.

        With `contest` on, the ranking leans toward the categories still in
        play for this roster, so the board shifts as the team takes shape
        instead of serving the same list every round.
        """
        team = self.on_the_clock if team is None else team
        z = self.values(self.punts.get(team, ()))
        board = z.loc[z.index.intersection(self.available, sort=False)]

        if contest and self.roster(team):
            w = self.weights(team)
            board = board.assign(total=(board[w.index] * w).sum(axis=1) / w.sum())
            board = board.sort_values("total", ascending=False)
        return board.head(limit)

    def auto_punt(self, player: str, n: int) -> tuple[str, ...]:
        """The n categories this player is worst at.

        The direct reading of "punt what you are bad at", which is what people
        mean by a punt build and what they can check against their own eyes.
        Ranking every possible build by value gives a similar answer but an
        occasionally surprising one, so that stays available as an alternative
        rather than as the default.
        """
        z = self.values(())
        cats = [c for c in self.league.categories if c in z.columns]
        return tuple(z.loc[player, cats].nsmallest(n).index)

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
