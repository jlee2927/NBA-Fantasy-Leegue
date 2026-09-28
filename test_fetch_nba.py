from pathlib import Path

import pandas as pd
import pytest

import puntfit.fetch_nba as F

CACHE = Path(__file__).parent / "data" / "hoopr"
SEASONS = Path(__file__).parent / "data" / "nba_seasons.parquet"
needs_cache = pytest.mark.skipif(not SEASONS.exists(),
                                 reason="run `python -m puntfit.fetch_nba` first")

STAT_DEFAULTS = {"points": 20.0, "three_point_field_goals_made": 2.0,
                 "rebounds": 5.0, "assists": 4.0, "steals": 1.0, "blocks": 0.5,
                 "turnovers": 3.0, "field_goals_made": 8.0,
                 "field_goals_attempted": 16.0, "free_throws_made": 2.0,
                 "free_throws_attempted": 3.0}


def games(team_id, athlete_id, n, first_game=0, minutes=30.0, **over):
    """n box score rows for one player on one team."""
    out = []
    for i in range(n):
        row = {"game_id": first_game + i, "season": 2020, "season_type": 2,
               "athlete_id": athlete_id, "athlete_display_name": f"P{athlete_id}",
               "team_id": team_id, "team_abbreviation": f"T{team_id}",
               "athlete_position_abbreviation": "PG", "minutes": minutes,
               "did_not_play": minutes is None, **STAT_DEFAULTS, **over}
        out.append(row)
    return out


def test_all_star_game_is_not_counted_as_a_club_game():
    # the All-Star squad plays one game; the club plays 60
    rows = (games(team_id=1, athlete_id=10, n=60)
            + games(team_id=99, athlete_id=10, n=1, first_game=900, points=40.0))
    out = F.season_totals(pd.DataFrame(rows), 2020)
    assert out.loc[10, "g"] == 60
    assert out.loc[10, "pts"] == 60 * 20.0          # the 40-point night is excluded
    assert out.loc[10, "team_games"] == 60


def test_did_not_play_rows_are_excluded():
    rows = games(1, 10, 55) + games(1, 10, 5, first_game=55, minutes=None)
    out = F.season_totals(pd.DataFrame(rows), 2020)
    assert out.loc[10, "g"] == 55
    assert out.loc[10, "team_games"] == 60          # the team still played 60


def test_games_share_normalises_a_shortened_season():
    full = F.season_totals(pd.DataFrame(games(1, 10, 82)), 2019)
    short = F.season_totals(pd.DataFrame(games(1, 10, 66)), 2012)
    assert full.loc[10, "g_share"] == pytest.approx(1.0)
    assert short.loc[10, "g_share"] == pytest.approx(1.0)  # every game of a 66-game year


def test_stat_columns_are_renamed_to_the_projection_contract():
    out = F.season_totals(pd.DataFrame(games(1, 10, 60)), 2020)
    for col in ("pts", "tpm", "reb", "ast", "stl", "blk", "tov",
                "fgm", "fga", "ftm", "fta", "min", "g", "g_share", "season"):
        assert col in out.columns, col
    assert out.loc[10, "tov"] == 60 * 3.0


# ------------------------------------------------------------- against real data
@needs_cache
def test_season_label_is_the_ending_year():
    box = F._read_parquet(CACHE / "player_box" / "player_box_2026.parquet")
    assert box.game_date.min().year == 2025 and box.game_date.max().year == 2026


@needs_cache
def test_known_season_lines_match_the_record_book():
    d = pd.read_parquet(SEASONS)
    curry = d[(d.name == "Stephen Curry") & (d.season == 2016)].iloc[0]
    assert curry.g == 79                              # not 80: the All-Star game
    assert curry.tpm == 402                           # the single-season record
    assert curry.pts / curry.g == pytest.approx(30.1, abs=0.05)


@needs_cache
def test_shortened_seasons_are_recorded_at_their_real_length():
    d = pd.read_parquet(SEASONS)
    lengths = d.groupby("season").team_games.median()
    assert lengths[2012] == 66     # lockout
    assert lengths[2020] == 72     # covid cutoff
    assert lengths[2021] == 72
    assert lengths[2019] == 82


ROSTER = CACHE / "crosswalk" / "nba_player_crosswalk_2027.parquet"
needs_roster = pytest.mark.skipif(not ROSTER.exists(), reason="roster file not cached")


@needs_roster
def test_the_roster_is_keyed_on_the_same_athlete_id():
    roster = F.current_roster(CACHE, 2027)
    assert roster.index.name == "athlete_id"
    assert not roster.index.duplicated().any()
    assert roster.team.notna().all() and roster.pos.notna().all()


@needs_roster
def test_players_who_missed_a_whole_season_are_still_rostered():
    """The bug this filter replaced: "played last season" deleted Lillard,
    Irving and Haliburton, who each sat out 2025-26 injured and are all
    drafted in 2026-27."""
    roster = F.current_roster(CACHE, 2027)
    seasons = pd.read_parquet(SEASONS)
    played_last = set(seasons[seasons.season == 2026].athlete_id)
    for who in ("Damian Lillard", "Kyrie Irving", "Tyrese Haliburton"):
        athlete = int(seasons[seasons.name == who].athlete_id.iloc[0])
        assert athlete not in played_last, who      # did not play in 2025-26
        assert athlete in roster.index, who         # but is on a 2026-27 roster


@needs_cache
@needs_roster
def test_the_draft_board_carries_those_players():
    import json
    board = json.loads((Path(__file__).parent / "data" /
                        "marcel_projections.json").read_text())
    assert board["params"]["roster_filtered"] is True
    names = {p["name"] for p in board["players"]}
    for who in ("Damian Lillard", "Kyrie Irving", "Tyrese Haliburton"):
        assert who in names, who
    assert all(p["team"] and p["pos"] for p in board["players"])


@needs_cache
def test_every_player_season_has_an_age_and_turnovers():
    d = pd.read_parquet(SEASONS)
    rated = d[d["min"] >= 500]
    assert rated.age.notna().all()
    assert rated.tov.notna().all()
    assert rated.age.between(17, 45).all()
