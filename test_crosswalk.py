from pathlib import Path

import pandas as pd
import pytest

from puntfit.crosswalk import normalize_name

CROSSWALK = Path(__file__).parent / "data" / "crosswalk.csv"
needs_crosswalk = pytest.mark.skipif(not CROSSWALK.exists(),
                                     reason="run `python -m puntfit.crosswalk` first")


def test_normalize_strips_accents_punctuation_and_suffixes():
    assert normalize_name("Nikola Jokić") == "nikola jokic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("Shaquille O'Neal") == "shaquille oneal"
    assert normalize_name("Karl-Anthony Towns") == "karlanthony towns"
    assert normalize_name("  LeBron   JAMES ") == "lebron james"


def test_normalize_keeps_genuinely_different_names_apart():
    assert normalize_name("Marcus Williams") != normalize_name("Marvin Williams")


@needs_crosswalk
def test_same_name_different_people_stay_separate():
    """Two Marcus Williamses overlapped in the league; matching on name alone
    merged a 62-game season with a 9-game one."""
    cw = pd.read_csv(CROSSWALK)
    both = cw[cw.name == "Marcus Williams"]
    assert len(both) == 2
    assert both.dob.nunique() == 2
    assert both.espn_athlete_id.nunique() == 2
    assert both.nba_person_id.nunique() == 2


@needs_crosswalk
def test_no_id_appears_twice():
    cw = pd.read_csv(CROSSWALK)
    assert not cw.espn_athlete_id.duplicated().any()
    assert not cw.nba_person_id.duplicated().any()


@needs_crosswalk
def test_carries_draft_position_for_rookie_minutes():
    cw = pd.read_csv(CROSSWALK)
    assert cw.draftNumber.notna().mean() > 0.95
