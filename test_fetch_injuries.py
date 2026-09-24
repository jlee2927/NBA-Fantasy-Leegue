from pathlib import Path

import pandas as pd
import pytest

import puntfit.fetch_injuries as I

PAYLOAD = {"injuries": [{"injuries": [
    {"status": "Out", "date": "2026-09-21T19:50Z",
     "shortComment": "torn ACL, out for the season",
     "details": {"type": "Knee", "detail": "Surgery", "returnDate": "2027-07-01"},
     "athlete": {"displayName": "Henri Veesaar",
                 "team": {"abbreviation": "ATL"},
                 "links": [{"href": "https://www.espn.com/nba/player/_/id/5105571/henri-veesaar"}]}},
    {"status": "Day-To-Day", "date": "2026-09-20T12:00Z", "shortComment": "sore ankle",
     "details": {"type": "Ankle", "detail": "Sprain", "returnDate": "2026-10-01"},
     "athlete": {"displayName": "PJ Hall", "team": {"abbreviation": "DEN"},
                 "links": [{"href": "https://www.espn.com/nba/player/_/id/4433134/pj-hall"}]}},
]}]}


def test_athlete_id_is_read_from_the_playercard_link():
    """ESPN leaves the id field empty; it only appears inside the link."""
    df = I.parse(PAYLOAD)
    assert df.athlete_id.tolist() == [5105571, 4433134]
    assert df.athlete_id.dtype == "int64"


def test_parse_pulls_status_and_return_date():
    df = I.parse(PAYLOAD).set_index("name")
    assert df.loc["Henri Veesaar", "status"] == "Out"
    assert df.loc["Henri Veesaar", "return_date"] == "2027-07-01"
    assert df.loc["PJ Hall", "body_part"] == "Ankle"
    assert df.loc["PJ Hall", "team"] == "DEN"


def test_out_is_more_severe_than_day_to_day():
    df = I.parse(PAYLOAD).set_index("name")
    assert df.loc["Henri Veesaar", "severity"] > df.loc["PJ Hall", "severity"]


def test_entries_without_a_resolvable_id_are_dropped():
    payload = {"injuries": [{"injuries": [
        {"status": "Out", "details": {}, "athlete": {"displayName": "Nobody", "links": []}}]}]}
    assert len(I.parse(payload)) == 0


def test_manual_overrides_beat_the_feed(tmp_path):
    feed = I.parse(PAYLOAD)
    override = tmp_path / "manual.csv"
    pd.DataFrame([{"athlete_id": 4433134, "name": "PJ Hall", "team": "DEN",
                   "status": "Out", "body_part": "Ankle", "detail": "Fracture",
                   "return_date": "2027-01-01", "reported": "2026-09-24",
                   "comment": "beat writer says fractured"}]).to_csv(override, index=False)

    merged = I.apply_overrides(feed, override).set_index("athlete_id")
    assert merged.loc[4433134, "status"] == "Out"
    assert merged.loc[4433134, "source"] == "manual"
    assert merged.loc[4433134, "severity"] == I.SEVERITY["Out"]
    assert merged.loc[5105571, "source"] == "espn"     # untouched
    assert len(merged) == 2                            # replaced, not appended


def test_missing_override_file_is_fine(tmp_path):
    feed = I.parse(PAYLOAD)
    assert I.apply_overrides(feed, tmp_path / "absent.csv").equals(feed)


def test_labelling_never_changes_a_projection():
    """The whole point of this being a separate layer: a live third-party feed
    must not be able to reorder a draft board."""
    proj = pd.DataFrame({"player": [5105571, 4433134, 999],
                         "pts_pg": [20.0, 10.0, 15.0],
                         "proj_mpg": [30.0, 20.0, 25.0]}).set_index("player", drop=False)
    out = I.label(proj, I.parse(PAYLOAD))
    pd.testing.assert_frame_equal(out[proj.columns], proj)
    assert out.loc[5105571, "injury_status"] == "Out"
    assert pd.isna(out.loc[999, "injury_status"])
    assert out.loc[999, "injury_severity"] == 0
