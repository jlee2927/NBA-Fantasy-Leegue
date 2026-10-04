"""Shared-room logic, tested against an in-memory stand-in for Postgres.

The network these tests run on blocks outbound 5432, so the database itself
is unreachable here. What is checked is everything above it: that a room
rebuilds from its rows, that a claimed seat is never autodrafted, that a
pick out of turn is refused, and that losing a race surfaces as a message
rather than as somebody else's player.
"""
import pytest

import puntfit.app as A
from puntfit.app import app as flask_app


class FakeDB:
    """The rows a room would have, with the same shape shared.load returns."""

    def __init__(self, teams=4, rounds=3, picks=(), tokens=()):
        self.teams, self.rounds = teams, rounds
        self.picks = list(picks)
        self.seats = {s: {"token": None, "name": None, "punts": (),
                          "emphasis": {}, "view": "value"} for s in range(teams)}
        for seat, token in tokens:
            self.seats[seat]["token"] = token
        self.rejected = False

    def load(self, room_id):
        return {"teams": self.teams, "rounds": self.rounds, "format": "9cat",
                "seed": 7, "fingerprint": A._fingerprint(), "host_token": "host",
                "picks": list(self.picks), "seats": self.seats}

    def add_pick(self, room_id, number, seat, player):
        if self.rejected:
            return False
        self.picks.append(player)
        return True


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(A.store, "configured", lambda: True)
    monkeypatch.setattr(A.shared, "load", fake.load)
    monkeypatch.setattr(A.shared, "add_pick", fake.add_pick)
    monkeypatch.setattr(A.shared, "claim", lambda rid, tok, name=None: 0)
    return fake


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_an_invite_link_without_a_seat_offers_one(client, db):
    page = client.get("/draft/sharedroom").data.decode()
    assert "You have been invited to a draft" in page
    assert "Take a seat" in page


def test_a_seated_viewer_gets_the_board(client, db):
    db.seats[1]["token"] = "mine"
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    page = client.get("/draft/sharedroom").data.decode()
    assert "Best available" in page
    assert "Invite the rest of your league" in page


def test_a_claimed_seat_is_never_autodrafted(client, db):
    """The worst failure this feature could have is drafting for a person
    who is sitting there deciding."""
    db.seats[1]["token"] = "mine"
    db.seats[2]["token"] = "theirs"
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    with flask_app.test_request_context(
            "/draft/sharedroom", headers={"Cookie": "puntfit_seat_sharedroom=mine"}):
        room = A._shared("sharedroom")
    assert 1 not in room["strategies"]      # me
    assert 2 not in room["strategies"]      # the other human
    assert 0 in room["strategies"] and 3 in room["strategies"]


def test_picking_out_of_turn_is_refused(client, db):
    """Seat 0 is on the clock at the start, so seat 2 may not pick."""
    db.seats[2]["token"] = "mine"
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    r = client.post("/draft/sharedroom/pick", data={"player": "Nikola Jokic"})
    assert r.status_code == 302 and "not+your+pick" in r.headers["Location"]
    assert db.picks == []


def test_losing_a_race_says_so_rather_than_lying(client, db):
    """The database settles simultaneous picks. The loser has to be told."""
    db.seats[0]["token"] = "mine"
    db.rejected = True
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    r = client.post("/draft/sharedroom/pick", data={"player": "Nikola Jokic"})
    assert r.status_code == 302
    assert "just+took" in r.headers["Location"]


def test_a_shared_room_writes_nothing_to_the_cookie(client, db):
    """The database is the one copy. A second copy in one browser is how the
    two would drift apart."""
    db.seats[0]["token"] = "mine"
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    r = client.post("/draft/sharedroom/pick", data={"player": "Nikola Jokic"})
    assert not any("puntfit_draft_" in h for h in r.headers.getlist("Set-Cookie"))


def test_a_full_room_says_so(client, db, monkeypatch):
    monkeypatch.setattr(A.shared, "claim", lambda rid, tok, name=None: None)
    r = client.post("/draft/sharedroom/join", data={"name": "Late"})
    assert r.status_code == 302 and "full=1" in r.headers["Location"]
    assert "Every seat is taken" in client.get("/draft/sharedroom?full=1").data.decode()
