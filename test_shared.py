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


# ------------------------------------------- the database API, not the logic
class StrictConn:
    """A stand-in with exactly the methods psycopg 3's Connection has.

    Written because shared.create called conn.executemany, which psycopg 2
    allows and psycopg 3 does not - executemany is a cursor method. The tests
    above all mock at the module level and sailed straight past it; this fails
    the same way the real driver did.
    """

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql.strip().split()[0].upper())
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        # Only reachable through cursor(); a connection-level call would have
        # raised AttributeError before getting here.
        self.statements.append("EXECUTEMANY")


def test_create_uses_only_methods_a_psycopg3_connection_has(monkeypatch):
    import contextlib

    import puntfit.shared as S

    conn = StrictConn()
    # The real Connection has no executemany, so remove it from the object the
    # module is handed while leaving it on the cursor.
    class NoExecuteMany(StrictConn):
        executemany = property(lambda self: (_ for _ in ()).throw(
            AttributeError("'Connection' object has no attribute 'executemany'")))

        def cursor(self):
            return conn

    @contextlib.contextmanager
    def fake_connection():
        yield NoExecuteMany()

    monkeypatch.setattr(S.store, "connection", fake_connection)
    S.create("room", teams=4, rounds=3, fmt="9cat", seed=1,
             fingerprint="abc", host_token="t")
    assert "EXECUTEMANY" in conn.statements


def test_the_pulse_is_cheap_enough_to_poll(client, db, monkeypatch):
    """It must not rebuild a draft to answer "nothing yet". Nothing in the
    room-loading path should be touched at all."""
    calls = []
    monkeypatch.setattr(A.shared, "pulse", lambda rid: (4, 3, 2))
    monkeypatch.setattr(A.shared, "load",
                        lambda rid: calls.append(rid) or db.load(rid))
    r = client.get("/draft/sharedroom/pulse")
    assert r.status_code == 200
    assert r.json["picks"] == 2
    assert calls == []                      # the room was never built


def test_the_pulse_names_who_is_on_the_clock(client, db, monkeypatch):
    monkeypatch.setattr(A.shared, "pulse", lambda rid: (4, 3, 5))
    # Pick 5 in a four-team snake is the second round coming back: seat 2.
    assert client.get("/draft/sharedroom/pulse").json["clock"] == 2


def test_the_pulse_survives_the_database_falling_over(client, db, monkeypatch):
    """A failed poll must never break the page somebody is drafting on."""
    def boom(_):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(A.shared, "pulse", boom)
    r = client.get("/draft/sharedroom/pulse")
    assert r.status_code == 200 and r.json["picks"] is None


def test_a_room_does_not_poll_while_you_are_on_the_clock(client, db):
    """The draft is waiting for you, so nothing can change. Polling then is
    the busiest moment of the draft spent on requests with no news."""
    db.seats[0]["token"] = "mine"
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    page = client.get("/draft/sharedroom").data.decode()
    assert "var mine = true;" in page


def test_standings_without_a_seat_offer_one_instead_of_failing(client, db):
    """Opening the invite link and going straight to standings used to raise
    a KeyError on seat None, which is a 500 on somebody's first contact with
    the room."""
    r = client.get("/draft/sharedroom/results")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/draft/sharedroom")


def test_leaving_a_shared_room_does_not_end_it_for_everyone(client, db):
    """Restart throws away a solo draft, which is the drafter's to throw. A
    shared draft belongs to the room."""
    db.seats[0]["token"] = "mine"
    db.picks.append("Nikola Jokic")
    client.set_cookie("puntfit_seat_sharedroom", "mine")
    r = client.post("/draft/sharedroom/restart")
    assert r.status_code == 302
    assert db.picks == ["Nikola Jokic"]      # the draft is untouched
    assert any("puntfit_seat_sharedroom=;" in h
               for h in r.headers.getlist("Set-Cookie"))
