"""Draft rooms several people share, backed by Postgres.

A solo draft lives in the drafter's own cookie, which is what lets it survive
the server restarting. That cannot work for a room shared with other people:
a cookie belongs to one browser, nobody else can read it, and there is no
copy the two sides could agree on. So a shared room keeps its picks in the
database and every browser reads the same rows.

Identity is a token in a cookie, not an account. The invite link is the
credential - whoever opens it claims a free seat - because asking a league to
register before they can draft is friction that would stop the feature being
used at all.

Seats nobody claims are drafted by the simulated managers, so a room does not
stall waiting for a tenth person who was never coming.

Two constraints in the schema settle the races rather than application code:
picks is keyed on (room_id, number) so two people cannot take the same slot
in the order, and unique on (room_id, player) so they cannot take the same
player. The loser gets an integrity error and is told somebody just picked,
which is the truth.
"""
from __future__ import annotations

import secrets

from . import store


def new_token() -> str:
    return secrets.token_urlsafe(16)


def create(room_id: str, *, teams: int, rounds: int, fmt: str, seed: int,
           fingerprint: str, host_token: str) -> None:
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO rooms (id, fingerprint, teams, rounds, format, mode,"
            " seed, host_token) VALUES (%s, %s, %s, %s, %s, 'shared', %s, %s)",
            (room_id, fingerprint, teams, rounds, fmt, seed, host_token))
        # executemany is a cursor method in psycopg 3, not a connection one.
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO seats (room_id, seat) VALUES (%s, %s)",
                [(room_id, s) for s in range(teams)])


def load(room_id: str) -> dict | None:
    """Everything needed to rebuild the room, in one round trip per table."""
    with store.connection() as conn:
        row = conn.execute(
            "SELECT teams, rounds, format, seed, fingerprint, host_token"
            " FROM rooms WHERE id = %s", (room_id,)).fetchone()
        if row is None:
            return None
        picks = conn.execute(
            "SELECT number, seat, player FROM picks WHERE room_id = %s"
            " ORDER BY number", (room_id,)).fetchall()
        seats = conn.execute(
            "SELECT seat, token, name, punts, emphasis, view FROM seats"
            " WHERE room_id = %s ORDER BY seat", (room_id,)).fetchall()

    teams, rounds, fmt, seed, fingerprint, host_token = row
    return {
        "teams": teams, "rounds": rounds, "format": fmt, "seed": seed,
        "fingerprint": fingerprint, "host_token": host_token,
        "picks": [p[2] for p in picks],
        "seats": {s[0]: {"token": s[1], "name": s[2], "punts": tuple(s[3] or ()),
                         "emphasis": s[4] or {}, "view": s[5]} for s in seats},
    }


def seat_for(room: dict, token: str | None) -> int | None:
    if not token:
        return None
    for seat, info in room["seats"].items():
        if info["token"] == token:
            return seat
    return None


def claim(room_id: str, token: str, name: str | None = None) -> int | None:
    """Take the lowest free seat, or None if the room is full.

    The update is conditional on the seat still being free, so two people
    opening the link at the same moment cannot land on the same one.
    """
    with store.connection() as conn:
        free = conn.execute(
            "SELECT seat FROM seats WHERE room_id = %s AND token IS NULL"
            " ORDER BY seat", (room_id,)).fetchall()
        for (seat,) in free:
            got = conn.execute(
                "UPDATE seats SET token = %s, name = %s"
                " WHERE room_id = %s AND seat = %s AND token IS NULL"
                " RETURNING seat", (token, name, room_id, seat)).fetchone()
            if got:
                return seat
    return None


def add_pick(room_id: str, number: int, seat: int, player: str) -> bool:
    """Record a pick. False means somebody got there first."""
    import psycopg
    try:
        with store.connection() as conn:
            conn.execute(
                "INSERT INTO picks (room_id, number, seat, player)"
                " VALUES (%s, %s, %s, %s)", (room_id, number, seat, player))
            conn.execute("UPDATE rooms SET updated_at = now() WHERE id = %s",
                         (room_id,))
        return True
    except psycopg.errors.UniqueViolation:
        return False


def update_seat(room_id: str, seat: int, **fields) -> None:
    """A seat's own settings: its build, its emphasis, how it reads the board.

    These belong to the person in the seat, not to the room, which is why
    they live on seats rather than rooms.
    """
    from psycopg.types.json import Jsonb

    allowed = {"punts", "emphasis", "view", "name"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    # emphasis is jsonb. psycopg 3 adapts a list to an array on its own but
    # not a dict to json, so that one is wrapped.
    if "emphasis" in fields:
        fields["emphasis"] = Jsonb(fields["emphasis"] or {})
    if "punts" in fields:
        fields["punts"] = list(fields["punts"] or ())

    sets = ", ".join(f"{k} = %s" for k in fields)
    with store.connection() as conn:
        conn.execute(f"UPDATE seats SET {sets} WHERE room_id = %s AND seat = %s",
                     (*fields.values(), room_id, seat))


def pulse(room_id: str) -> tuple[int, int, int] | None:
    """Just enough to know whether anything has happened: teams, rounds and
    how many picks are in.

    One round trip and no draft reconstruction, because this is polled by
    every browser in the room and the answer is almost always "nothing yet".
    """
    with store.connection() as conn:
        row = conn.execute(
            "SELECT r.teams, r.rounds,"
            " (SELECT count(*) FROM picks p WHERE p.room_id = r.id)"
            " FROM rooms r WHERE r.id = %s", (room_id,)).fetchone()
    return tuple(row) if row else None


def pick_count(room_id: str) -> int:
    with store.connection() as conn:
        return conn.execute("SELECT count(*) FROM picks WHERE room_id = %s",
                            (room_id,)).fetchone()[0]
