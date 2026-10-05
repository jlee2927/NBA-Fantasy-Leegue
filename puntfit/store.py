"""Shared storage for draft rooms.

A draft currently lives in the server's memory and in the drafter's own
cookie. That is enough for one person: the cookie is what lets a solo mock
survive the app restarting. It cannot work for a room shared with other
people, because a cookie belongs to one browser. Nobody else can read it, and
there is no copy either side can agree on.

So a shared room needs storage that is visible to everyone in it and that
outlives the process. This module is that storage.

Two constraints in the schema do the work that would otherwise need careful
locking:

  * picks (room_id, number) is the primary key, so two people cannot both
    claim the same pick in the order.
  * picks (room_id, player) is unique, so two people cannot take the same
    player, however close together they click.

Whoever loses the race gets a database error rather than a corrupted draft,
and the app can tell them plainly that somebody just took him.

Configuration is the DATABASE_URL environment variable. It is a credential,
so it is never committed: Render holds the real one, and a local .env (which
is gitignored) holds one for development.

The schema creates itself on first use, so there is no migration step to
remember and no command that has to be run from a particular machine. That
matters more than it sounds: many corporate networks block outbound port
5432, so the laptop this is developed on may not be able to reach the
database at all even though the deployed app can. Nothing should depend on
a human having run a command from the right network.

Usage:
    python -m puntfit.store --check     # can we reach the database?
    python -m puntfit.store --init      # create the tables (also automatic)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    id           TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Which projection set the draft was started against. Picks are stored by
    -- name here rather than by position, but a room opened against a different
    -- set would still be ranking against numbers its drafters never saw.
    fingerprint  TEXT NOT NULL,
    teams        INTEGER NOT NULL,
    rounds       INTEGER NOT NULL,
    format       TEXT NOT NULL,
    mode         TEXT NOT NULL,
    seed         INTEGER NOT NULL,
    host_token   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seats (
    room_id   TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    seat      INTEGER NOT NULL,
    -- NULL means nobody has claimed it: an open seat, or one the bots run.
    token     TEXT,
    name      TEXT,
    is_bot    BOOLEAN NOT NULL DEFAULT FALSE,
    punts     TEXT[] NOT NULL DEFAULT '{}',
    emphasis  JSONB  NOT NULL DEFAULT '{}',
    view      TEXT   NOT NULL DEFAULT 'value',
    PRIMARY KEY (room_id, seat)
);

CREATE TABLE IF NOT EXISTS picks (
    room_id  TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    number   INTEGER NOT NULL,
    seat     INTEGER NOT NULL,
    player   TEXT NOT NULL,
    made_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (room_id, number),
    CONSTRAINT one_team_per_player UNIQUE (room_id, player)
);

-- Rooms are cheap to keep but not worth keeping forever; this makes the
-- sweep of abandoned ones an index scan rather than a table scan.
CREATE INDEX IF NOT EXISTS rooms_updated_at ON rooms (updated_at);

-- Added after the first rooms existed. CREATE TABLE IF NOT EXISTS does
-- nothing to a table already there, so new columns need their own statement.
--
-- A slow draft runs over days rather than minutes: everybody gets hours to
-- think, and a seat is only drafted for once its clock has run out. pick_hours
-- is how long that is.
ALTER TABLE rooms ADD COLUMN IF NOT EXISTS pace TEXT NOT NULL DEFAULT 'live';
ALTER TABLE rooms ADD COLUMN IF NOT EXISTS pick_hours INTEGER NOT NULL DEFAULT 8;
"""


def _read_env_file() -> None:
    """Load .env for local development.

    Real environment variables always win: on Render the platform sets
    DATABASE_URL, and a stale local file must never override it.
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def database_url() -> str:
    _read_env_file()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Put the Neon connection string in a "
            f"local {ENV_FILE.name}, or in the environment on Render."
        )
    return url


def configured() -> bool:
    """Whether shared rooms are available at all.

    Solo drafts do not need the database, so the app should keep working
    without one rather than refusing to start.
    """
    _read_env_file()
    return bool(os.environ.get("DATABASE_URL"))


_pool = None


def pool():
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        # Small: a draft room is a handful of queries per page, and Neon's
        # free compute does not want a large pool held open against it.
        # Short timeouts on purpose. An unreachable database should fail a
        # request in seconds, not hold a worker for half a minute - during a
        # draft that is the difference between an error and a hang.
        _pool = ConnectionPool(database_url(), min_size=0, max_size=4,
                               max_idle=60, open=True, timeout=6,
                               kwargs={"connect_timeout": 5})
    return _pool


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn


_schema_ready = False


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA)


def ensure_schema() -> None:
    """Create the tables if they are not there, once per process.

    Every statement in SCHEMA is IF NOT EXISTS, so this is safe to call on a
    database that is already set up and safe to race against another worker
    doing the same thing.
    """
    global _schema_ready
    if _schema_ready:
        return
    init_schema()
    _schema_ready = True


class Unreachable(RuntimeError):
    """The database could not be reached, as opposed to refusing us.

    Worth separating, because the two have completely different fixes and a
    timeout is routinely misread as a wrong password.
    """


def check() -> dict:
    """Round-trip the database and report what is there."""
    with connection() as conn:
        version = conn.execute("SELECT version()").fetchone()[0]
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    return {"version": version.split(",")[0], "tables": [t[0] for t in tables]}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create the tables")
    ap.add_argument("--check", action="store_true", help="test the connection")
    a = ap.parse_args()

    if a.init:
        init_schema()
        print("schema created")
    # A check after --init is how you see that it worked.
    if a.check or a.init:
        info = check()
        host = database_url().split("@")[-1].split("/")[0]
        print(f"connected to {host}")
        print(f"  {info['version']}")
        print(f"  tables: {', '.join(info['tables']) or '(none yet)'}")
