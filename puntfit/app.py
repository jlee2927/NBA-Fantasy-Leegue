"""Flask app: the draft room.

Two modes over one state machine. In a mock draft the other seats are filled
by simulated managers; in the live assistant every pick is typed in by hand as
it happens in a draft being run somewhere else. Both call
DraftState.make_pick, so the rules live in one place.

Projections are loaded once at startup and never refetched, so a draft session
makes no network calls.

Usage:
    flask --app puntfit.app run
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from pathlib import Path

import pandas as pd
from flask import (Flask, make_response, redirect, render_template, request,
                   send_from_directory, url_for)

from . import managers as Mg
from .categories import FORMATS
from .draft import MAX_EMPHASIS, NEUTRAL_EMPHASIS, DraftState, League
from .positions import apply_overrides, derive, disagreements, roster_view
from . import store
from .fetch_injuries import label as label_injuries
from .valuation import load_projections, stat_headings, stat_table

# The landing page is the same files GitHub Pages serves, not a copy of them,
# so the two can never drift apart.
DOCS = Path(__file__).resolve().parent.parent / "docs"

PROJECTIONS = Path("data/marcel_projections.json")
PROSPECTS = Path("data/prospect_projections.json")
INJURIES = Path("data/injuries.csv")
POSITIONS = Path("data/positions_manual.csv")

app = Flask(__name__)
app.config["PROJECTIONS"] = PROJECTIONS

BOARD_SIZE = 40
# Manual mode exists so a manager can see the field rather than be handed one
# answer, so it offers every build of that size worth considering.
BUILDS_OFFERED = {"manual": 12, "auto": 4}

# The server keeps rooms in memory for speed, but the browser holds the
# authoritative copy. Any restart - a deploy, a crash, a free instance idling
# out - empties this dictionary, and a draft that only lived here would be
# gone with it. Everything needed to rebuild a room is small enough to ride in
# a cookie, so a restart costs a replay rather than the draft.
_rooms: dict[str, dict] = {}

STATE_VERSION = 1
COOKIE_AGE = 60 * 60 * 12        # a draft is one sitting
COOKIE = "puntfit_draft_"


def _fingerprint() -> str:
    """Identifies the projection set a draft was started against.

    Picks travel as positions in the player list to keep the cookie small, so
    replaying them against a list that has since changed would quietly hand
    someone a different team. Refusing is the only safe answer.
    """
    players, _ = _load()
    seed = f"{len(players)}|{players.index[0]}|{players.index[-1]}"
    return hashlib.sha1(seed.encode()).hexdigest()[:8]


def save(room_id: str, room: dict, response):
    """Write the room back to the browser after anything that changed it."""
    players, _ = _load()
    where = {name: i for i, name in enumerate(players.index)}
    state: DraftState = room["state"]
    blob = json.dumps({
        "v": STATE_VERSION, "fp": _fingerprint(),
        "t": state.league.teams, "r": state.league.rounds, "f": room["format"],
        "s": room["seat"], "m": room["mode"], "sd": room["seed"],
        "b": room["build_mode"], "d": room["draft_mode"],
        "vw": room["view"],
        "p": [where[pick.player] for pick in state.picks],
        "pu": list(state.punts.get(room["seat"], ())),
        "e": state.emphasis.get(room["seat"], {}),
    }, separators=(",", ":"))
    response.set_cookie(COOKIE + room_id, blob, max_age=COOKIE_AGE,
                        samesite="Lax", httponly=True)
    return response


def restore(room_id: str) -> dict | None:
    """Rebuild a room the server has forgotten, from what the browser kept.

    The picks are replayed rather than the roster being restored wholesale,
    so the rebuilt room is a genuine DraftState rather than a summary of one.
    """
    raw = request.cookies.get(COOKIE + room_id)
    if not raw:
        return None
    try:
        saved = json.loads(raw)
    except ValueError:
        return None
    if saved.get("v") != STATE_VERSION or saved.get("fp") != _fingerprint():
        return None          # different projections; the positions would lie

    players, _ = _load()
    league = League(teams=saved["t"], rounds=saved["r"],
                    categories=FORMATS[saved["f"]])
    state = DraftState(players, league)
    try:
        for position in saved["p"]:
            state.make_pick(players.index[position])
    except (IndexError, ValueError):
        return None          # corrupt or stale; better to start over

    seat = saved["s"]
    if saved.get("pu"):
        state.set_punt(seat, tuple(saved["pu"]))
    for category, value in (saved.get("e") or {}).items():
        state.set_emphasis(seat, category, value)

    return {"state": state, "seat": seat, "mode": saved["m"],
            "seed": saved["sd"], "format": saved["f"],
            "rng": random.Random(saved["sd"]),
            "strategies": Mg.assign(league.teams, seat, saved["sd"])
                          if saved["m"] == "mock" else {},
            "build_mode": saved["b"], "draft_mode": saved["d"],
            "view": saved.get("vw", "value")}


def _load():
    if "players" not in app.config:
        df, params = load_projections(app.config["PROJECTIONS"], PROSPECTS)
        if INJURIES.exists():
            df = label_injuries(df, pd.read_csv(INJURIES))
        else:
            df["injury_status"] = None
        # A missing status arrives as NaN, which is truthy in a template and
        # renders a "nan" badge beside every healthy player.
        for col in ("injury_status", "injury_detail"):
            df[col] = df[col].where(df[col].notna(), "")
        app.config["players"], app.config["params"] = df, params
        app.config["positions"] = apply_overrides(derive(df), df, POSITIONS)
    return app.config["players"], app.config["params"]


def _room(room_id: str) -> dict | None:
    """The room, from memory if it is there and from the browser if not."""
    room = _rooms.get(room_id)
    if room is None:
        room = restore(room_id)
        if room is not None:
            _rooms[room_id] = room
    return room


@app.get("/")
def home():
    """The front door is the landing page, not the draft room."""
    return send_from_directory(DOCS, "index.html")


@app.get("/assets/<path:name>")
def landing_asset(name: str):
    return send_from_directory(DOCS / "assets", name)


@app.get("/positions")
def positions_review():
    """Positions worth checking against the league's own platform.

    Not a decision, a worklist. The derived split can only shuffle a player
    inside the group ESPN put him in, so when ESPN is wrong the derivation
    inherits it, and nothing in a box score can promote Wembanyama to a centre
    slot. Ordered by draft rank, because a wrong position in the first round
    matters and one in the three hundredth does not.
    """
    players, _ = _load()
    from .valuation import value_players
    z, _ = value_players(players)
    ranks = pd.Series(range(1, len(z) + 1), index=z.index)
    rows = disagreements(players, app.config["positions"], ranks)
    for r in rows:
        athlete = players.player.get(r["player"])
        r["athlete_id"] = "" if pd.isna(athlete) else str(int(athlete))
    return render_template("positions.html", rows=rows,
                           overrides=POSITIONS.as_posix())


@app.get("/health")
def health():
    """Whether the app can reach its database.

    Exists so the Render environment variable can be verified the moment it is
    set, rather than waiting until shared rooms are built to find out it was
    wrong. Reports the host but never the credential.
    """
    if not store.configured():
        return {"database": "not configured",
                "detail": "DATABASE_URL is unset; solo drafts work, shared "
                          "rooms are unavailable"}, 200
    try:
        store.ensure_schema()
        info = store.check()
        host = store.database_url().split("@")[-1].split("/")[0]
        return {"database": "ok", "host": host,
                "server": info["version"], "tables": info["tables"]}, 200
    except Exception as e:
        # The class name alone, because the message can carry the connection
        # string and this endpoint is public.
        return {"database": "unreachable", "error": type(e).__name__}, 503


@app.get("/draft-board.html")
def draft_board():
    return send_from_directory(DOCS, "draft-board.html")


@app.get("/draft")
def setup():
    _, params = _load()
    return render_template("setup.html", params=params, formats=list(FORMATS))


@app.post("/draft")
def create():
    players, _ = _load()
    form = request.form
    teams = int(form.get("teams", 12))
    mode = form.get("mode", "mock")
    seed = int(form.get("seed") or random.randrange(1_000_000))

    # Drawing for position is how a real room starts, and drafting from the
    # same seat every time teaches only that seat. Random is the default.
    wanted = form.get("seat", "random")
    if wanted == "random":
        seat = random.Random(seed).randrange(teams)
    else:
        seat = min(max(int(wanted), 1), teams) - 1

    league = League(teams=teams, rounds=int(form.get("rounds", 13)),
                    categories=FORMATS[form.get("format", "9cat")])
    state = DraftState(players, league)
    room_id = uuid.uuid4().hex[:10]
    room = {
        "state": state, "seat": seat, "mode": mode, "seed": seed,
        "format": form.get("format", "9cat"),
        "rng": random.Random(seed),
        "strategies": Mg.assign(teams, seat, seed) if mode == "mock" else {},
        # Manual on both counts by default. The app proposing a build and then
        # ranking everything around it is the thing that pins a manager into a
        # strategy they never chose, so it waits to be asked.
        "build_mode": form.get("build_mode", "manual"),
        "draft_mode": form.get("draft_mode", "manual"),
        # Z-scores are the default because they are what the ranking is built
        # from; the raw projections are a click away for anyone who reads the
        # categories faster that way.
        "view": "value",
    }
    _rooms[room_id] = room
    _advance(room)
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


def _advance(room: dict) -> None:
    """Move the draft on as far as it can go without the manager.

    Simulated seats always pick themselves. The manager's own seat only picks
    itself when they have asked it to, which is what lets a whole mock run
    through to the end.
    """
    state: DraftState = room["state"]
    seat, auto = room["seat"], room["draft_mode"] == "auto"
    auto_build(room)       # independent of who is doing the drafting
    for _ in range(state.league.total_picks + 1):
        if state.complete:
            return
        if state.on_the_clock in room["strategies"]:
            Mg.autopick(state, room["strategies"], room["rng"])
        elif auto and state.on_the_clock == seat:
            board = state.recommend(team=seat, limit=1)
            if board.empty:
                return
            state.make_pick(board.index[0])
            auto_build(room)   # that may have been the first pick
        else:
            return


def auto_build(room: dict) -> bool:
    """Apply the suggested build once, if the manager asked for that."""
    state: DraftState = room["state"]
    seat = room["seat"]
    if room["build_mode"] != "auto" or seat in state.punts:
        return False
    roster = state.roster(seat)
    if not roster:
        return False
    state.set_punt(seat, state.auto_punt(roster[0], room.get("n_punts", 1)))
    return True


@app.get("/draft/<room_id>")
def room(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    state: DraftState = room["state"]
    players, _ = _load()
    seat = room["seat"]

    ranked = state.recommend(team=seat, limit=len(state.projections))
    search = request.args.get("q", "").strip()
    if search:
        ranked = ranked[ranked.index.str.contains(search, case=False, regex=False)]
    board = ranked.head(BOARD_SIZE)
    detail = players.loc[board.index]
    mine = state.roster(seat)
    values = state.values(state.punts.get(seat, ()))
    totals = values.loc[values.index.intersection(mine, sort=False)] if mine else None

    # The build chooser. It stays offered after a build is set, so a manager
    # can change their mind rather than living with a first-round decision.
    auto, builds, n_punts = None, None, 0
    if mine:
        try:
            wanted = int(request.args.get("punts", 1))
        except ValueError:
            wanted = 1
        n_punts = max(1, min(wanted, state.league.max_punts))
        auto = state.auto_punt(mine[0], n_punts)
        builds = state.suggest_punts(mine[0], n_punts,
                                     limit=BUILDS_OFFERED[room["build_mode"]])

    slots = roster_view(mine, app.config["positions"], state.league.rounds)

    cats = [c for c in board.columns if c != "total"]
    view = room.get("view", "value")
    stats = stat_table(detail, cats) if view in ("stats", "both") else None

    return render_template(
        "draft.html", room_id=room_id, state=state, seat=seat, mode=room["mode"],
        board=board, detail=detail, cats=cats,
        view=view, stats=stats, stat_headings=stat_headings(cats),
        mine=mine, totals=totals, builds=builds, auto=auto, search=search,
        matches=len(ranked), pool=len(state.available),
        build_mode=room["build_mode"], draft_mode=room["draft_mode"],
        n_punts=n_punts, max_punts=state.league.max_punts,
        weights=state.weights(seat) if mine else None,
        auto_weights=state.category_weights(seat) if mine else None,
        emphasis=state.emphasis.get(seat, {}), slots=slots,
        max_emphasis=MAX_EMPHASIS,
        confirm_restart=request.args.get("restart") == "1",
        my_turn=state.on_the_clock == seat,
        log=list(reversed(state.picks)),
    )


@app.post("/draft/<room_id>/pick")
def pick(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    player = request.form.get("player", "").strip()
    try:
        room["state"].make_pick(player)
    except ValueError as e:
        return save(room_id, room, redirect(url_for("room", room_id=room_id, error=str(e))))
    _advance(room)
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


@app.post("/draft/<room_id>/restart")
def restart(room_id: str):
    """Abandon this draft and go back to setup.

    Behind a confirmation because it throws away a draft in progress, and a
    manager reaching for it after something went wrong should not be able to
    destroy a good draft with one stray click.
    """
    _rooms.pop(room_id, None)
    response = redirect(url_for("setup"))
    response.delete_cookie(COOKIE + room_id)
    return response


@app.post("/draft/<room_id>/settings")
def settings(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    for key in ("build_mode", "draft_mode"):
        value = request.form.get(key)
        if value in {"auto", "manual"}:
            room[key] = value
    if request.form.get("view") in {"value", "stats", "both"}:
        room["view"] = request.form["view"]
    _advance(room)
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


@app.post("/draft/<room_id>/clear-punt")
def clear_punt(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    room["state"].clear_punt(room["seat"])
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


@app.post("/draft/<room_id>/emphasis")
def emphasis(room_id: str):
    """Per-category emphasis, multiplied over the automatic weighting."""
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    state: DraftState = room["state"]
    if request.form.get("reset"):
        state.clear_emphasis(room["seat"])
        return save(room_id, room, redirect(url_for("room", room_id=room_id)))
    for category in state.league.categories:
        raw = request.form.get(f"w_{category}")
        if raw is None:
            continue
        try:
            state.set_emphasis(room["seat"], category, float(raw))
        except ValueError:
            continue
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


@app.post("/draft/<room_id>/punt")
def punt(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    state: DraftState = room["state"]
    chosen = tuple(c for c in request.form.get("build", "").split(",") if c)
    if len(chosen) > state.league.max_punts:
        chosen = chosen[:state.league.max_punts]
    if chosen:
        state.set_punt(room["seat"], chosen)
    else:
        state.clear_punt(room["seat"])   # "no build" is a legitimate choice
    _advance(room)
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
