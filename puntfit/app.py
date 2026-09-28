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

import random
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

from . import managers as Mg
from .categories import FORMATS
from .draft import MAX_EMPHASIS, NEUTRAL_EMPHASIS, DraftState, League
from .fetch_injuries import label as label_injuries
from .valuation import load_projections

PROJECTIONS = Path("data/marcel_projections.json")
INJURIES = Path("data/injuries.csv")

app = Flask(__name__)
app.config["PROJECTIONS"] = PROJECTIONS

_rooms: dict[str, dict] = {}


def _load():
    if "players" not in app.config:
        df, params = load_projections(app.config["PROJECTIONS"])
        if INJURIES.exists():
            df = label_injuries(df, pd.read_csv(INJURIES))
        else:
            df["injury_status"] = None
        # A missing status arrives as NaN, which is truthy in a template and
        # renders a "nan" badge beside every healthy player.
        for col in ("injury_status", "injury_detail"):
            df[col] = df[col].where(df[col].notna(), "")
        app.config["players"], app.config["params"] = df, params
    return app.config["players"], app.config["params"]


def _room(room_id: str) -> dict:
    room = _rooms.get(room_id)
    if room is None:
        abort(404)
    return room


@app.route("/")
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
    _rooms[room_id] = {
        "state": state, "seat": seat, "mode": mode, "seed": seed,
        "rng": random.Random(seed),
        "strategies": Mg.assign(teams, seat, seed) if mode == "mock" else {},
    }
    _advance(_rooms[room_id])
    return redirect(url_for("room", room_id=room_id))


def _advance(room: dict) -> None:
    """In a mock, let the simulated seats pick until it is the user's turn."""
    if room["mode"] == "mock":
        Mg.autopick(room["state"], room["strategies"], room["rng"])


@app.get("/draft/<room_id>")
def room(room_id: str):
    room = _room(room_id)
    state: DraftState = room["state"]
    players, _ = _load()
    seat = room["seat"]

    board = state.recommend(team=seat, limit=40)
    detail = players.loc[board.index]
    mine = state.roster(seat)
    totals = (state.values(state.punts.get(seat, ()))
              .loc[state.values(state.punts.get(seat, ())).index.intersection(mine, sort=False)]
              if mine else None)

    auto, suggestions, n_punts = None, None, 0
    if mine and seat not in state.punts:
        try:
            wanted = int(request.args.get("punts", 1))
        except ValueError:
            wanted = 1
        n_punts = max(1, min(wanted, state.league.max_punts))
        auto = state.auto_punt(mine[0], n_punts)
        suggestions = state.suggest_punts(mine[0], n_punts)

    return render_template(
        "draft.html", room_id=room_id, state=state, seat=seat, mode=room["mode"],
        board=board, detail=detail, cats=[c for c in board.columns if c != "total"],
        mine=mine, totals=totals, suggestions=suggestions, auto=auto,
        n_punts=n_punts, max_punts=state.league.max_punts,
        weights=state.weights(seat) if mine else None,
        auto_weights=state.category_weights(seat) if mine else None,
        emphasis=state.emphasis.get(seat, {}),
        max_emphasis=MAX_EMPHASIS,
        my_turn=state.on_the_clock == seat or room["mode"] == "live",
        recent=list(reversed(state.picks[-12:])),
    )


@app.post("/draft/<room_id>/pick")
def pick(room_id: str):
    room = _room(room_id)
    player = request.form.get("player", "").strip()
    try:
        room["state"].make_pick(player)
    except ValueError as e:
        return redirect(url_for("room", room_id=room_id, error=str(e)))
    _advance(room)
    return redirect(url_for("room", room_id=room_id))


@app.post("/draft/<room_id>/emphasis")
def emphasis(room_id: str):
    """Per-category emphasis, multiplied over the automatic weighting."""
    room = _room(room_id)
    state: DraftState = room["state"]
    if request.form.get("reset"):
        state.clear_emphasis(room["seat"])
        return redirect(url_for("room", room_id=room_id))
    for category in state.league.categories:
        raw = request.form.get(f"w_{category}")
        if raw is None:
            continue
        try:
            state.set_emphasis(room["seat"], category, float(raw))
        except ValueError:
            continue
    return redirect(url_for("room", room_id=room_id))


@app.post("/draft/<room_id>/punt")
def punt(room_id: str):
    room = _room(room_id)
    state: DraftState = room["state"]
    chosen = tuple(c for c in request.form.get("build", "").split(",") if c)
    if len(chosen) > state.league.max_punts:
        chosen = chosen[:state.league.max_punts]
    state.set_punt(room["seat"], chosen)
    return redirect(url_for("room", room_id=room_id))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
