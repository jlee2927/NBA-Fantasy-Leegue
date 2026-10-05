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
from .categories import FORMATS, label as cat_label
from .draft import MAX_EMPHASIS, NEUTRAL_EMPHASIS, DraftState, League
from .positions import apply_overrides, derive, disagreements, roster_view
from . import shared, store
from .fetch_injuries import label as label_injuries
from .valuation import (last_season, load_projections, stat_headings,
                        stat_table)

# The landing page is the same files GitHub Pages serves, not a copy of them,
# so the two can never drift apart.
DOCS = Path(__file__).resolve().parent.parent / "docs"

PROJECTIONS = Path("data/marcel_projections.json")
PROSPECTS = Path("data/prospect_projections.json")
INJURIES = Path("data/injuries.csv")
SEASONS = Path("data/nba_seasons.parquet")
POSITIONS = Path("data/positions_manual.csv")
ADP = Path("data/adp.csv")

app = Flask(__name__)
# Categories are keyed by their column name and shown by their fantasy name.
app.jinja_env.filters["cat"] = cat_label
app.jinja_env.filters["cats"] = lambda seq: [cat_label(c) for c in seq]
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
SEAT = "puntfit_seat_"      # which seat a browser holds in a shared room
SEAT_AGE = 60 * 60 * 24 * 7  # a league draft can be scheduled days out


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
    if room.get("shared"):
        return response                  # the database already has it
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
            df["injury_short"] = None
        # A missing status arrives as NaN, which is truthy in a template and
        # renders a "nan" badge beside every healthy player.
        for col in ("injury_status", "injury_detail", "injury_short"):
            df[col] = df[col].where(df[col].notna(), "")
        app.config["players"], app.config["params"] = df, params
        app.config["positions"] = apply_overrides(derive(df), df, POSITIONS)
        # Season labels are the ending year, so 2026-2027 projections are
        # compared against season 2026, the year just finished.
        finished = int(str(params.get("target_season", "0-0")).split("-")[0])
        app.config["last_season"] = last_season(SEASONS, df, finished)
        # Where the market takes players, kept strictly apart from what they
        # are worth. Blending the two would make this ranking a partial copy
        # of the consensus it exists to disagree with.
        adp = pd.Series(dtype=float)
        if ADP.exists():
            table = pd.read_csv(ADP).dropna(subset=["adp"])
            by_id = dict(zip(table.athlete_id, table.adp))
            adp = df.player.map(by_id)
        app.config["adp"] = adp
        app.config["last_label"] = f"{finished - 1}-{str(finished)[2:]}"
    return app.config["players"], app.config["params"]


def _shared(room_id: str) -> dict | None:
    """Rebuild a shared room from the database.

    Always from the rows, never from a cache: other people are picking in the
    same room, and a copy held in this process would be stale the moment they
    did. A draft is a few hundred rows, so reading it back is cheap.
    """
    if not store.configured():
        return None
    try:
        saved = shared.load(room_id)
    except Exception:
        return None                      # database down; solo drafts still work
    if saved is None:
        return None

    players, _ = _load()
    league = League(teams=saved["teams"], rounds=saved["rounds"],
                    categories=FORMATS[saved["format"]])
    state = DraftState(players, league)
    for name in saved["picks"]:
        try:
            state.make_pick(name)
        except (ValueError, KeyError):
            return None                  # projections moved under the room

    seat = shared.seat_for(saved, request.cookies.get(SEAT + room_id))
    settings = saved["seats"].get(seat, {}) if seat is not None else {}
    if seat is not None and settings.get("punts"):
        state.set_punt(seat, tuple(settings["punts"]))
    for category, value in (settings.get("emphasis") or {}).items():
        state.set_emphasis(seat, category, value)

    return {
        "id": room_id,
        "state": state, "seat": seat, "mode": "shared", "seed": saved["seed"],
        "format": saved["format"], "rng": random.Random(saved["seed"]),
        # Seats nobody claimed are played by the simulated managers, so the
        # room does not stall waiting for someone who was never coming. Every
        # claimed seat is excluded: a second human being autodrafted for is
        # the worst failure this feature could have.
        "strategies": {t: strat for t, strat
                       in Mg.assign(saved["teams"], -1, saved["seed"]).items()
                       if not saved["seats"].get(t, {}).get("token")},
        "build_mode": "manual", "draft_mode": "manual",
        "view": settings.get("view") or "value",
        "shared": True, "seats": saved["seats"],
        "pace": saved.get("pace", "live"),
        "pick_hours": saved.get("pick_hours", 8),
        "waiting": saved.get("waiting", 0.0),
        "taken": {s: i for s, i in saved["seats"].items() if i["token"]},
    }


def _room(room_id: str) -> dict | None:
    """The room, from memory, then the database, then the browser."""
    room = _rooms.get(room_id)
    if room is None:
        room = _shared(room_id)
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

    rounds = int(form.get("rounds", 13))
    fmt = form.get("format", "9cat")
    room_id = uuid.uuid4().hex[:10]

    if mode in ("shared", "slow"):
        if not store.configured():
            return render_template("setup.html", params=_load()[1],
                                   formats=list(FORMATS),
                                   error="Shared rooms need a database. "
                                         "DATABASE_URL is not set."), 503
        token = shared.new_token()
        pace = "slow" if mode == "slow" else "live"
        hours = min(max(int(form.get("pick_hours") or 8), 1), 72)
        shared.create(room_id, teams=teams, rounds=rounds, fmt=fmt, seed=seed,
                      fingerprint=_fingerprint(), host_token=token,
                      pace=pace, pick_hours=hours)
        shared.claim(room_id, token, form.get("name") or "Host")
        response = redirect(url_for("room", room_id=room_id))
        response.set_cookie(SEAT + room_id, token, max_age=SEAT_AGE,
                            samesite="Lax", httponly=True)
        return response

    league = League(teams=teams, rounds=rounds, categories=FORMATS[fmt])
    state = DraftState(players, league)
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

    In a shared room whatever it picks is written to the database here rather
    than by the caller. Four routes call this, and persisting at each of them
    meant one was missed: bot picks made while changing a setting happened on
    screen and never reached the rows, so the next person to load the room saw
    a draft that had gone backwards.
    """
    before = len(room["state"].picks)
    try:
        _advance_locally(room)
    finally:
        if room.get("shared") and room.get("id"):
            _persist(room["id"], room, before)


def _advance_locally(room: dict) -> None:
    state: DraftState = room["state"]
    seat, auto = room["seat"], room["draft_mode"] == "auto"
    auto_build(room)       # independent of who is doing the drafting

    # A slow draft gives every seat hours to think, so nothing is drafted for
    # anybody until that time has run out. Checked here rather than by a
    # scheduler: the deadline only matters when somebody is looking, and this
    # is where looking happens.
    if room.get("pace") == "slow":
        if room.get("waiting", 0) < room.get("pick_hours", 8) * 3600:
            return

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


@app.post("/draft/<room_id>/join")
def join(room_id: str):
    room = _room(room_id)
    if room is None or not room.get("shared"):
        return render_template("gone.html", room_id=room_id), 404
    token = shared.new_token()
    seat = shared.claim(room_id, token, (request.form.get("name") or "").strip() or None)
    if seat is None:
        return redirect(url_for("room", room_id=room_id, full=1))
    response = redirect(url_for("room", room_id=room_id))
    response.set_cookie(SEAT + room_id, token, max_age=SEAT_AGE,
                        samesite="Lax", httponly=True)
    return response


@app.get("/draft/<room_id>/pulse")
def pulse(room_id: str):
    """Has anything happened in this room?

    Deliberately not a room: no projections, no DraftState, no z-scores. One
    query and some arithmetic, because every browser in the draft asks this
    every few seconds and the answer is usually no.
    """
    if not store.configured():
        return {"picks": None}, 200
    try:
        got = shared.pulse(room_id)
    except Exception:
        return {"picks": None}, 200      # never let a poll break the page
    if got is None:
        return {"picks": None}, 404

    teams, rounds, picks = got
    league = League(teams=teams, rounds=rounds, categories=())
    complete = picks >= teams * rounds
    return {"picks": picks, "complete": complete,
            "clock": None if complete else league.slot(picks)}, 200


@app.get("/draft/<room_id>")
def room(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404

    # Whoever opens the invite link without a seat is being invited, not
    # locked out.
    if room.get("shared") and room["seat"] is None:
        return render_template(
            "join.html", room_id=room_id, room=room,
            taken=len(room["taken"]), teams=room["state"].league.teams,
            started=len(room["state"].picks) > 0,
            full=request.args.get("full") == "1")

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

    # Our ranking against the room's. A positive edge means the market
    # takes him later than we rate him, which is the only place a draft is
    # actually won.
    adp = app.config.get("adp", pd.Series(dtype=float))
    our_rank = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    edge = (adp.reindex(ranked.index) - our_rank).dropna()
    def gap(names):
        return [{"player": n, "rank": int(our_rank[n]), "adp": int(adp[n]),
                 "edge": int(edge[n])} for n in names]

    # Undervalued only: players the room rates well below this board, which
    # are the ones worth waiting on. A list of players to avoid is a longer
    # answer to a question nobody asks mid-pick.
    bargains = gap(edge[edge > 0].sort_values(ascending=False).head(10).index)

    cats = [c for c in board.columns if c != "total"]
    view = room.get("view", "value")
    stats = stat_table(detail, cats) if view in ("stats", "both") else None
    past = app.config.get("last_season")
    if view == "last" and past is not None:
        stats = stat_table(past.loc[board.index], cats)
    elif view == "last":
        view = "stats"                 # no history available; show the forecast
        stats = stat_table(detail, cats)

    return render_template(
        "draft.html", room_id=room_id, state=state, seat=seat, mode=room["mode"],
        board=board, detail=detail, cats=cats,
        view=view, stats=stats, stat_headings=stat_headings(cats),
        last_label=app.config.get("last_label", "last season"),
        mine=mine, totals=totals, builds=builds, auto=auto, search=search,
        matches=len(ranked), pool=len(state.available),
        build_mode=room["build_mode"], draft_mode=room["draft_mode"],
        n_punts=n_punts, max_punts=state.league.max_punts,
        weights=state.weights(seat) if mine else None,
        auto_weights=state.category_weights(seat) if mine else None,
        emphasis=state.emphasis.get(seat, {}), slots=slots,
        adp=adp, bargains=bargains, has_adp=bool(len(edge)),
        max_emphasis=MAX_EMPHASIS,
        confirm_restart=request.args.get("restart") == "1",
        room_shared=bool(room.get("shared")),
        pace=room.get("pace", "live"),
        pick_hours=room.get("pick_hours", 8),
        # How long the seat on the clock has left, as the people waiting on it
        # would say it rather than in seconds.
        clock_left=_time_left(room),
        invite=request.url_root.rstrip("/") + url_for("room", room_id=room_id),
        seats_taken=len(room.get("taken", {})),
        my_turn=state.on_the_clock == seat,
        log=list(reversed(state.picks)),
    )


@app.get("/draft/<room_id>/results")
def results(room_id: str):
    """How the draft turned out.

    A mock that fills every roster and stops has not told the manager
    anything. The question a category league is played for is how many
    categories a roster would take, so that is the headline, and the punt is
    graded on whether it actually happened rather than whether it was chosen.
    """
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    state: DraftState = room["state"]
    seat = room["seat"]
    if seat is None:
        # Opened the invite link and came straight here. There is no "your
        # team" to report on until a seat is taken.
        return redirect(url_for("room", room_id=room_id))
    players, _ = _load()

    table = state.standings()
    wins = state.category_wins(seat, table)
    # Rank 1 is the best in a category. A punted category is executed when the
    # rank is high, which is the one place on this page where last is good.
    ranks = table.rank(ascending=False, method="min").astype(int)
    punted = tuple(state.punts.get(seat, ()))

    rows = [{"cat": c, "total": float(table.loc[seat, c]),
             "rank": int(ranks.loc[seat, c]), "win": float(wins[c]),
             "punted": c in punted}
            for c in table.columns]
    contested = [r for r in rows if not r["punted"]]

    mine = state.roster(seat)
    return render_template(
        "results.html", room_id=room_id, state=state, seat=seat,
        rows=rows, punted=punted,
        expected=float(wins.sum()), categories=len(rows),
        contested_won=sum(r["win"] for r in contested),
        contested=len(contested),
        standings=table, ranks=ranks, mine=mine,
        slots=roster_view(mine, app.config["positions"], state.league.rounds),
        detail=players.loc[players.index.intersection(mine)],
        placing=int(ranks.loc[seat].mean().round()),
    )


def _time_left(room: dict) -> str | None:
    """What is left of a slow draft's clock, in words."""
    if room.get("pace") != "slow" or room["state"].complete:
        return None
    left = room.get("pick_hours", 8) * 3600 - room.get("waiting", 0)
    if left <= 0:
        return "time is up"
    hours, minutes = int(left // 3600), int(left % 3600 // 60)
    if hours:
        return f"{hours}h {minutes}m left" if minutes else f"{hours}h left"
    return f"{minutes}m left" if minutes else "under a minute left"


def _persist(room_id: str, room: dict, start: int) -> bool:
    """Write picks from `start` onward to the database."""
    for p in room["state"].picks[start:]:
        if not shared.add_pick(room_id, p.number, p.team, p.player):
            return False
    return True


@app.post("/draft/<room_id>/pick")
def pick(room_id: str):
    room = _room(room_id)
    if room is None:
        return render_template("gone.html", room_id=room_id), 404
    state: DraftState = room["state"]
    player = request.form.get("player", "").strip()

    if room.get("shared"):
        if room["seat"] is None:
            return redirect(url_for("room", room_id=room_id))
        if state.on_the_clock != room["seat"]:
            return redirect(url_for("room", room_id=room_id,
                                    error="it is not your pick"))
        number = len(state.picks)
        try:
            state.make_pick(player)
        except ValueError as e:
            return redirect(url_for("room", room_id=room_id, error=str(e)))
        # The database decides the race, not this process. A unique constraint
        # on (room_id, player) and a primary key on (room_id, number) mean the
        # loser of a simultaneous pick finds out here rather than ending up
        # with somebody else's player.
        if not shared.add_pick(room_id, number, room["seat"], player):
            return redirect(url_for("room", room_id=room_id,
                                    error=f"somebody just took {player}"))
        _advance(room)
        return redirect(url_for("room", room_id=room_id))

    try:
        state.make_pick(player)
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
    room = _room(room_id)
    if room is not None and room.get("shared"):
        # Everybody else in the room is still drafting. Leaving gives up the
        # seat; it does not throw away their draft.
        response = redirect(url_for("setup"))
        response.delete_cookie(SEAT + room_id)
        return response

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
    if request.form.get("view") in {"value", "stats", "both", "last"}:
        room["view"] = request.form["view"]
    if room.get("shared") and room["seat"] is not None:
        shared.update_seat(room_id, room["seat"], view=room["view"])
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
    if room.get("shared") and room["seat"] is not None:
        shared.update_seat(room_id, room["seat"],
                           emphasis=state.emphasis.get(room["seat"], {}))
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
    if room.get("shared") and room["seat"] is not None:
        shared.update_seat(room_id, room["seat"], punts=chosen)
    _advance(room)
    return save(room_id, room, redirect(url_for("room", room_id=room_id)))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
