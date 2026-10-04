import re
import pytest

from puntfit.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _start(client, **over):
    form = {"mode": "mock", "teams": "12", "rounds": "13", "seat": "1",
            "format": "9cat", "seed": "7", **over}
    r = client.post("/draft", data=form)
    assert r.status_code == 302
    return r.headers["Location"]


def _first_player(client, room):
    page = client.get(room).data.decode()
    return page.split('name="player" value="')[1].split('"')[0]


def _text(client, room):
    """Rendered page with runs of whitespace collapsed, since the template
    wraps sentences across lines."""
    import re
    return re.sub(r"\s+", " ", client.get(room).data.decode())


def test_the_front_door_is_the_landing_page(client):
    """Somebody arriving at the site should meet the pitch, not a form. The
    draft lives one level in, at /draft."""
    r = client.get("/")
    assert r.status_code == 200
    assert b"DRAFT<br>WITH <span>INTENT.</span>" in r.data
    assert b"New draft" not in r.data


def test_the_landing_page_serves_its_own_stylesheet(client):
    assert client.get("/assets/site.css").status_code == 200
    assert client.get("/assets/ball.js").status_code == 200


def test_positions_review_lists_the_doubtful_ones(client):
    r = client.get("/positions")
    assert r.status_code == 200
    body = r.data.decode()
    # ESPN files him under F, so nothing derived can reach a centre slot.
    assert "Wembanyama" in body
    assert "Positions worth checking" in body


def test_positions_review_is_reachable_from_setup(client):
    assert b"/positions" in client.get("/draft").data


def test_health_says_so_when_no_database_is_configured(client, monkeypatch):
    """Solo drafts do not need one, so this is a healthy state, not an error.

    DATABASE_URL is cleared rather than left to the environment: a developer
    with a real one set would otherwise make this test dial out, and a test
    suite that depends on the network is a test suite that hangs.
    """
    monkeypatch.setenv("DATABASE_URL", "")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["database"] == "not configured"


def test_health_never_leaks_the_credential(client, monkeypatch):
    """It is a public endpoint and the connection string holds a password."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secret@example/db")
    body = client.get("/health").data.decode()
    assert "secret" not in body and "postgresql://" not in body


def test_setup_page_renders(client):
    r = client.get("/draft")
    assert r.status_code == 200
    assert b"New draft" in r.data


def test_starting_a_draft_opens_a_room(client):
    r = client.get(_start(client))
    assert r.status_code == 200
    assert b"Best available" in r.data


def test_an_unknown_room_is_a_404(client):
    assert client.get("/draft/nosuchroom").status_code == 404


# ------------------------------------------------ surviving a server restart
def _play(client, room, picks):
    for _ in range(picks):
        client.post(f"{room}/pick", data={"player": _first_player(client, room)})


def test_a_draft_survives_the_server_restarting(client):
    """A deploy, a crash or a free instance idling out all empty the server's
    memory. Losing someone's draft at pick 90 is unrecoverable, so the browser
    keeps the authoritative copy and the room is replayed from it."""
    from puntfit.app import _rooms

    room = _start(client, mode="mock", seat="7")
    _play(client, room, 3)
    client.post(f"{room}/punt", data={"build": "tov"})
    before = _text(client, room)

    _rooms.clear()                      # exactly what a redeploy does

    after = _text(client, room)
    assert re.search(r"Every pick \((\d+)\)", after).group(1) == \
           re.search(r"Every pick \((\d+)\)", before).group(1)
    assert re.search(r"Your roster \((\d+)\)", after).group(1) == \
           re.search(r"Your roster \((\d+)\)", before).group(1)
    assert "punting" in after.split("Best available")[0]


def test_a_rebuilt_draft_keeps_its_settings(client):
    from puntfit.app import _rooms

    room = _start(client, mode="live", seat="4", teams="10", rounds="8",
                  build_mode="auto")
    _play(client, room, 2)
    _rooms.clear()

    page = _text(client, room)
    assert "you are team 4 of 10" in page
    assert "live draft" in page
    # the toggles are part of the draft, not of the server that forgot it
    assert '<option value="auto" selected>Pick one for me</option>' in page
    assert '<option value="manual" selected>I choose</option>' in page


def test_a_forgotten_draft_explains_itself(client):
    """Rather than a bare Flask 404 with no way forward."""
    r = client.get("/draft/neverexisted")
    assert r.status_code == 404
    assert b"no longer available" in r.data
    assert b"Start a new draft" in r.data


def test_a_changed_projection_set_refuses_to_replay(client):
    """Picks travel as positions in the player list, so replaying them against
    a different list would hand someone another player's team."""
    from puntfit.app import _rooms, app as flask_app

    room = _start(client)
    _play(client, room, 2)
    _rooms.clear()
    players = flask_app.config["players"]
    flask_app.config["players"] = players.iloc[:-5]      # projections refreshed
    try:
        assert client.get(room).status_code == 404
    finally:
        flask_app.config["players"] = players


# -------------------------------------------------------------- starting over
def test_starting_over_asks_first(client):
    room = _start(client)
    _play(client, room, 2)
    assert "start over" in _text(client, room)

    confirm = _text(client, f"{room}?restart=1")
    assert "Start over?" in confirm
    assert "cannot be undone" in confirm
    picks, mine = re.search(r"(\d+) picks? so far, including your (\d+)", confirm).groups()
    assert int(picks) > 0 and int(mine) == 2


def test_declining_keeps_the_draft(client):
    room = _start(client)
    _play(client, room, 2)
    before = re.search(r"Every pick \((\d+)\)", _text(client, room)).group(1)
    client.get(f"{room}?restart=1")                      # looked, did not confirm
    assert re.search(r"Every pick \((\d+)\)", _text(client, room)).group(1) == before


def test_confirming_ends_the_draft(client):
    room = _start(client)
    _play(client, room, 2)
    r = client.post(f"{room}/restart")
    assert r.status_code == 302 and r.headers["Location"] == "/draft"
    assert client.get(room).status_code == 404          # cookie cleared too


def test_healthy_players_get_no_injury_badge(client):
    """A missing status arrives as NaN, which is truthy in a template and
    rendered a "nan" badge beside every healthy player."""
    assert b">nan<" not in client.get(_start(client)).data


def test_drafting_a_player_removes_them_from_the_board(client):
    room = _start(client)
    name = _first_player(client, room)
    client.post(f"{room}/pick", data={"player": name})
    assert f'value="{name}"' not in client.get(room).data.decode()


def test_a_duplicate_pick_is_rejected_without_crashing(client):
    room = _start(client, mode="live")
    name = _first_player(client, room)
    client.post(f"{room}/pick", data={"player": name})
    assert client.post(f"{room}/pick", data={"player": name}).status_code == 302
    assert client.get(room).status_code == 200


def test_draft_position_is_drawn_at_random_by_default(client):
    """Drafting from the same seat every mock only teaches that seat."""
    seats = set()
    for seed in range(20):
        room = _start(client, seat="random", seed=str(seed))
        page = _text(client, room)
        slot = page.split("you are team ")[1].split(" of ")[0]
        seats.add(int(slot))
    assert len(seats) > 1, "random should not keep landing on one seat"
    assert all(1 <= s <= 12 for s in seats)


def test_a_chosen_draft_position_is_respected(client):
    page = _text(client, _start(client, seat="9"))
    assert "you are team 9 of 12" in page


def test_a_seat_past_the_league_size_is_pulled_back(client):
    page = _text(client, _start(client, teams="8", seat="14"))
    assert "you are team 8 of 8" in page


def test_live_mode_does_not_draft_for_you(client):
    """Every seat is entered by hand, so opening the room must not advance it."""
    assert b"No picks yet" in client.get(_start(client, mode="live")).data


def test_a_mock_makes_the_earlier_seats_pick_before_you(client):
    """Drawing pick 7 has to mean six players are gone before your turn."""
    page = _text(client, _start(client, mode="mock", seat="7"))
    assert "Round 1 &middot; pick 7" in page or "Round 1 · pick 7" in page
    assert "Your pick." in page
    assert "Every pick (6)" in page


def test_live_mode_says_whose_pick_you_are_entering(client):
    """Live mode has no simulated managers - you type in every team's pick -
    and looking identical to a mock is what made that confusing."""
    room = _start(client, mode="live", seat="7")
    page = _text(client, room)
    assert "Team 1 is on the clock." in page
    assert "enter whoever team 1 takes" in page

    for _ in range(6):                      # enter the six picks ahead of you
        client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    assert "Your pick." in _text(client, room)


def test_taken_players_leave_the_board(client):
    room = _start(client, mode="mock", seat="7")
    page = client.get(room).data.decode()
    log = page.split("Every pick")[1]
    taken = re.findall(r"<td>([^<]+)</td>\s*</tr>", log)
    board = page.split("Best available")[1].split("Every pick")[0]
    assert len(taken) == 6
    for name in taken:
        assert f'value="{name}"' not in board, name


# ------------------------------------------------------------ punt builds
def test_the_first_pick_brings_up_the_build_chooser(client):
    room = _start(client)
    name = _first_player(client, room)
    client.post(f"{room}/pick", data={"player": name})
    page = client.get(room).data.decode()
    assert "Choose your build" in page
    assert name in page


def test_the_chooser_stops_at_the_last_winnable_build(client):
    """Punting five of nine cannot win a week, so it is not offered."""
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    page = _text(client, room)
    assert '<option value="4"' in page
    assert '<option value="5"' not in page
    assert "you need 5 of 9 to win a week" in page


def test_asking_for_too_many_punts_is_clamped(client):
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    page = _text(client, f"{room}?punts=9")
    assert '<option value="4" selected' in page


def test_the_count_changes_which_build_is_offered(client):
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    one = _text(client, f"{room}?punts=1")
    three = _text(client, f"{room}?punts=3")
    assert "weakest 1" in one
    assert "weakest 3" in three
    # a three-category build names three categories
    assert one.count('name="build" value="') < three.count('name="build" value="')


def test_confirming_a_build_reranks_the_board(client):
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    before = client.get(room).data.decode()
    client.post(f"{room}/punt", data={"build": "tov,ft"})
    after = client.get(room).data.decode()

    assert "punting" in after
    assert before.split("Best available")[1] != after.split("Best available")[1]


def _board_header(client, room):
    return client.get(room).data.decode().split("Best available")[1].split("</thead>")[0]


def test_the_board_shows_standardised_value_by_default(client):
    """The ranking is built from z-scores, so that is what it opens on."""
    header = _board_header(client, _start(client))
    assert ">pts<" in header and ">p/g<" not in header


def test_the_board_can_switch_to_projected_statistics(client):
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "stats"})
    header = _board_header(client, room)
    assert ">p/g<" in header and ">pts<" not in header
    # Games and minutes only mean something once the line is a projection.
    assert ">g<" in header and ">m/g<" in header


def test_the_chosen_view_survives_a_pick(client):
    """A display preference that reset every round would be unusable."""
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "stats"})
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    assert ">p/g<" in _board_header(client, room)


def test_an_unknown_view_is_ignored(client):
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "nonsense"})
    assert ">pts<" in _board_header(client, room)


def test_the_combined_view_shows_both_numbers(client):
    """The z-score is what the board is sorted by; the raw line is context."""
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "both"})
    header = _board_header(client, room)
    assert ">pts<" in header          # still the standardised heading
    assert "raw" in client.get(room).data.decode()


def test_an_empty_roster_shows_every_slot_open(client):
    page = client.get(_start(client)).data.decode()
    assert "Roster &mdash; 0/13" in page or "Roster — 0/13" in page
    assert page.count("Empty") == 13
    assert "Position tracker" in page


def test_drafting_puts_the_player_in_a_slot(client):
    room = _start(client)
    name = _first_player(client, room)
    client.post(f"{room}/pick", data={"player": name})
    page = client.get(room).data.decode()
    assert page.count("Empty") == 12
    assert "13 starting slot" not in page


def test_the_board_can_show_last_seasons_actuals(client):
    """A forecast is easier to judge beside the season it came from."""
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "last"})
    header = _board_header(client, room)
    assert ">p/g<" in header and ">g<" in header


def test_last_season_differs_from_the_projection(client):
    """If the two views rendered the same numbers, one of them is wrong."""
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "stats"})
    forecast = client.get(room).data.decode().split("Best available")[1]
    client.post(f"{room}/settings", data={"view": "last"})
    actual = client.get(room).data.decode().split("Best available")[1]
    assert forecast != actual


def test_a_player_with_no_history_renders_a_dash(client):
    """Rookies have no last season. They should not show a zero, which would
    read as a projection of nothing rather than an absence of data."""
    room = _start(client)
    client.post(f"{room}/settings", data={"view": "last"})
    body = client.get(room).data.decode()
    assert "nan" not in body.lower().split("best available")[1]


def test_injury_badges_are_abbreviated(client):
    """The badge sits in a table already wider than a phone, and "Game Time
    Decision" is wider than most player names."""
    page = client.get(_start(client)).data.decode()
    assert "Day-To-Day<" not in page
    if "DTD" in page:                      # only if somebody is listed day-to-day
        assert 'title="Day-To-Day' in page  # the full status survives on hover


def test_a_confirmed_build_drops_those_columns(client):
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    client.post(f"{room}/punt", data={"build": "tov,ft"})
    page = client.get(room).data.decode()
    header = page.split("Best available")[1].split("</thead>")[0]
    assert ">tov<" not in header and ">ft<" not in header
    assert ">pts<" in header


# ------------------------------------------------------ automatic / manual
def _after_first_pick(client, **over):
    room = _start(client, **over)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    return room


def _build_applied(client, room):
    page = client.get(room).data.decode()
    return "punting" in page.split("Best available")[0]


def test_nothing_is_applied_until_the_manager_asks(client):
    """The default. Proposing a build and ranking the whole board around it
    is what pins someone into a strategy they never chose."""
    assert not _build_applied(client, _after_first_pick(client))


def test_automatic_build_mode_applies_one(client):
    assert _build_applied(client, _after_first_pick(client, build_mode="auto"))


def test_the_two_toggles_are_independent(client):
    """Automatic builds must not require handing over the picks as well."""
    room = _after_first_pick(client, build_mode="auto", draft_mode="manual")
    assert _build_applied(client, room)
    assert "draft complete" not in client.get(room).data.decode()


def test_switching_to_automatic_mid_draft_applies_a_build(client):
    room = _after_first_pick(client)
    assert not _build_applied(client, room)
    client.post(f"{room}/settings", data={"build_mode": "auto"})
    assert _build_applied(client, room)


def test_automatic_picks_run_the_draft_out(client):
    room = _start(client, draft_mode="auto")
    page = _text(client, room)
    assert "Draft complete" in page
    assert "Your roster (13)" in page


def test_the_builds_are_offered_in_order_of_fit(client):
    import re
    page = client.get(_after_first_pick(client)).data.decode()
    values = [float(v) for v in
              re.findall(r'<td class="n muted">(-?\d+\.\d\d)</td>', page)]
    ranked = values[:9]
    assert ranked == sorted(ranked, reverse=True)


def test_a_build_can_be_changed_after_it_is_set(client):
    room = _after_first_pick(client)
    client.post(f"{room}/punt", data={"build": "tov"})
    page = client.get(room).data.decode()
    assert "Change your build" in page          # the chooser stays offered
    client.post(f"{room}/punt", data={"build": "blk"})
    assert "blk" in client.get(room).data.decode().split("Best available")[0]


def test_choosing_no_build_is_allowed(client):
    room = _after_first_pick(client)
    client.post(f"{room}/punt", data={"build": "tov"})
    assert _build_applied(client, room)
    client.post(f"{room}/punt", data={"build": ""})
    assert not _build_applied(client, room)


# ----------------------------------------------------------------- search
def test_search_narrows_the_board(client):
    room = _start(client)
    page = client.get(f"{room}?q=curry").data.decode()
    assert "Stephen Curry" in page
    assert "matched" in page


def test_search_reaches_players_below_the_top_forty(client):
    """The board shows 40; search is how you get at the other 480."""
    room = _start(client)
    shown = client.get(room).data.decode()
    deep = "Jalen Duren"
    if deep not in shown:
        assert deep in client.get(f"{room}?q=duren").data.decode()


def test_an_unmatched_search_does_not_break_the_page(client):
    room = _start(client)
    r = client.get(f"{room}?q=zzzznobody")
    assert r.status_code == 200
    assert b"0 matched" in r.data


# --------------------------------------------------------------- emphasis
def _with_a_roster(client):
    room = _start(client)
    for _ in range(3):
        client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    return room


def test_the_chase_panel_appears_once_you_have_players(client):
    page = _text(client, _with_a_roster(client))
    assert "What you&#39;re chasing" in page or "What you're chasing" in page
    assert 'name="w_pts"' in page


def test_adjusting_a_category_changes_the_board(client):
    room = _with_a_roster(client)
    before = client.get(room).data.decode().split("Best available")[1]
    client.post(f"{room}/emphasis", data={"w_stl": "2", "w_blk": "0"})
    after = client.get(room).data.decode().split("Best available")[1]
    assert before != after


def test_reset_restores_the_automatic_weights(client):
    room = _with_a_roster(client)
    plain = client.get(room).data.decode().split("Best available")[1]
    client.post(f"{room}/emphasis", data={"w_stl": "2", "w_blk": "0"})
    assert client.get(room).data.decode().split("Best available")[1] != plain
    client.post(f"{room}/emphasis", data={"reset": "1"})
    assert client.get(room).data.decode().split("Best available")[1] == plain


def test_a_nonsense_slider_value_is_ignored(client):
    room = _with_a_roster(client)
    r = client.post(f"{room}/emphasis", data={"w_stl": "banana"})
    assert r.status_code == 302
    assert client.get(room).status_code == 200


def test_an_oversized_build_posted_directly_is_trimmed(client):
    room = _start(client)
    client.post(f"{room}/pick", data={"player": _first_player(client, room)})
    client.post(f"{room}/punt", data={"build": "tov,ft,fg,blk,reb,ast"})
    page = client.get(room).data.decode()
    header = page.split("Best available")[1].split("</thead>")[0]
    kept = [c for c in ("fg", "ft", "tpm", "pts", "reb", "ast", "stl", "blk", "tov")
            if f">{c}<" in header]
    assert len(kept) >= 5


# ------------------------------------------------------------ post-draft
def _finish(client, **over):
    """Run a draft to the final whistle."""
    room = _start(client, teams="4", rounds="3", **over)
    for _ in range(60):
        page = client.get(room).data.decode()
        if 'name="player" value="' not in page:
            break
        client.post(f"{room}/pick",
                    data={"player": page.split('name="player" value="')[1].split('"')[0]})
    return room


def test_a_finished_draft_has_somewhere_to_go(client):
    """A mock that fills every roster and stops has told the manager nothing."""
    room = _finish(client)
    page = client.get(room).data.decode()
    assert "Draft complete" in page
    assert f"{room}/results" in page


def test_results_lead_with_categories_won(client):
    """Not a total value - no category league awards one."""
    page = client.get(f"{_finish(client)}/results").data.decode()
    assert "How the draft turned out" in page
    assert "of 9" in page


def test_results_grade_the_punt_that_was_set(client):
    room = _finish(client)
    client.post(f"{room}/punt", data={"build": "tov"})
    page = client.get(f"{room}/results").data.decode()
    assert "You punted" in page and "tov" in page


def test_results_say_when_no_build_was_set(client):
    page = client.get(f"{_finish(client)}/results").data.decode()
    assert "No build was set" in page


def test_results_show_every_team(client):
    page = client.get(f"{_finish(client)}/results").data.decode()
    assert "Every team" in page
    for t in range(1, 5):
        assert f"T{t}" in page


def test_results_of_an_unfinished_draft_say_so(client):
    """Useful mid-draft, as long as it does not pretend to be final."""
    room = _start(client, teams="4", rounds="3")
    page = client.get(f"{room}/results").data.decode()
    assert "still in progress" in page


def test_results_of_a_missing_room_are_not_found(client):
    assert client.get("/draft/nosuchroom/results").status_code == 404
