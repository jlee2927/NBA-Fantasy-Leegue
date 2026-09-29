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


def test_setup_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"New draft" in r.data


def test_starting_a_draft_opens_a_room(client):
    r = client.get(_start(client))
    assert r.status_code == 200
    assert b"Best available" in r.data


def test_an_unknown_room_is_a_404(client):
    assert client.get("/draft/nosuchroom").status_code == 404


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
    assert "draft complete" in page
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
