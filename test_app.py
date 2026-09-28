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
    assert "weakest category" in one
    assert "weakest 3 categories" in three


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
