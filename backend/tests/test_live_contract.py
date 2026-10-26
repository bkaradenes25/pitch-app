"""
Contract tests: the live-feed parser against saved feed files (tests/fixtures/feed_*.json).

Why: live.py reads a JSON feed we don't control. A renamed or missing field would otherwise fail
silently in the middle of a game. Every fixture file is parsed and the output is checked against
exactly what the model and the API accept. Add real feeds with `python capture_fixture.py <gamePk>`.
"""
import copy
import json
from pathlib import Path

import pytest

import capture_fixture
import live

FIX = Path(__file__).parent / "fixtures"
ALL_FEEDS = sorted(FIX.glob("feed_*.json"))
PLAYING = [p for p in ALL_FEEDS if "pregame" not in p.name]
PREGAME = [p for p in ALL_FEEDS if "pregame" in p.name]


def load(path):
    return json.loads(path.read_text())


def test_fixtures_exist():
    assert PLAYING and PREGAME, "expected in-play and pre-game fixtures in tests/fixtures"


@pytest.mark.parametrize("path", PLAYING, ids=lambda p: p.stem)
def test_parser_output_is_valid_model_input(path):
    snap = live.parse_snapshot(load(path))
    s = snap["state"]
    assert isinstance(snap["pitcher_id"], int) and isinstance(snap["batter_id"], int)
    assert snap["pitcher_name"] and snap["batter_name"]
    # exactly the ranges recommender / the API accept
    assert s["stand"] in ("L", "R") and s["p_throws"] in ("L", "R")
    assert 0 <= s["balls"] <= 3 and 0 <= s["strikes"] <= 2 and 0 <= s["outs_when_up"] <= 2
    assert s["on_base"] in (0, 1) and s["inning"] >= 1
    for p in snap["pitches_this_at_bat"]:
        assert isinstance(p["type"], str) and p["type"]
        assert p["plate_x"] is None or isinstance(p["plate_x"], (int, float))
        assert p["plate_z"] is None or isinstance(p["plate_z"], (int, float))


@pytest.mark.parametrize("path", PLAYING, ids=lambda p: p.stem)
def test_parsed_state_gets_a_recommendation(path, modules):
    snap = live.parse_snapshot(load(path))
    out = modules.recommender.recommend(snap["pitcher_id"], snap["state"], snap["batter_id"], top_n=3)
    assert 1 <= len(out["recommendations"]) <= 3


@pytest.mark.parametrize("path", PREGAME, ids=lambda p: p.stem)
def test_pregame_feed_raises_a_clear_error_not_a_keyerror(path):
    with pytest.raises(live.NoActivePlay):
        live.parse_snapshot(load(path))


def test_known_values_from_the_midgame_fixture():
    snap = live.parse_snapshot(load(FIX / "feed_midgame_runner_on.json"))
    assert (snap["pitcher_id"], snap["batter_id"]) == (676440, 678009)
    assert snap["state"] == {"stand": "L", "p_throws": "R", "balls": 1, "strikes": 2,
                             "outs_when_up": 1, "inning": 4, "on_base": 1}
    assert snap["score"] == {"away": 2, "home": 1}
    pitches = snap["pitches_this_at_bat"]
    assert [p["type"] for p in pitches] == ["FC", "SL", "FF"]       # the pickoff attempt is not a pitch
    assert pitches[0]["plate_x"] == pytest.approx(-0.2505, abs=1e-3)
    assert pitches[0]["speed"] == 89.7 and pitches[0]["spin"] == 2545


def test_finished_plate_appearance_is_clamped_to_trained_ranges():
    s = live.parse_snapshot(load(FIX / "feed_finished_plate_appearance.json"))["state"]
    assert (s["balls"], s["strikes"], s["outs_when_up"]) == (3, 1, 2)       # 4 balls / 3 outs in the feed


def test_bases_loaded_detection_uses_any_base():
    feed = load(FIX / "feed_full_count_bases_empty.json")
    assert live.parse_snapshot(feed)["state"]["on_base"] == 0
    feed["liveData"]["linescore"]["offense"] = {"third": {"id": 1}}
    assert live.parse_snapshot(feed)["state"]["on_base"] == 1


# ---- resilience: the feed shapes we saw (or may see) in the wild ----
def test_falls_back_to_the_last_play_when_current_play_is_missing():
    feed = load(FIX / "feed_midgame_runner_on.json")
    expected = live.parse_snapshot(feed)
    del feed["liveData"]["plays"]["currentPlay"]
    assert live.parse_snapshot(feed) == expected


def test_a_pitch_without_tracking_data_has_empty_coordinates_instead_of_crashing():
    feed = load(FIX / "feed_midgame_runner_on.json")
    for ev in feed["liveData"]["plays"]["currentPlay"]["playEvents"]:
        ev.pop("pitchData", None)
    pitches = live.parse_snapshot(feed)["pitches_this_at_bat"]
    assert len(pitches) == 3 and all(p["plate_x"] is None and p["speed"] is None for p in pitches)


def test_missing_linescore_defaults_sensibly():
    feed = load(FIX / "feed_midgame_runner_on.json")
    del feed["liveData"]["linescore"]
    snap = live.parse_snapshot(feed)
    assert snap["state"]["inning"] == 1 and snap["state"]["on_base"] == 0
    assert snap["score"] == {"away": None, "home": None}


def test_renamed_required_field_fails_loudly():
    """If MLB renames a field we depend on, we want an error (-> HTTP 502), never a wrong default."""
    feed = load(FIX / "feed_midgame_runner_on.json")
    del feed["liveData"]["plays"]["currentPlay"]["matchup"]["pitchHand"]
    with pytest.raises(KeyError):
        live.parse_snapshot(feed)


# ---- capture_fixture.trim_feed must keep everything the parser reads ----
def test_trimming_a_bulky_feed_does_not_change_the_parse():
    feed = load(FIX / "feed_midgame_runner_on.json")
    bulky = copy.deepcopy(feed)
    play = bulky["liveData"]["plays"]["currentPlay"]
    play["matchup"]["batterHotColdZones"] = [{"zone": i} for i in range(13)]
    play["matchup"]["splits"] = {"batter": "vs_RHP"}
    play["result"] = {"type": "atBat"}
    play["playEvents"][1]["pitchData"]["coordinates"].update(extraTracking=1.0, spinAxis=200)
    bulky["liveData"]["linescore"]["offense"]["second"] = {"id": 11, "fullName": "Runner", "stats": [1] * 50}
    bulky["liveData"]["linescore"]["offense"]["batter"] = {"id": 5}      # not a base: must not count
    bulky["gameData"] = {"huge": list(range(1000))}

    trimmed = capture_fixture.trim_feed(bulky)
    assert live.parse_snapshot(trimmed) == live.parse_snapshot(feed)
    assert len(json.dumps(trimmed)) < len(json.dumps(bulky)) / 2
    assert "batterHotColdZones" not in json.dumps(trimmed)


def test_trimming_a_pregame_feed_still_raises_no_active_play():
    pre = load(PREGAME[0])
    with pytest.raises(live.NoActivePlay):
        live.parse_snapshot(capture_fixture.trim_feed(pre))


# ---- schedule contract ----
def test_schedule_fixture_parses_and_orders_live_games_first(monkeypatch):
    payload = json.loads((FIX / "schedule_sample.json").read_text())

    class Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    monkeypatch.setattr(live.requests, "get", lambda *a, **k: Resp())
    games = live.todays_games("2025-10-01")
    assert [g["status"] for g in games] == ["In Progress", "Pre-Game", "Final", "Final", "Game Over"]
    first_final = next(g for g in games if g["game_pk"] == 813071)
    assert first_final == {"game_pk": 813071, "status": "Final",
                           "away": "Detroit Tigers", "home": "Cleveland Guardians"}
