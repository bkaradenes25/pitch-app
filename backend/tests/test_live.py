from datetime import date, datetime
from zoneinfo import ZoneInfo

import live


def test_parse_snapshot_maps_feed_to_model_inputs(fake_feed):
    snap = live.parse_snapshot(fake_feed)
    assert snap["pitcher_id"] == 1 and snap["batter_id"] == 7
    assert snap["state"] == {"stand": "L", "p_throws": "R", "balls": 1, "strikes": 2,
                             "outs_when_up": 1, "inning": 4, "on_base": 1}
    assert snap["score"] == {"away": 2, "home": 1}


def test_only_real_pitches_are_included(fake_feed):
    pitches = live.parse_snapshot(fake_feed)["pitches_this_at_bat"]
    assert len(pitches) == 1
    assert pitches[0]["type"] == "FF" and pitches[0]["plate_x"] == 0.1


def test_bases_empty_means_on_base_zero(fake_feed):
    fake_feed["liveData"]["linescore"]["offense"] = {}
    assert live.parse_snapshot(fake_feed)["state"]["on_base"] == 0


def test_strikes_and_outs_are_capped_to_model_range(fake_feed):
    fake_feed["liveData"]["plays"]["currentPlay"]["count"].update(strikes=3, outs=3)
    s = live.parse_snapshot(fake_feed)["state"]
    assert s["strikes"] == 2 and s["outs_when_up"] == 2


def test_todays_games_parsing(monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"dates": [{"games": [{"gamePk": 5, "status": {"detailedState": "In Progress"},
                                          "teams": {"away": {"team": {"name": "A"}},
                                                    "home": {"team": {"name": "B"}}}}]}]}
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: Resp())
    assert live.todays_games("2026-10-04") == [
        {"game_pk": 5, "status": "In Progress", "away": "A", "home": "B"}]


# ---- which day's schedule do we ask for? ----
def test_baseball_today_follows_us_eastern_time_even_when_the_host_clock_is_utc():
    utc = ZoneInfo("UTC")
    # 00:36 UTC on Oct 9 is 8:36 pm Eastern on Oct 8: the game that just started is still "today's"
    assert live.baseball_today(datetime(2026, 10, 9, 0, 36, tzinfo=utc)) == date(2026, 10, 8)
    assert live.baseball_today(datetime(2026, 10, 8, 20, 36, tzinfo=live.ET)) == date(2026, 10, 8)


def test_games_running_past_midnight_still_belong_to_the_previous_day():
    assert live.baseball_today(datetime(2026, 10, 9, 0, 40, tzinfo=live.ET)) == date(2026, 10, 8)
    assert live.baseball_today(datetime(2026, 10, 9, 5, 0, tzinfo=live.ET)) == date(2026, 10, 9)


def test_todays_games_asks_for_the_eastern_date_and_lists_live_games_first(monkeypatch):
    seen = {}

    def game(pk, status):
        return {"gamePk": pk, "status": {"detailedState": status},
                "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "B"}}}}

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"dates": [{"games": [game(1, "Final"), game(2, "In Progress"), game(3, "Pre-Game")]}]}

    def fake_get(url, params=None, timeout=None):
        seen.update(params)
        return Resp()

    monkeypatch.setattr(live, "baseball_today", lambda: date(2026, 10, 8))
    monkeypatch.setattr(live.requests, "get", fake_get)
    games = live.todays_games()
    assert seen["date"] == "2026-10-08"
    assert [g["game_pk"] for g in games] == [2, 3, 1]
