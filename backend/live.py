"""
live.py -- reads the MLB Stats API live game feed and maps it to model inputs.

Field names were checked against a real feed (game 813071, 2025-10-01) for the schedule, the
count, the matchup and the pitch events (see tests/fixtures + tests/test_live_contract.py).
Still taken from the documented shape, not yet seen in a real response: liveData.plays.currentPlay
and liveData.linescore (inning, offense bases, runs). The parser tolerates both being absent.
To add a real fixture:  python capture_fixture.py <gamePk>
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

import trajectory

BASE = "https://statsapi.mlb.com/api"
ET = ZoneInfo("America/New_York")


def baseball_today(now=None):
    """Date used for the MLB schedule. MLB dates follow US Eastern time, and a game that runs past
    midnight still belongs to the previous day, so shift back 4 hours before taking the date.
    (Using the server's own clock breaks on hosts running in UTC: games vanish at 8 pm Eastern.)"""
    now = (now or datetime.now(ET)).astimezone(ET)
    return (now - timedelta(hours=4)).date()


class NoActivePlay(ValueError):
    """The feed has no play yet (pre-game, warmup) so there is nothing to recommend for."""


def _rank(status):
    if status == "In Progress":
        return 0
    if status.startswith("Final") or status == "Game Over":
        return 2
    return 1


def todays_games(day=None):
    day = day or baseball_today().isoformat()
    r = requests.get(f"{BASE}/v1/schedule", params={"sportId": 1, "date": day}, timeout=10)
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            games.append({
                "game_pk": g["gamePk"],
                "status": g.get("status", {}).get("detailedState", ""),
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
            })
    return sorted(games, key=lambda g: _rank(g["status"]))   # live games first, finished last


def fetch_feed(game_pk):
    r = requests.get(f"{BASE}/v1.1/game/{game_pk}/feed/live", timeout=10)
    r.raise_for_status()
    return r.json()


def _current_play(live):
    plays = live.get("plays") or {}
    play = plays.get("currentPlay")
    if not play:                                   # fall back to the most recent play
        all_plays = plays.get("allPlays") or []
        play = all_plays[-1] if all_plays else None
    if not play or "matchup" not in play or "count" not in play:
        raise NoActivePlay("no play in progress yet (the game may not have started)")
    return play


def _pitch(ev):
    pd_ = ev.get("pitchData") or {}
    c = pd_.get("coordinates") or {}
    return {"type": (ev.get("details") or {}).get("type", {}).get("code"),
            "plate_x": c.get("pX"), "plate_z": c.get("pZ"),
            "call": (ev.get("details") or {}).get("description"),
            "speed": pd_.get("startSpeed"), "spin": (pd_.get("breaks") or {}).get("spinRate"),
            "traj": trajectory.from_feed_coordinates(c)}


def parse_snapshot(feed):
    live = feed.get("liveData") or {}
    play = _current_play(live)
    count, matchup = play["count"], play["matchup"]
    ls = live.get("linescore") or {}
    offense = ls.get("offense") or {}
    return {
        "pitcher_id": matchup["pitcher"]["id"],
        "pitcher_name": matchup["pitcher"]["fullName"],
        "batter_id": matchup["batter"]["id"],
        "batter_name": matchup["batter"]["fullName"],
        "state": {
            "stand": matchup["batSide"]["code"],
            "p_throws": matchup["pitchHand"]["code"],
            # a finished plate appearance can still show 4 balls / 3 strikes / 3 outs: clamp to the
            # ranges the model was trained on
            "balls": min(count["balls"], 3),
            "strikes": min(count["strikes"], 2),
            "outs_when_up": min(count["outs"], 2),
            "inning": ls.get("currentInning") or 1,
            "on_base": int(any(b in offense for b in ("first", "second", "third"))),
        },
        "score": {"away": (ls.get("teams") or {}).get("away", {}).get("runs"),
                  "home": (ls.get("teams") or {}).get("home", {}).get("runs")},
        "pitches_this_at_bat": [_pitch(ev) for ev in play.get("playEvents") or [] if ev.get("isPitch")],
    }


def snapshot(game_pk):
    return parse_snapshot(fetch_feed(game_pk))
