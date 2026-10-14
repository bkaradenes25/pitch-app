"""
live.py -- reads the MLB Stats API live game feed and maps it to model inputs.
Field names below match the feed as commonly documented; verify against a real
response (curl one gamePk) before relying on them.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

BASE = "https://statsapi.mlb.com/api"
ET = ZoneInfo("America/New_York")


def baseball_today(now=None):
    """Date used for the MLB schedule. MLB dates follow US Eastern time, and a game that runs past
    midnight still belongs to the previous day, so shift back 4 hours before taking the date.
    (Using the server's own clock breaks on hosts running in UTC: games vanish at 8 pm Eastern.)"""
    now = (now or datetime.now(ET)).astimezone(ET)
    return (now - timedelta(hours=4)).date()


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


def parse_snapshot(feed):
    live = feed["liveData"]
    play = live["plays"]["currentPlay"]
    count, matchup = play["count"], play["matchup"]
    ls = live.get("linescore", {})
    offense = ls.get("offense", {})
    pitches = []
    for ev in play.get("playEvents", []):
        if ev.get("isPitch"):
            c = ev.get("pitchData", {}).get("coordinates", {})
            pitches.append({"type": ev.get("details", {}).get("type", {}).get("code"),
                            "plate_x": c.get("pX"), "plate_z": c.get("pZ"),
                            "call": ev.get("details", {}).get("description")})
    return {
        "pitcher_id": matchup["pitcher"]["id"],
        "pitcher_name": matchup["pitcher"]["fullName"],
        "batter_id": matchup["batter"]["id"],
        "batter_name": matchup["batter"]["fullName"],
        "state": {
            "stand": matchup["batSide"]["code"],
            "p_throws": matchup["pitchHand"]["code"],
            "balls": count["balls"],
            "strikes": min(count["strikes"], 2),
            "outs_when_up": min(count["outs"], 2),
            "inning": ls.get("currentInning", 1),
            "on_base": int(any(b in offense for b in ("first", "second", "third"))),
        },
        "score": {"away": ls.get("teams", {}).get("away", {}).get("runs"),
                  "home": ls.get("teams", {}).get("home", {}).get("runs")},
        "pitches_this_at_bat": pitches,
    }


def snapshot(game_pk):
    return parse_snapshot(fetch_feed(game_pk))
