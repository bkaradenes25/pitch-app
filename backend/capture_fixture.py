"""
capture_fixture.py -- save a trimmed REAL MLB live feed as a test fixture.

    python capture_fixture.py 813071                 # a finished game
    python capture_fixture.py 813071 --name feed_wildcard_final

Writes tests/fixtures/<name>.json containing only the keys live.parse_snapshot reads (plus the
pitch-quality fields), so fixtures stay small and carry no hot/cold-zone blobs.
tests/test_live_contract.py picks up every tests/fixtures/feed_*.json automatically, so a real
feed that the parser cannot handle shows up as a failing test instead of a silent live bug.

Run it during a live game too: that is the only way to see a real in-progress currentPlay.
Feeds without any play yet are saved as feed_pregame_<gamePk>.json (the contract test expects
the parser to raise NoActivePlay for those).
"""
import argparse
import json
from pathlib import Path

import live

FIXTURES = Path(__file__).parent / "tests" / "fixtures"
NOTE = ("Trimmed copy of a real MLB Stats API feed made by capture_fixture.py; "
        "only the keys live.parse_snapshot reads are kept.")


def _pick(d, keys):
    d = d or {}
    return {k: d[k] for k in keys if k in d}


def trim_event(ev):
    out = {"isPitch": bool(ev.get("isPitch")), "details": {}}
    det = ev.get("details") or {}
    if "description" in det:
        out["details"]["description"] = det["description"]
    if (det.get("type") or {}).get("code") is not None:
        out["details"]["type"] = {"code": det["type"]["code"]}
    if ev.get("isPitch"):
        pdata = ev.get("pitchData") or {}
        out["pitchData"] = {
            **_pick(pdata, ["startSpeed", "extension", "zone"]),
            "coordinates": _pick(pdata.get("coordinates"), ["pX", "pZ"]),
            "breaks": _pick(pdata.get("breaks"), ["spinRate", "breakVerticalInduced", "breakHorizontal"]),
        }
    return out


def trim_play(play):
    m = play.get("matchup") or {}
    return {
        "count": _pick(play.get("count"), ["balls", "strikes", "outs"]),
        "matchup": {
            "batter": _pick(m.get("batter"), ["id", "fullName"]),
            "pitcher": _pick(m.get("pitcher"), ["id", "fullName"]),
            "batSide": _pick(m.get("batSide"), ["code"]),
            "pitchHand": _pick(m.get("pitchHand"), ["code"]),
        },
        "playEvents": [trim_event(e) for e in play.get("playEvents") or []],
    }


def trim_feed(feed):
    live_data = feed.get("liveData") or {}
    plays = live_data.get("plays") or {}
    out_plays = {}
    if plays.get("currentPlay"):
        out_plays["currentPlay"] = trim_play(plays["currentPlay"])
    all_plays = plays.get("allPlays") or []
    out_plays["allPlays"] = [trim_play(all_plays[-1])] if all_plays else []

    ls = live_data.get("linescore") or {}
    offense = ls.get("offense") or {}
    teams = ls.get("teams") or {}
    return {
        "_fixture_note": NOTE,
        "gamePk": feed.get("gamePk"),
        "liveData": {
            "plays": out_plays,
            "linescore": {
                "currentInning": ls.get("currentInning"),
                "offense": {b: _pick(offense[b], ["id"]) for b in ("first", "second", "third") if b in offense},
                "teams": {side: _pick(teams.get(side), ["runs"]) for side in ("away", "home")},
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_pk", type=int)
    ap.add_argument("--name", help="file name without .json (default: feed_<gamePk>)")
    args = ap.parse_args()

    trimmed = trim_feed(live.fetch_feed(args.game_pk))
    try:
        live.parse_snapshot(trimmed)
        name = args.name or f"feed_{args.game_pk}"
    except live.NoActivePlay:
        name = args.name or f"feed_pregame_{args.game_pk}"
        print("No play in this feed yet: saving as a pre-game fixture.")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"{name}.json"
    out.write_text(json.dumps(trimmed, indent=1))
    print(f"Saved {out} ({out.stat().st_size:,} bytes). Run: python -m pytest tests/test_live_contract.py")


if __name__ == "__main__":
    main()
