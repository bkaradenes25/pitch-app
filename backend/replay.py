"""
replay.py -- scores the model against pitches it has never seen.

Held-out set = postseason games + the final (test) slice of the regular season.
For every pitch we rebuild the game state BEFORE the pitch, ask the recommender
what it would call, then compare with what was actually thrown and how it turned out.
(recommender/train are imported lazily so db/evaluate can be tested without a model.)
"""
import functools
import json
import os
from pathlib import Path

import pandas as pd

import db

POSTSEASON = ("F", "D", "L", "W")
DEMO_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "demo_data"
ART = os.environ.get("ARTIFACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))


def _test_start():
    try:
        return pd.Timestamp(json.load(open(os.path.join(ART, "metrics.json")))["test_start"])
    except (OSError, KeyError):
        return None   # older model: postseason only


def _load_raw():
    """Full Statcast data if present, else the small bundled demo set (see make_demo_data.py)."""
    import train
    try:
        return train.load_pitches()
    except SystemExit:
        files = sorted(DEMO_DIR.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No Statcast data found (run get_data.py) and no bundled demo data.")
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


@functools.lru_cache(maxsize=1)
def heldout() -> pd.DataFrame:
    import train
    raw = _load_raw()
    parts = [train.prepare(raw, game_types=POSTSEASON)]
    ts = _test_start()
    if ts is not None:
        reg = train.prepare(raw, game_types=("R",))
        parts.append(reg[reg["game_date"] >= ts])
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]
                          ).reset_index(drop=True)


@functools.lru_cache(maxsize=64)
def game_pitches(game_pk):
    df = heldout()
    return df[df["game_pk"] == game_pk].reset_index(drop=True)


def _last(name):
    return str(name).split(",")[0]


def list_games():
    df = heldout()
    out = []
    for gk, g in df.groupby("game_pk", sort=False):
        top = g[g["inning_topbot"] == "Top"]["player_name"]   # home team pitching
        bot = g[g["inning_topbot"] == "Bot"]["player_name"]   # away team pitching
        away = _last(bot.iloc[0]) if len(bot) else "?"
        home = _last(top.iloc[0]) if len(top) else "?"
        post = g["game_type"].iloc[0] != "R"
        date = pd.Timestamp(g["game_date"].iloc[0])
        out.append({"game_pk": int(gk), "date": str(date.date()), "n_pitches": len(g),
                    "postseason": bool(post),
                    "label": f"{'[Postseason] ' if post else ''}{date:%b %d}: {away} @ {home} ({len(g)} pitches)"})
    return sorted(out, key=lambda x: (x["postseason"], x["date"]), reverse=True)


def state_of(row):
    return {"stand": row["stand"], "p_throws": row["p_throws"], "on_base": int(row["on_base"]),
            "balls": int(row["balls"]), "strikes": int(row["strikes"]),
            "outs_when_up": int(row["outs_when_up"]), "inning": int(row["inning"])}


def evaluate(row, rec):
    """Compare the model's ranked list with the pitch that was actually thrown."""
    recs = rec["recommendations"]
    types = [r["pitch_type"] for r in recs]
    actual = row["pitch_type"]
    rank = types.index(actual) + 1 if actual in types else None
    succ = row.get("success")
    return {
        "rec_pitch": types[0], "rec_prob": recs[0]["success_prob"],
        "actual_pitch": actual,
        "actual_prob": recs[rank - 1]["success_prob"] if rank else None,
        "actual_rank": rank,
        "matched": int(rank == 1),
        "top3": int(rank is not None and rank <= 3),
        "actual_success": None if succ is None or pd.isna(succ) else int(succ),
    }


def score_row(row, detail=False):
    """row: dict for one held-out pitch -> (recommendation, event dict)."""
    import recommender
    rec = recommender.recommend(int(row["pitcher"]), state_of(row), int(row["batter"]),
                                top_n=20, detail=detail)
    s = state_of(row)
    ev = {**evaluate(row, rec),
          "game_pk": row["game_pk"], "at_bat_number": row["at_bat_number"],
          "pitch_number": row["pitch_number"], "game_date": str(pd.Timestamp(row["game_date"]).date()),
          "pitcher_id": row["pitcher"], "pitcher_name": row["player_name"], "batter_id": row["batter"],
          "balls": s["balls"], "strikes": s["strikes"], "outs": s["outs_when_up"],
          "inning": s["inning"], "on_base": s["on_base"], "stand": s["stand"], "p_throws": s["p_throws"]}
    return rec, ev


def step(game_pk, index, run_id):
    row = game_pitches(game_pk).iloc[index].to_dict()
    rec, ev = score_row(row, detail=True)
    db.log_events(run_id, [ev])
    return {**rec, **{k: ev[k] for k in ("rec_pitch", "actual_pitch", "actual_success")},
            "matched": bool(ev["matched"]),
            "pitcher_name": row["player_name"],
            "state": state_of(row),
            "actual_x": float(row["plate_x"]), "actual_z": float(row["plate_z"]),
            "scoreboard": db.scoreboard(run_id=run_id)}


def log_live_pitch(run_id, prev, snap, game_pk):
    """Live mode: compare the call made BEFORE a pitch with the pitch that was then thrown."""
    n = len(snap["pitches_this_at_bat"])
    if (not prev or prev["batter"] != snap["batter_id"] or prev["pitcher"] != snap["pitcher_id"]
            or n != prev["n"] + 1):
        return
    actual = snap["pitches_this_at_bat"][-1].get("type")
    if not actual:
        return
    s = prev["state"]
    ev = {**evaluate({"pitch_type": actual}, prev["rec"]),
          "game_pk": game_pk, "pitcher_id": snap["pitcher_id"], "pitcher_name": snap["pitcher_name"],
          "batter_id": snap["batter_id"], "balls": s["balls"], "strikes": s["strikes"],
          "outs": s["outs_when_up"], "inning": s["inning"], "on_base": s["on_base"],
          "stand": s["stand"], "p_throws": s["p_throws"]}
    db.log_events(run_id, [ev])
