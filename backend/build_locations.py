"""
build_locations.py -- samples where each pitcher REALLY throws each pitch (training period only).

    python build_locations.py      # after get_data.py and train.py; takes a minute or two

Writes artifacts/locations.json. No model retraining needed.
Structure: pitchers[id][pitch_type][ahead|even|behind|all] and league[pitch_type][...] = [[x, z], ...]
"""
import json
import os

import numpy as np
import pandas as pd

import train
from shared import MIN_PITCHER_SAMPLES

PITCHER_SAMPLE, LEAGUE_SAMPLE = 30, 150
rng = np.random.default_rng(42)


def take(frame, n):
    pts = frame[["plate_x", "plate_z"]].to_numpy(dtype=float)   # columns can load as object dtype
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return np.round(pts, 2).tolist()


def buckets(frame, n, min_n):
    out = {}
    for name, sub in (("all", frame),
                      ("ahead", frame[frame["strikes"] > frame["balls"]]),
                      ("behind", frame[frame["balls"] > frame["strikes"]]),
                      ("even", frame[frame["balls"] == frame["strikes"]])):
        if len(sub) >= min_n:
            out[name] = take(sub, n)
    return out


def main():
    df = train.prepare(train.load_pitches())
    try:
        end = pd.Timestamp(json.load(open(f"{train.ART}/metrics.json"))["train_end"])
        df = df[df["game_date"] < end]
        print(f"Using training-period pitches only (before {end.date()}): {len(df):,}")
    except (OSError, KeyError):
        print("metrics.json not found: using all regular-season pitches (re-run after train.py).")

    pitchers = {}
    for (pid, pt), g in df.groupby(["pitcher", "pitch_type"]):
        b = buckets(g, PITCHER_SAMPLE, MIN_PITCHER_SAMPLES)
        if "all" in b:
            pitchers.setdefault(str(int(pid)), {})[pt] = b
    league = {pt: buckets(g, LEAGUE_SAMPLE, 1) for pt, g in df.groupby("pitch_type")}

    path = f"{train.ART}/locations.json"
    json.dump({"pitchers": pitchers, "league": league}, open(path, "w"), separators=(",", ":"))
    print(f"Saved {path}: {len(pitchers)} pitchers, {len(league)} pitch types "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
