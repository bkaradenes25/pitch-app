"""
recommender.py -- ranks the pitch types a pitcher throws.

success_prob (the ranking score) = the model's AVERAGE success probability if the pitch is thrown
to locations sampled from where THIS pitcher really locates it in this kind of count
(ahead / even / behind). That is command-realistic and avoids the "best cell of a grid" bias.

The heatmap grid (including out-of-zone cells) answers a different question:
"how likely is success if the pitch is located at exactly this spot?"
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from shared import GRID_POINTS, MIN_PITCHER_SAMPLES, count_bucket

ART = os.environ.get("ARTIFACT_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))

MODEL = joblib.load(os.path.join(ART, "model.pkl"))
X_COLUMNS = json.load(open(os.path.join(ART, "x_columns.json")))
ARSENALS = json.load(open(os.path.join(ART, "arsenals.json")))
HITTER_CLUSTERS = json.load(open(os.path.join(ART, "hitter_clusters.json")))

REQUIRED_STATE = ["stand", "p_throws", "on_base", "balls", "strikes", "outs_when_up", "inning"]
DEFAULT_ARSENAL = ["FF", "SI", "SL", "CH", "CU"]   # for pitchers not in training data
_COL = {c: i for i, c in enumerate(X_COLUMNS)}


def _load_locations():
    """Real pitch locations from the training period (build_locations.py). Optional."""
    path = os.path.join(ART, "locations.json")
    if not os.path.exists(path):
        return {"pitchers": {}, "league": {}}
    raw = json.load(open(path))
    conv = lambda d: {pt: {b: np.asarray(v, dtype=float).reshape(-1, 2) for b, v in bk.items()}
                      for pt, bk in d.items()}
    return {"pitchers": {pid: conv(d) for pid, d in raw["pitchers"].items()},
            "league": conv(raw["league"])}


LOCATIONS = _load_locations()


def list_pitchers():
    return sorted(({"pitcher_id": int(pid), "name": e["name"], "pitches": list(e["pitches"])}
                   for pid, e in ARSENALS.items()), key=lambda p: p["name"])


def pitcher_name(pitcher_id):
    e = ARSENALS.get(str(pitcher_id))
    return e["name"] if e else f"Pitcher {pitcher_id}"


def pick_locations(pitcher_id, pitch_type, bucket):
    """Most specific location sample available: pitcher+count -> pitcher -> league+count -> league."""
    mine = LOCATIONS["pitchers"].get(str(pitcher_id), {}).get(pitch_type, {})
    for key, tag in ((bucket, "pitcher_count"), ("all", "pitcher")):
        a = mine.get(key)
        if a is not None and len(a) >= MIN_PITCHER_SAMPLES:
            return a, tag
    league = LOCATIONS["league"].get(pitch_type, {})
    for key, tag in ((bucket, "league_count"), ("all", "league")):
        a = league.get(key)
        if a is not None and len(a) > 0:
            return a, tag
    return None, "grid"


def _matrix(state, p_cluster, h_cluster, pitch_types, xs, zs):
    """Model input built directly in the training column layout (same as get_dummies + reindex)."""
    xs, zs, pts = np.asarray(xs, float), np.asarray(zs, float), np.asarray(pitch_types)
    M = np.zeros((len(xs), len(X_COLUMNS)))

    def put(name, value):
        i = _COL.get(name)
        if i is not None:
            M[:, i] = value

    for k in ("on_base", "balls", "strikes", "outs_when_up", "inning"):
        put(k, state[k])
    put("cluster", p_cluster)
    put("hitter_cluster", h_cluster)
    put("plate_x", xs)
    put("plate_z", zs)
    put(f"stand_{state['stand']}", 1.0)
    put(f"p_throws_{state['p_throws']}", 1.0)
    for pt in set(pts.tolist()):
        i = _COL.get(f"pitch_type_{pt}")
        if i is not None:
            M[pts == pt, i] = 1.0
    return pd.DataFrame(M, columns=X_COLUMNS)


def recommend(pitcher_id, state, batter_id=None, top_n=5, detail=True):
    """detail=False skips the heatmap grid (used by the backtest for speed)."""
    missing = [k for k in REQUIRED_STATE if k not in state]
    if missing:   # never silently zero-fill a missing feature
        raise ValueError(f"state is missing required fields: {missing}")

    entry = ARSENALS.get(str(pitcher_id))
    arsenal = list(entry["pitches"]) if entry else DEFAULT_ARSENAL
    p_cluster = entry["cluster"] if entry else -1
    h_cluster = HITTER_CLUSTERS.get(str(batter_id), -1) if batter_id is not None else -1
    bucket = count_bucket(state["balls"], state["strikes"])

    pts, xs, zs, kind, meta = [], [], [], [], {}
    for pt in arsenal:
        samples, source = pick_locations(pitcher_id, pt, bucket)
        if samples is None:
            samples = np.asarray(GRID_POINTS, dtype=float)
        meta[pt] = (len(samples), source)
        pts += [pt] * len(samples); xs += samples[:, 0].tolist(); zs += samples[:, 1].tolist()
        kind += [0] * len(samples)
        if detail:
            pts += [pt] * len(GRID_POINTS)
            xs += [g[0] for g in GRID_POINTS]; zs += [g[1] for g in GRID_POINTS]
            kind += [1] * len(GRID_POINTS)

    probs = MODEL.predict_proba(_matrix(state, p_cluster, h_cluster, pts, xs, zs))[:, 1]
    pts, xs, zs, kind = np.array(pts), np.array(xs), np.array(zs), np.array(kind)

    results = []
    for pt in arsenal:
        m = pts == pt
        r = {"pitch_type": pt, "success_prob": round(float(probs[m & (kind == 0)].mean()), 4),
             "n_samples": meta[pt][0], "locations_source": meta[pt][1]}
        if detail:
            g = m & (kind == 1)
            gp, gx, gz = probs[g], xs[g], zs[g]
            j = int(gp.argmax())
            r["best_location"] = {"plate_x": float(gx[j]), "plate_z": float(gz[j])}
            r["best_cell_prob"] = round(float(gp[j]), 4)
            r["grid"] = [{"plate_x": float(x), "plate_z": float(z), "prob": round(float(p), 4)}
                         for x, z, p in zip(gx, gz, gp)]
        results.append(r)
    results.sort(key=lambda r: -r["success_prob"])
    return {"pitcher_id": pitcher_id, "pitcher_name": pitcher_name(pitcher_id),
            "used_default_arsenal": entry is None, "pitcher_cluster": p_cluster,
            "hitter_cluster": h_cluster, "count_bucket": bucket, "recommendations": results[:top_n]}
