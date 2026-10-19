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

from shared import GRID_POINTS, MIN_PITCHER_SAMPLES, QUALITY_RAW, count_bucket, pitch_category

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


def _load_json(name, default):
    path = os.path.join(ART, name)
    return json.load(open(path)) if os.path.exists(path) else default


# Written by train.py / fetch_names.py. All optional: an older model simply has no such features.
BATTERS = _load_json("batter_profiles.json", {"league": {}, "batters": {}})
PITCH_Q = _load_json("pitch_profiles.json", {"pitchers": {}, "league": {}})
NAMES = _load_json("batter_names.json", {})
_QUALITY_KEYS = {name: name[3:] for name in QUALITY_RAW}      # pq_velo -> velo (the JSON key)


def list_pitchers():
    return sorted(({"pitcher_id": int(pid), "name": e["name"], "pitches": list(e["pitches"]),
                    "throws": e.get("throws")}
                   for pid, e in ARSENALS.items()), key=lambda p: p["name"])


def batter_name(batter_id):
    return NAMES.get(str(batter_id)) or f"Batter {batter_id}"


def list_batters():
    """Batters with enough pitches seen to have their own profile (everyone else = league average)."""
    return sorted(({"batter_id": int(b), "name": batter_name(b), "stand": e["stand"], "switch": e["switch"]}
                   for b, e in BATTERS["batters"].items()), key=lambda x: x["name"])


def batter_vs(batter_id, cat):
    """(whiff rate, xwOBA on contact) this batter shows against a pitch category. Unknown batter ->
    league rate for that category, which is exactly what a no-history hitter looks like in training."""
    e = BATTERS["batters"].get(str(batter_id)) if batter_id is not None else None
    src = (e or {}).get("cats", {}).get(cat) or BATTERS["league"].get(cat) or {}
    nan = float("nan")
    return (nan if src.get("whiff") is None else src["whiff"],
            nan if src.get("xwoba_con") is None else src["xwoba_con"])


def batter_info(batter_id):
    if batter_id is None:
        return None
    e = BATTERS["batters"].get(str(batter_id))
    return {"batter_id": int(batter_id), "name": batter_name(batter_id), "known": e is not None,
            "stand": e["stand"] if e else None, "switch": bool(e and e["switch"]),
            "vs": {cat: {"whiff": c["whiff"], "xwoba_con": c["xwoba_con"], "swings": c["swings"]}
                   for cat, c in (e["cats"] if e else {}).items()}}


def pitch_stuff(pitcher_id, pitch_type, p_throws):
    """This pitcher's average velo/spin/movement for the pitch type, else the league's for (type, hand).
    Same per-column fallback as training (train.quality_feature_values)."""
    own = PITCH_Q["pitchers"].get(str(pitcher_id), {}).get(pitch_type, {})
    lg = PITCH_Q["league"].get(f"{pitch_type}|{p_throws}", {})
    out, from_own = {}, False
    for name, key in _QUALITY_KEYS.items():
        v = own.get(key)
        from_own = from_own or v is not None
        out[name] = v if v is not None else lg.get(key)
    return out, ("pitcher" if from_own else "league" if any(v is not None for v in out.values()) else None)


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


def _matrix(state, p_cluster, h_cluster, pitch_types, xs, zs, feats=None):
    """Model input built directly in the training column layout (same as get_dummies + reindex).
    feats: {column: per-row array} for the batter-vs-pitch and pitch-quality columns (NaN = unknown)."""
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
    for name, values in (feats or {}).items():
        put(name, np.asarray(values, dtype=float))
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
    feats = {name: [] for name in ("bat_whiff_vs_cat", "bat_xwoba_vs_cat", *QUALITY_RAW)}
    stuff = {}
    for pt in arsenal:
        samples, source = pick_locations(pitcher_id, pt, bucket)
        if samples is None:
            samples = np.asarray(GRID_POINTS, dtype=float)
        n_rows = len(samples) + (len(GRID_POINTS) if detail else 0)
        meta[pt] = (len(samples), source)
        pts += [pt] * len(samples); xs += samples[:, 0].tolist(); zs += samples[:, 1].tolist()
        kind += [0] * len(samples)
        if detail:
            pts += [pt] * len(GRID_POINTS)
            xs += [g[0] for g in GRID_POINTS]; zs += [g[1] for g in GRID_POINTS]
            kind += [1] * len(GRID_POINTS)
        # the same value for every location of this pitch type: who is batting, what this pitch looks like
        whiff, xw = batter_vs(batter_id, pitch_category(pt))
        stuff[pt] = pitch_stuff(pitcher_id, pt, state["p_throws"])
        for name, v in (("bat_whiff_vs_cat", whiff), ("bat_xwoba_vs_cat", xw), *stuff[pt][0].items()):
            feats[name] += [float("nan") if v is None else v] * n_rows

    probs = MODEL.predict_proba(_matrix(state, p_cluster, h_cluster, pts, xs, zs, feats))[:, 1]
    pts, xs, zs, kind = np.array(pts), np.array(xs), np.array(zs), np.array(kind)

    results = []
    for pt in arsenal:
        m = pts == pt
        q, q_src = stuff[pt]
        r = {"pitch_type": pt, "success_prob": round(float(probs[m & (kind == 0)].mean()), 4),
             "n_samples": meta[pt][0], "locations_source": meta[pt][1],
             "stuff": None if q_src is None else {
                 "velo": q["pq_velo"], "spin": q["pq_spin"], "ext": q["pq_ext"], "source": q_src,
                 # Statcast pfx_* are in feet; show inches
                 "ivb_in": None if q["pq_pfx_z"] is None else round(q["pq_pfx_z"] * 12, 1),
                 "hb_in": None if q["pq_pfx_x"] is None else round(q["pq_pfx_x"] * 12, 1)}}
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
            "hitter_cluster": h_cluster, "count_bucket": bucket, "batter": batter_info(batter_id),
            "recommendations": results[:top_n]}
