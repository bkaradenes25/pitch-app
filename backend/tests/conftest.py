"""Shared fixtures: a tiny synthetic model + artifacts so tests never need real data."""
import copy
import importlib
import json
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

FEED = {  # minimal MLB Stats API live-feed shape used by live.parse_snapshot
    "liveData": {
        "plays": {"currentPlay": {
            "count": {"balls": 1, "strikes": 2, "outs": 1},
            "matchup": {"pitcher": {"id": 1, "fullName": "Test Pitcher"},
                        "batter": {"id": 7, "fullName": "Test Batter"},
                        "batSide": {"code": "L"}, "pitchHand": {"code": "R"}},
            "playEvents": [
                {"isPitch": True, "details": {"type": {"code": "FF"}, "description": "Ball"},
                 "pitchData": {"coordinates": {"pX": 0.1, "pZ": 2.5}}},
                {"isPitch": False, "details": {}},
            ]}},
        "linescore": {"currentInning": 4, "offense": {"first": {"id": 9}},
                      "teams": {"away": {"runs": 2}, "home": {"runs": 1}}},
    }
}


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(0)
    n = 800
    raw = pd.DataFrame({
        "pitch_type": rng.choice(["FF", "SL", "CH"], n), "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n), "on_base": rng.integers(0, 2, n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
        "outs_when_up": rng.integers(0, 3, n), "inning": rng.integers(1, 10, n),
        "plate_x": rng.normal(0, .7, n), "plate_z": rng.normal(2.5, .6, n),
        "cluster": rng.integers(0, 4, n), "hitter_cluster": rng.integers(-1, 5, n),
        "bat_whiff_vs_cat": rng.uniform(.1, .4, n), "bat_xwoba_vs_cat": rng.uniform(.25, .5, n),
        "pq_velo": rng.normal(90, 5, n), "pq_spin": rng.normal(2300, 200, n),
        "pq_pfx_x": rng.normal(0, .8, n), "pq_pfx_z": rng.normal(.8, .5, n), "pq_ext": rng.normal(6.4, .3, n)})
    enc = pd.get_dummies(raw, columns=["pitch_type", "stand", "p_throws"]).astype(float)
    # HistGradientBoosting accepts NaN, like XGBoost does for the "unknown" feature values
    model = HistGradientBoostingClassifier(max_iter=15, random_state=0).fit(enc, rng.integers(0, 2, n))
    joblib.dump(model, d / "model.pkl")
    json.dump(list(enc.columns), open(d / "x_columns.json", "w"))
    json.dump({"1": {"name": "Pitcher, Test", "cluster": 2, "throws": "R",
                     "pitches": {"FF": 100, "SL": 80, "CH": 40}},
               "2": {"name": "Arm, Another", "cluster": 1, "pitches": {"FF": 50}}},      # no 'throws' (old file)
              open(d / "arsenals.json", "w"))
    json.dump({"7": 3}, open(d / "hitter_clusters.json", "w"))
    cats = ["fastballs", "breaking_balls", "offspeed", "other"]
    cat = lambda w, x: {"whiff": w, "xwoba_con": x, "swings": 300, "bip": 120}
    json.dump({"league": {c: {"whiff": .25, "xwoba_con": .35} for c in cats},
               "batters": {"7": {"stand": "L", "switch": False, "n": 900,
                                 "cats": {"fastballs": cat(.18, .42), "breaking_balls": cat(.40, .30),
                                          "offspeed": cat(.30, .33)}},        # no 'other' -> league value
                           "8": {"stand": "R", "switch": True, "n": 500, "cats": {}}}},
              open(d / "batter_profiles.json", "w"))
    json.dump({"7": "Test Batter"}, open(d / "batter_names.json", "w"))          # batter 8 has no name
    q = lambda v: {"velo": v, "spin": 2300.0, "pfx_x": -0.5, "pfx_z": 1.0, "ext": 6.4}
    json.dump({"pitchers": {"1": {"FF": {**q(95.1), "pfx_z": 1.3, "n": 100},
                                  "SL": {**q(85.0), "ext": None, "n": 80}}},      # SL extension unknown
               "league": {f"{pt}|{h}": q(v) for pt, v in (("FF", 93), ("SI", 92), ("SL", 84), ("CH", 84), ("CU", 79))
                          for h in ("L", "R")}},
              open(d / "pitch_profiles.json", "w"))
    loc = lambda n, cx, cz: np.round(np.column_stack(
        [rng.normal(cx, .3, n), rng.normal(cz, .5, n)]), 2).tolist()
    json.dump({"pitchers": {"1": {"FF": {"ahead": loc(30, 0, 2.5), "all": loc(60, 0, 2.5)},
                                  "SL": {"all": loc(40, 0.3, 1.8)}}},
               "league": {"FF": {"all": loc(100, 0, 2.5)}}}, open(d / "locations.json", "w"))
    return d


@pytest.fixture()
def modules(artifacts_dir, tmp_path, monkeypatch):
    """Fresh DB per test; app modules reloaded so they pick up the test env vars."""
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts_dir))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)          # no ./data folder here
    import db, recommender, replay
    for m in (db, recommender, replay):
        importlib.reload(m)
    db.init_db()
    return types.SimpleNamespace(db=db, recommender=recommender, replay=replay)


@pytest.fixture()
def fake_feed():
    return copy.deepcopy(FEED)


@pytest.fixture()
def state():
    return dict(stand="L", p_throws="R", on_base=1, balls=1, strikes=2, outs_when_up=1, inning=4)
