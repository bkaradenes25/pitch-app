"""Shared fixtures: a tiny synthetic model + artifacts so tests never need real data."""
import copy
import importlib
import json
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

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
        "cluster": rng.integers(0, 4, n), "hitter_cluster": rng.integers(-1, 5, n)})
    enc = pd.get_dummies(raw, columns=["pitch_type", "stand", "p_throws"]).astype(float)
    model = GradientBoostingClassifier(n_estimators=15, random_state=0).fit(enc, rng.integers(0, 2, n))
    joblib.dump(model, d / "model.pkl")
    json.dump(list(enc.columns), open(d / "x_columns.json", "w"))
    json.dump({"1": {"name": "Pitcher, Test", "cluster": 2, "pitches": {"FF": 100, "SL": 80, "CH": 40}},
               "2": {"name": "Arm, Another", "cluster": 1, "pitches": {"FF": 50}}},
              open(d / "arsenals.json", "w"))
    json.dump({"7": 3}, open(d / "hitter_clusters.json", "w"))
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
