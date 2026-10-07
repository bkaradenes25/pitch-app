"""
train.py -- builds every artifact the app needs from data/statcast_*.parquet.

    python get_data.py     # once
    python train.py

Design choices (good talking points):
  * all pitchers (no hardcoded list); low-volume pitchers/hitters get cluster -1
  * regular season only for training -> POSTSEASON is a true out-of-sample demo set
  * chronological 70/15/15 split by game date
  * pitcher AND hitter profiles are built from TRAIN-period pitches only (no leakage)
  * hitter profiles are derived from the same Statcast data (no extra downloads)
  * balls/strikes are separate features; success label = weak contact or strike
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ART = "artifacts"
DATA_DIR = Path("data")
MIN_PITCHER_PITCHES = 300   # min train pitches to be clustered (else cluster -1)
MIN_ARSENAL_PITCHES = 30    # min pitches of a type to be in a pitcher's arsenal
MIN_PA = 100                # min train PAs for a hitter profile (else cluster -1)
SEARCH_ROWS = 200_000       # subsample size for hyperparameter search

PITCH_CATS = ["fastballs", "breaking_balls", "offspeed", "other"]
FASTBALLS = ["FF", "FT", "SI", "FC", "FA"]
BREAKING = ["SL", "CU", "KC", "KN", "ST", "SB", "SV", "CS"]
OFFSPEED = ["FS", "CH", "EP", "SC"]

WHIFFS = ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
SWINGS = WHIFFS + ["foul", "hit_into_play", "hit_into_play_no_out",
                   "hit_into_play_score", "foul_bunt", "missed_bunt"]
HITTER_BASE = ["k_pct", "bb_pct", "xwoba_con", "whiff_pct", "hard_hit_pct",
               "barrel_pct", "avg_ev", "avg_la"]
HITTER_OPTIONAL = ["attack_angle", "attack_direction", "bat_speed"]  # bat tracking

CATEGORICAL = ["pitch_type", "stand", "p_throws"]
FEATURES = ["pitch_type", "stand", "p_throws", "on_base", "balls", "strikes",
            "outs_when_up", "inning", "plate_x", "plate_z", "cluster", "hitter_cluster"]
BASE_FEATURES = [f for f in FEATURES if f not in ("cluster", "hitter_cluster")]


def load_pitches() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("statcast_*.parquet"))
    if not files:
        raise SystemExit("No data found. Run: python get_data.py")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def prepare(df: pd.DataFrame, game_types=("R",)) -> pd.DataFrame:
    df = df[df["game_type"].isin(game_types)].copy()   # default: regular season only
    xw = df["estimated_woba_using_speedangle"]
    df = df[~((df["type"] == "X") & xw.isna())]     # unlabeled balls in play
    xw = df["estimated_woba_using_speedangle"]
    df["success"] = ((xw.notna() & (xw <= 0.300))
                     | df["description"].isin(["called_strike", "swinging_strike"])).astype(int)
    df["pitch_cat"] = np.select(
        [df["pitch_type"].isin(FASTBALLS), df["pitch_type"].isin(BREAKING),
         df["pitch_type"].isin(OFFSPEED)],
        ["fastballs", "breaking_balls", "offspeed"], default="other")
    df["on_base"] = df[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=BASE_FEATURES + ["success"])
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]
                          ).reset_index(drop=True)


def cluster_pitchers(train_df, k=4):
    counts = train_df.groupby("pitcher").size()
    keep = counts[counts >= MIN_PITCHER_PITCHES].index
    d = train_df[train_df["pitcher"].isin(keep)]
    mix = (d.groupby(["pitcher", "pitch_cat"]).size().unstack(fill_value=0)
           .reindex(columns=PITCH_CATS, fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(mix.values)
    return km, {int(p): int(c) for p, c in zip(mix.index, km.labels_)}


def hitter_profile(d: pd.DataFrame) -> pd.DataFrame:
    """Per-batter profile derived from pitch-level data (train period only)."""
    d = d.assign(_whiff=d["description"].isin(WHIFFS), _swing=d["description"].isin(SWINGS))
    g = d.groupby("batter")
    prof = pd.DataFrame({"whiff_pct": g["_whiff"].sum() / g["_swing"].sum().replace(0, np.nan)})

    gp = d[d["events"].notna()].groupby("batter")  # events only on the PA-ending pitch
    prof["pa"] = gp.size()
    prof["k_pct"] = gp["events"].apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).mean())
    prof["bb_pct"] = gp["events"].apply(lambda e: e.isin(["walk", "intent_walk"]).mean())

    gb = d[d["launch_speed"].notna()].groupby("batter")  # batted balls
    prof["xwoba_con"] = gb["estimated_woba_using_speedangle"].mean()
    prof["avg_ev"] = gb["launch_speed"].mean()
    prof["avg_la"] = gb["launch_angle"].mean()
    prof["hard_hit_pct"] = gb["launch_speed"].apply(lambda s: (s >= 95).mean())
    prof["barrel_pct"] = gb["launch_speed_angle"].apply(lambda s: (s == 6).mean())
    for c in HITTER_OPTIONAL:
        if c in d.columns:
            prof[c] = g[c].mean()
    return prof[prof["pa"] >= MIN_PA]


def cluster_hitters(train_df, k=5):
    prof = hitter_profile(train_df)
    feats = HITTER_BASE + [c for c in HITTER_OPTIONAL if c in prof.columns]
    prof = prof.dropna(subset=HITTER_BASE)
    X = prof[feats].fillna(prof[feats].median())
    scaler = StandardScaler()
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaler.fit_transform(X))
    return scaler, km, feats, {int(b): int(c) for b, c in zip(prof.index, km.labels_)}


def score(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)), "base_rate": float(y.mean())}


def main():
    os.makedirs(ART, exist_ok=True)
    df = prepare(load_pitches())
    print(f"{len(df):,} regular-season pitches, {df['pitcher'].nunique()} pitchers, "
          f"{df['batter'].nunique()} batters")

    dates = np.sort(df["game_date"].unique())
    t1, t2 = dates[int(len(dates) * 0.70)], dates[int(len(dates) * 0.85)]
    train_m = df["game_date"] < t1
    val_m = (df["game_date"] >= t1) & (df["game_date"] < t2)
    test_m = df["game_date"] >= t2

    pitcher_km, pitcher_map = cluster_pitchers(df[train_m])
    scaler, hitter_km, hitter_feats, hitter_map = cluster_hitters(df[train_m])
    df["cluster"] = df["pitcher"].map(pitcher_map).fillna(-1)
    df["hitter_cluster"] = df["batter"].map(hitter_map).fillna(-1)
    print(f"clustered {len(pitcher_map)} pitchers, {len(hitter_map)} hitters (rest = -1)")

    enc = pd.get_dummies(df[FEATURES], columns=CATEGORICAL).astype(float)
    x_columns, y = list(enc.columns), df["success"]
    X_tr, y_tr = enc[train_m], y[train_m]

    # tune on an ordered subsample (fast), then refit best params on all train data
    step = max(1, len(X_tr) // SEARCH_ROWS)
    search = RandomizedSearchCV(
        XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1),
        {"max_depth": randint(4, 10), "learning_rate": loguniform(0.02, 0.3),
         "n_estimators": randint(100, 400), "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.7, 0.85, 1.0]},
        n_iter=12, scoring="roc_auc", cv=TimeSeriesSplit(n_splits=3),
        random_state=42, verbose=1)
    search.fit(X_tr.iloc[::step], y_tr.iloc[::step])
    best = XGBClassifier(**search.best_params_, tree_method="hist", eval_metric="logloss",
                         random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    metrics = {
        "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else
                            int(v) if isinstance(v, (np.integer, int)) else v)
                        for k, v in search.best_params_.items()},
        "cv_auc": float(search.best_score_),
        "val": score(best, enc[val_m], y[val_m]),
        "test": score(best, enc[test_m], y[test_m]),
        "test_baseline_brier": float(brier_score_loss(
            y[test_m], np.full(int(test_m.sum()), y_tr.mean()))),
        "train_end": str(pd.Timestamp(t1).date()),
        "test_start": str(pd.Timestamp(t2).date()),
        "data_through": str(df["game_date"].max().date()),
        "note": "Postseason games are excluded from training: use them as the out-of-sample demo set.",
    }
    print(json.dumps(metrics, indent=2))

    arsenals = {}
    for (pid, name, pt), n in df.groupby(["pitcher", "player_name", "pitch_type"]).size().items():
        if n >= MIN_ARSENAL_PITCHES:
            e = arsenals.setdefault(str(int(pid)), {
                "name": name, "cluster": pitcher_map.get(int(pid), -1), "pitches": {}})
            e["pitches"][pt] = int(n)

    joblib.dump(best, f"{ART}/model.pkl")
    joblib.dump({"pitcher_kmeans": pitcher_km, "hitter_kmeans": hitter_km,
                 "hitter_scaler": scaler, "pitch_cats": PITCH_CATS,
                 "hitter_features": hitter_feats}, f"{ART}/cluster_models.pkl")
    json.dump(x_columns, open(f"{ART}/x_columns.json", "w"))
    json.dump(arsenals, open(f"{ART}/arsenals.json", "w"), indent=1)
    json.dump({str(k): v for k, v in hitter_map.items()}, open(f"{ART}/hitter_clusters.json", "w"))
    json.dump(metrics, open(f"{ART}/metrics.json", "w"), indent=2)
    print("Saved artifacts to", ART)


if __name__ == "__main__":
    main()
