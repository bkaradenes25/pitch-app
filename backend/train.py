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
  * batter-vs-pitch-category features (whiff rate, xwOBA on contact) are shrunk toward the league and
    computed OUT-OF-FOLD for training rows (5 folds by game): a row's feature comes from other games only.
    (Leave-one-out looks similar but leaks: within a group the feature equals (total - own outcome)/(n-1),
    so the model can read the row's own outcome from it. That version scored worse on the test set.)
  * pitch-quality features (velocity, spin, movement, extension) are this pitcher's train-period
    averages for that pitch type, falling back to the league average for (type, hand)
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

from shared import (BATTER_FEATURES, BREAKING, FASTBALLS, MIN_BATTER_PITCHES, MIN_QUALITY_PITCHES,  # noqa: F401
                    OFFSPEED, PITCH_CATS, PRIOR_BATTED_BALLS, PRIOR_SWINGS, QUALITY_FEATURES,
                    QUALITY_RAW, shrink)

ART = "artifacts"
DATA_DIR = Path("data")
MIN_PITCHER_PITCHES = 300   # min train pitches to be clustered (else cluster -1)
MIN_ARSENAL_PITCHES = 30    # min pitches of a type to be in a pitcher's arsenal
MIN_PA = 100                # min train PAs for a hitter profile (else cluster -1)
SEARCH_ROWS = 200_000       # subsample size for hyperparameter search

WHIFFS = ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
SWINGS = WHIFFS + ["foul", "hit_into_play", "hit_into_play_no_out",
                   "hit_into_play_score", "foul_bunt", "missed_bunt"]
HITTER_BASE = ["k_pct", "bb_pct", "xwoba_con", "whiff_pct", "hard_hit_pct",
               "barrel_pct", "avg_ev", "avg_la"]
HITTER_OPTIONAL = ["attack_angle", "attack_direction", "bat_speed"]  # bat tracking

CATEGORICAL = ["pitch_type", "stand", "p_throws"]
BASE_FEATURES = ["pitch_type", "stand", "p_throws", "on_base", "balls", "strikes",
                 "outs_when_up", "inning", "plate_x", "plate_z"]      # straight from the raw data
CLUSTER_FEATURES = ["cluster", "hitter_cluster"]
FEATURES = BASE_FEATURES + CLUSTER_FEATURES + BATTER_FEATURES + QUALITY_FEATURES
COUNT_COLS = ["swings", "whiffs", "bip", "xw_sum"]
N_FOLDS = 5


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


def _num(x, nd=4):
    """JSON-safe rounded float (NaN -> None)."""
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _own_counts(df):
    """What each pitch itself contributes to the batter count table."""
    xw = df["estimated_woba_using_speedangle"]
    return {"swings": df["description"].isin(SWINGS).to_numpy(dtype=float),
            "whiffs": df["description"].isin(WHIFFS).to_numpy(dtype=float),
            "bip": xw.notna().to_numpy(dtype=float),
            "xw_sum": xw.fillna(0.0).to_numpy(dtype=float)}


def batter_tables(d):
    """Per-(batter, pitch category) swing / whiff / batted-ball counts from train-period pitches,
    plus league rates per category (the shrinkage targets)."""
    t = pd.DataFrame({"batter": d["batter"].to_numpy(), "cat": d["pitch_cat"].to_numpy(),
                      **_own_counts(d)})
    tab = t.groupby(["batter", "cat"])[COUNT_COLS].sum()
    lg = t.groupby("cat")[COUNT_COLS].sum()
    rates = pd.DataFrame({"whiff": lg["whiffs"] / lg["swings"].replace(0, np.nan),
                          "xwoba_con": lg["xw_sum"] / lg["bip"].replace(0, np.nan)})
    return tab, rates


def batter_feature_values(df, tab, rates):
    """Shrunk batter-vs-category features for every row of df, from the counts in `tab`."""
    idx = pd.MultiIndex.from_arrays([df["batter"].to_numpy(), df["pitch_cat"].to_numpy()])
    t = tab.reindex(idx)
    n = {c: t[c].fillna(0.0).to_numpy(dtype=float) for c in COUNT_COLS}
    lw = df["pitch_cat"].map(rates["whiff"]).to_numpy(dtype=float)
    lx = df["pitch_cat"].map(rates["xwoba_con"]).to_numpy(dtype=float)
    return {"bat_whiff_vs_cat": shrink(n["whiffs"], n["swings"], lw, PRIOR_SWINGS),
            "bat_xwoba_vs_cat": shrink(n["xw_sum"], n["bip"], lx, PRIOR_BATTED_BALLS)}


def batter_feature_values_oof(train_df, n_folds=N_FOLDS):
    """Training-row features computed out-of-fold: games are split into folds and each fold's rows get
    their values from the OTHER folds' games. No row's feature depends on its own game's outcomes, and
    (unlike leave-one-out) the feature carries no trace of the row's own outcome."""
    fold = (pd.factorize(train_df["game_pk"])[0] % n_folds)
    out = {k: np.full(len(train_df), np.nan) for k in BATTER_FEATURES}
    for f in range(n_folds):
        mask = fold == f
        tab, rates = batter_tables(train_df[~mask])
        for k, v in batter_feature_values(train_df[mask], tab, rates).items():
            out[k][mask] = v
    return out


def pitch_quality_tables(d):
    """Train-period mean velocity / spin / movement / extension per (pitcher, pitch type), and per
    (pitch type, handedness) for the fallback."""
    raw = list(QUALITY_RAW.values())
    g = d.groupby(["pitcher", "pitch_type"])
    own = g[raw].mean()[g.size() >= MIN_QUALITY_PITCHES]
    g2 = d.groupby(["pitch_type", "p_throws"])
    lg = g2[raw].mean()[g2.size() >= MIN_QUALITY_PITCHES]
    return own, lg


def quality_feature_values(df, own, lg):
    raw = list(QUALITY_RAW.values())
    a = own.reindex(pd.MultiIndex.from_arrays(
        [df["pitcher"].to_numpy(), df["pitch_type"].to_numpy()]))[raw].to_numpy(dtype=float)
    b = lg.reindex(pd.MultiIndex.from_arrays(
        [df["pitch_type"].to_numpy(), df["p_throws"].to_numpy()]))[raw].to_numpy(dtype=float)
    vals = np.where(np.isnan(a), b, a)                       # per column: own value, else league
    return {name: vals[:, i] for i, name in enumerate(QUALITY_RAW)}


def add_features(df, train_m):
    """Adds every model feature that is derived from train-period data (shared by train.py and
    ablation.py so the two can never disagree). Returns (df, ctx) where ctx holds the fitted pieces."""
    missing = [c for c in QUALITY_RAW.values() if c not in df.columns]
    if missing:
        raise SystemExit(f"The Statcast files are missing {missing}. Re-run: python get_data.py "
                         "(it re-downloads files that lack these columns).")
    train_df = df[train_m]
    pitcher_km, pitcher_map = cluster_pitchers(train_df)
    scaler, hitter_km, hitter_feats, hitter_map = cluster_hitters(train_df)
    df["cluster"] = df["pitcher"].map(pitcher_map).fillna(-1)
    df["hitter_cluster"] = df["batter"].map(hitter_map).fillna(-1)

    b_tab, b_rates = batter_tables(train_df)
    for k, v in batter_feature_values(df, b_tab, b_rates).items():        # validation/test rows: full table
        df[k] = v
    for k, v in batter_feature_values_oof(train_df).items():              # training rows: out-of-fold
        df.loc[train_m.to_numpy(), k] = v
    q_own, q_lg = pitch_quality_tables(train_df)
    for k, v in quality_feature_values(df, q_own, q_lg).items():
        df[k] = v
    ctx = {"pitcher_km": pitcher_km, "pitcher_map": pitcher_map, "scaler": scaler,
           "hitter_km": hitter_km, "hitter_feats": hitter_feats, "hitter_map": hitter_map,
           "batter_tab": b_tab, "batter_rates": b_rates, "quality_own": q_own, "quality_league": q_lg}
    return df, ctx


def batter_profiles_json(train_df, tab, rates):
    """What the recommender needs about each batter (already shrunk, so serving needs no math)."""
    n = train_df.groupby("batter").size()
    share_l = train_df.groupby("batter")["stand"].apply(lambda s: float((s == "L").mean()))
    rows = tab.to_dict("index")
    league = {cat: {"whiff": _num(rates.loc[cat, "whiff"], 5), "xwoba_con": _num(rates.loc[cat, "xwoba_con"], 5)}
              for cat in rates.index}
    batters = {}
    for b in n.index[n >= MIN_BATTER_PITCHES]:
        cats = {}
        for cat in PITCH_CATS:
            if cat not in rates.index:
                continue
            c = rows.get((b, cat), dict.fromkeys(COUNT_COLS, 0.0))
            cats[cat] = {"whiff": _num(shrink(c["whiffs"], c["swings"], rates.loc[cat, "whiff"], PRIOR_SWINGS), 5),
                         "xwoba_con": _num(shrink(c["xw_sum"], c["bip"], rates.loc[cat, "xwoba_con"],
                                                  PRIOR_BATTED_BALLS), 5),
                         "swings": int(c["swings"]), "bip": int(c["bip"])}
        left = share_l[b]
        batters[str(int(b))] = {"stand": "L" if left >= 0.5 else "R", "switch": bool(0.15 < left < 0.85),
                                "n": int(n[b]), "cats": cats}
    return {"league": league, "batters": batters}


def pitch_profiles_json(train_df, own, lg):
    cnt = train_df.groupby(["pitcher", "pitch_type"]).size()
    short = {k[3:]: v for k, v in QUALITY_RAW.items()}            # pq_velo -> velo
    pitchers = {}
    for (pid, pt), row in own.iterrows():
        pitchers.setdefault(str(int(pid)), {})[pt] = {
            **{name: _num(row[col], 3) for name, col in short.items()}, "n": int(cnt[(pid, pt)])}
    league = {f"{pt}|{hand}": {name: _num(row[col], 3) for name, col in short.items()}
              for (pt, hand), row in lg.iterrows()}
    return {"pitchers": pitchers, "league": league}


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

    df, ctx = add_features(df, train_m)
    pitcher_map, hitter_map = ctx["pitcher_map"], ctx["hitter_map"]
    print(f"clustered {len(pitcher_map)} pitchers, {len(hitter_map)} hitters (rest = -1); "
          f"batter-vs-pitch and pitch-quality features added")

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
        "features": FEATURES,
        "note": "Postseason games are excluded from training: use them as the out-of-sample demo set.",
    }
    print(json.dumps(metrics, indent=2))

    throws = df.groupby("pitcher")["p_throws"].agg(lambda x: x.mode().iloc[0])
    arsenals = {}
    for (pid, name, pt), n in df.groupby(["pitcher", "player_name", "pitch_type"]).size().items():
        if n >= MIN_ARSENAL_PITCHES:
            e = arsenals.setdefault(str(int(pid)), {
                "name": name, "cluster": pitcher_map.get(int(pid), -1),
                "throws": throws[pid], "pitches": {}})
            e["pitches"][pt] = int(n)

    joblib.dump(best, f"{ART}/model.pkl")
    joblib.dump({"pitcher_kmeans": ctx["pitcher_km"], "hitter_kmeans": ctx["hitter_km"],
                 "hitter_scaler": ctx["scaler"], "pitch_cats": PITCH_CATS,
                 "hitter_features": ctx["hitter_feats"]}, f"{ART}/cluster_models.pkl")
    train_df = df[train_m]
    json.dump(batter_profiles_json(train_df, ctx["batter_tab"], ctx["batter_rates"]),
              open(f"{ART}/batter_profiles.json", "w"))
    json.dump(pitch_profiles_json(train_df, ctx["quality_own"], ctx["quality_league"]),
              open(f"{ART}/pitch_profiles.json", "w"))
    json.dump(x_columns, open(f"{ART}/x_columns.json", "w"))
    json.dump(arsenals, open(f"{ART}/arsenals.json", "w"), indent=1)
    json.dump({str(k): v for k, v in hitter_map.items()}, open(f"{ART}/hitter_clusters.json", "w"))
    json.dump(metrics, open(f"{ART}/metrics.json", "w"), indent=2)
    print("Saved artifacts to", ART, "(next: python fetch_names.py, so the batter picker shows names)")


if __name__ == "__main__":
    main()
