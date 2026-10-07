"""
ablation.py -- which feature groups actually drive the model's AUC?

    python ablation.py       # retrains several small variants with the tuned params (a few minutes)

Uses the same chronological split as train.py. Results are saved to artifacts/ablation.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import train

VARIANTS = {
    "all features": train.FEATURES,
    "no location (plate_x/z)": [f for f in train.FEATURES if f not in ("plate_x", "plate_z")],
    "no pitch type": [f for f in train.FEATURES if f != "pitch_type"],
    "location + count only": ["plate_x", "plate_z", "balls", "strikes"],
    "pitch type + count only": ["pitch_type", "balls", "strikes"],
    "count only": ["balls", "strikes"],
}


def main():
    df = train.prepare(train.load_pitches())
    dates = np.sort(df["game_date"].unique())
    t1, t2 = dates[int(len(dates) * 0.70)], dates[int(len(dates) * 0.85)]
    train_m, test_m = df["game_date"] < t1, df["game_date"] >= t2

    _, pmap = train.cluster_pitchers(df[train_m])
    _, _, _, hmap = train.cluster_hitters(df[train_m])
    df["cluster"] = df["pitcher"].map(pmap).fillna(-1)
    df["hitter_cluster"] = df["batter"].map(hmap).fillna(-1)
    y = df["success"]
    params = json.load(open(f"{train.ART}/metrics.json"))["best_params"]

    base_brier = brier_score_loss(y[test_m], np.full(int(test_m.sum()), y[train_m].mean()))
    print(f"{'variant':28s} {'test AUC':>9s} {'test Brier':>11s}   (baseline Brier {base_brier:.4f})")
    results = {}
    for name, cols in VARIANTS.items():
        cats = [c for c in cols if c in train.CATEGORICAL]
        X = pd.get_dummies(df[cols], columns=cats or None).astype(float)
        model = XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                              random_state=42, n_jobs=-1).fit(X[train_m], y[train_m])
        p = model.predict_proba(X[test_m])[:, 1]
        results[name] = {"auc": float(roc_auc_score(y[test_m], p)),
                         "brier": float(brier_score_loss(y[test_m], p))}
        print(f"{name:28s} {results[name]['auc']:9.4f} {results[name]['brier']:11.4f}", flush=True)
    json.dump({"baseline_brier": float(base_brier), "variants": results},
              open(f"{train.ART}/ablation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
