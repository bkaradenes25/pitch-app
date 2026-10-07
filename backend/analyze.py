"""
analyze.py -- sanity checks, baselines and robustness checks for the latest backtest run.

    python analyze.py              # includes a 200-resample bootstrap (about a minute)
    python analyze.py --boot 50    # faster, rougher interval
    python analyze.py --boot 0     # skip the bootstrap
"""
import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import db

ART = os.environ.get("ARTIFACT_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))

# strata used to compare "pitcher matched the model" vs "did not" among otherwise similar pitches
SPECS = {
    "count only": ["balls", "strikes"],
    "same pitcher, count bucket (ahead/even/behind)": ["pitcher_id", "bucket"],
    "same pitcher, exact count": ["pitcher_id", "balls", "strikes"],
}


def stratified_diff(df, keys):
    """Weighted average of (matched - not matched) good-outcome rate within strata that contain
    both kinds of pitch. Weights n1*n0/(n1+n0) (Mantel-Haenszel style) favor well-populated cells."""
    if set(df["matched"].unique()) != {0, 1}:
        return float("nan"), 0
    t = df.groupby(keys + ["matched"])["actual_success"].agg(["mean", "size"]).unstack("matched")
    t = t.dropna(subset=[("size", 0), ("size", 1)])
    if t.empty:
        return float("nan"), 0
    n0, n1 = t[("size", 0)], t[("size", 1)]
    w = n0 * n1 / (n0 + n1)
    d = t[("mean", 1)] - t[("mean", 0)]
    return float((w * d).sum() / w.sum()), len(t)


def bootstrap(df, specs, n_boot, seed=42):
    """Cluster bootstrap: resample whole GAMES (pitches within a game are not independent)."""
    games = df.groupby("game_pk").indices           # game -> row positions
    ids = np.array(list(games))
    rng = np.random.default_rng(seed)
    out = {name: [] for name in specs}
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sub = df.iloc[np.concatenate([games[g] for g in pick])]
        for name, keys in specs.items():
            out[name].append(stratified_diff(sub, keys)[0])
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in out.items()}


def robustness(conn, run_id, n_boot):
    df = pd.read_sql_query(
        "SELECT pitcher_id, game_pk, balls, strikes, matched, actual_success FROM pitch_events "
        "WHERE run_id=? AND actual_success IS NOT NULL", conn, params=(run_id,))
    df["matched"] = df["matched"].astype(int)
    df["bucket"] = np.where(df["strikes"] > df["balls"], "ahead",
                            np.where(df["balls"] > df["strikes"], "behind", "even"))
    print("Robustness: good-outcome difference (matched - not matched)")
    cis = bootstrap(df, SPECS, n_boot) if n_boot > 0 else {}
    for name, keys in SPECS.items():
        est, k = stratified_diff(df, keys)
        ci = f"  95% CI [{cis[name][0]:+.2%}, {cis[name][1]:+.2%}]" if name in cis else ""
        print(f"  {name:48s} {est:+.2%}{ci}   ({k:,} strata)")
    print("  Read: if the 'same pitcher' rows stay positive with an interval above 0, the gap is not")
    print("  just 'different pitchers'. It is still an association, not proof the model causes better outcomes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (0 = skip)")
    args = ap.parse_args()

    c = sqlite3.connect(db.DB_PATH)
    c.row_factory = sqlite3.Row
    run = c.execute("SELECT id FROM runs WHERE source='backtest' ORDER BY id DESC LIMIT 1").fetchone()
    if not run:
        raise SystemExit("No backtest run found. Run backtest.py first.")
    rows = c.execute(
        "SELECT pitcher_id, balls, strikes, rec_pitch, actual_pitch, matched, actual_success "
        "FROM pitch_events WHERE run_id=?", (run["id"],)).fetchall()
    n = len(rows)
    print(f"Backtest run {run['id']}: {n:,} pitches\n")

    # 1. baselines for "model's top pick = pitch thrown"
    try:
        arsenals = json.load(open(os.path.join(ART, "arsenals.json")))
    except OSError:
        arsenals = {}
    top_hits = rand = 0.0
    for r in rows:
        a = arsenals.get(str(r["pitcher_id"]))
        if a:
            top_hits += max(a["pitches"], key=a["pitches"].get) == r["actual_pitch"]
            rand += 1 / len(a["pitches"])
        else:
            rand += 1 / 5
    print("Top-1 match rate (how often the call equals the pitch thrown)")
    print(f"  model                        {sum(r['matched'] for r in rows) / n:.1%}")
    print(f"  pitcher's most-used pitch    {top_hits / n:.1%}")
    print(f"  random pick from arsenal     {rand / n:.1%}\n")

    # 2. matched vs not matched, by count
    g = defaultdict(lambda: {"m": [0, 0], "u": [0, 0]})   # [good outcomes, pitches]
    for r in rows:
        if r["actual_success"] is None:
            continue
        d = g[(r["balls"], r["strikes"])]["m" if r["matched"] else "u"]
        d[0] += r["actual_success"]
        d[1] += 1
    tot = diff = 0
    print("Good-outcome rate by count: pitcher matched the model vs did not")
    for (b, s), d in sorted(g.items()):
        if d["m"][1] and d["u"][1]:
            sm, su = d["m"][0] / d["m"][1], d["u"][0] / d["u"][1]
            w = d["m"][1] + d["u"][1]
            print(f"  {b}-{s}: matched {sm:.1%} (n={d['m'][1]:,})   not {su:.1%} (n={d['u'][1]:,})   diff {sm - su:+.1%}")
            diff += w * (sm - su)
            tot += w
    if tot:
        print(f"  count-adjusted difference: {diff / tot:+.2%}  (positive = matching the model helped)\n")

    # 3. what does the model like to recommend?
    rc, ac = Counter(r["rec_pitch"] for r in rows), Counter(r["actual_pitch"] for r in rows)
    print("Pitch type share: model's top pick vs pitch thrown")
    for pt in sorted(set(rc) | set(ac), key=lambda p: -ac[p]):
        if rc[pt] / n > 0.01 or ac[pt] / n > 0.01:
            print(f"  {pt:3s} recommended {rc[pt] / n:6.1%}   thrown {ac[pt] / n:6.1%}")

    # 4. is the gap just "different pitchers"? (within-pitcher comparison + game-level bootstrap)
    print()
    robustness(c, run["id"], args.boot)


if __name__ == "__main__":
    main()
