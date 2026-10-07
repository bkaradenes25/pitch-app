"""
make_demo_data.py -- bundle a few held-out games so anyone can use the Replay tab without the
full Statcast download (the deployed app falls back to this).

    python make_demo_data.py            # needs data/ (get_data.py) and artifacts/ (train.py)

Writes demo_data/statcast_demo.parquet (small). Commit it.
"""
import argparse

import pandas as pd

import replay
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", type=int, default=4, help="postseason games to bundle")
    ap.add_argument("--regular", type=int, default=4, help="held-out regular-season games to bundle")
    args = ap.parse_args()

    games = replay.list_games()
    pick = ([g for g in games if g["postseason"]][:args.post]
            + [g for g in games if not g["postseason"]][:args.regular])
    ids = {g["game_pk"] for g in pick}
    raw = train.load_pitches()
    demo = raw[pd.to_numeric(raw["game_pk"], errors="coerce").isin(ids)]
    replay.DEMO_DIR.mkdir(exist_ok=True)
    out = replay.DEMO_DIR / "statcast_demo.parquet"
    demo.to_parquet(out, index=False)
    print(f"Saved {len(ids)} games, {len(demo):,} pitches -> {out}")


if __name__ == "__main__":
    main()
