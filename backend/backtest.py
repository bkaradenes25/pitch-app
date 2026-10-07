"""
backtest.py -- score the model on every held-out pitch and store it as one 'backtest' run.

    python backtest.py

Takes a few minutes. The result appears in the app under "Full held-out backtest".
"""
import time

import db
import replay


def main():
    db.init_db()
    df = replay.heldout()
    rows = df.to_dict("records")
    run_id = db.start_run("backtest", f"{len(rows)} held-out pitches")
    print(f"Scoring {len(rows):,} held-out pitches (run {run_id}) ...")
    t0, batch = time.time(), []
    for i, row in enumerate(rows, 1):
        batch.append(replay.score_row(row)[1])
        if len(batch) >= 500:
            db.log_events(run_id, batch)
            batch = []
        if i % 2000 == 0:
            print(f"  {i:,}/{len(rows):,}  ({time.time() - t0:.0f}s)")
    if batch:
        db.log_events(run_id, batch)
    sb = db.scoreboard(run_id=run_id)
    print("Done.", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sb.items()})


if __name__ == "__main__":
    main()
