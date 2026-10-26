"""
get_data.py -- download the 2026 Statcast season efficiently.

    python get_data.py

Why this is fast/light:
  * pybaseball.statcast() splits a range into per-day requests and runs them in parallel
  * pybaseball's disk cache makes re-runs instant
  * one Parquet file per month (columns trimmed) -> small and fast to load
  * resumable: completed months are skipped; only the current month is refreshed
Regular season AND postseason are downloaded (game_type column tells them apart),
so postseason games can be your out-of-sample demo set.
"""
from datetime import date
from pathlib import Path

import pandas as pd

OUT = Path("data")
SEASON_START = date(2026, 3, 25)   # adjust to the real Opening Day if needed
SEASON_END = date.today()

KEEP = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "pitcher", "batter", "player_name", "stand", "p_throws", "pitch_type",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b", "plate_x", "plate_z",
    "type", "description", "events",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle",
    "bat_speed", "swing_length", "attack_angle", "attack_direction",  # bat tracking (if present)
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_extension",  # pitch quality
    "vx0", "vy0", "vz0", "ax", "ay", "az",   # flight path (optional: only the 3D pitch view uses them)
]
REQUIRED_NEW = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_extension"]


def needs_download(path, month_end, last_day, force=False):
    """Download if the month isn't saved yet, is still in progress, was saved without the
    pitch-quality columns (files made by an older version of this script), or force is set."""
    if force or not path.exists() or month_end >= last_day:
        return True
    import pyarrow.parquet as pq
    return any(c not in pq.read_schema(path).names for c in REQUIRED_NEW)


def month_ranges(start, end):
    cur, last = pd.Timestamp(start), pd.Timestamp(end)
    while cur <= last:
        stop = min(cur + pd.offsets.MonthEnd(0), last)
        yield cur, stop, last
        cur = stop + pd.Timedelta(days=1)


def main():
    import sys
    force = "--refresh" in sys.argv       # also re-download months saved without the flight-path columns
    from pybaseball import cache, statcast
    cache.enable()
    OUT.mkdir(exist_ok=True)
    for s, e, last in month_ranges(SEASON_START, SEASON_END):
        f = OUT / f"statcast_{s:%Y-%m}.parquet"
        if not needs_download(f, e, last, force):   # finished month already saved with every column
            print(f"skip {f.name}")
            continue
        print(f"downloading {s.date()} to {e.date()} ...")
        df = statcast(start_dt=f"{s:%Y-%m-%d}", end_dt=f"{e:%Y-%m-%d}",
                      verbose=False, parallel=True)
        if df is None or df.empty:
            print("  no data")
            continue
        df = df[[c for c in KEEP if c in df.columns]]
        df.to_parquet(f, index=False)
        print(f"  saved {len(df):,} pitches -> {f.name}")


if __name__ == "__main__":
    main()
