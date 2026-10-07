"""db.py -- SQLite persistence (Python stdlib only: no server, no cost)."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch.db"))

EVENT_COLS = [
    "game_pk", "at_bat_number", "pitch_number", "game_date",
    "pitcher_id", "pitcher_name", "batter_id",
    "balls", "strikes", "outs", "inning", "on_base", "stand", "p_throws",
    "rec_pitch", "rec_prob", "actual_pitch", "actual_prob", "actual_rank",
    "matched", "top3", "actual_success",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,            -- 'replay' | 'backtest' | 'live'
  label TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pitch_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  game_pk INTEGER, at_bat_number INTEGER, pitch_number INTEGER, game_date TEXT,
  pitcher_id INTEGER, pitcher_name TEXT, batter_id INTEGER,
  balls INTEGER, strikes INTEGER, outs INTEGER, inning INTEGER, on_base INTEGER,
  stand TEXT, p_throws TEXT,
  rec_pitch TEXT, rec_prob REAL,          -- model's top pick and its score
  actual_pitch TEXT, actual_prob REAL,    -- what was thrown and the model's score for it
  actual_rank INTEGER,                    -- rank of the thrown pitch in the model's list
  matched INTEGER, top3 INTEGER,
  actual_success INTEGER                  -- real outcome (NULL for live pitches)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON pitch_events(run_id);
"""

SUMMARY_SQL = """
SELECT COUNT(*) AS n,
       AVG(matched) AS top1_pct, AVG(top3) AS top3_pct,
       AVG(rec_prob) AS avg_rec_prob, AVG(actual_prob) AS avg_actual_prob,
       AVG(CASE WHEN matched = 1 THEN actual_success END) AS success_matched,
       AVG(CASE WHEN matched = 0 THEN actual_success END) AS success_unmatched,
       SUM(matched) AS n_matched
FROM pitch_events WHERE run_id = ?
"""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _py(v):
    return v.item() if hasattr(v, "item") else v   # numpy scalar -> python


def init_db():
    with closing(_connect()) as c:
        c.executescript(SCHEMA)
        c.commit()


def start_run(source, label=None):
    with closing(_connect()) as c:
        cur = c.execute("INSERT INTO runs (source, label, created_at) VALUES (?,?,?)",
                        (source, label, datetime.now(timezone.utc).isoformat()))
        c.commit()
        return cur.lastrowid


def log_events(run_id, events):
    rows = [(run_id, *[_py(e.get(col)) for col in EVENT_COLS]) for e in events]
    q = (f"INSERT INTO pitch_events (run_id, {', '.join(EVENT_COLS)}) "
         f"VALUES (?{',?' * len(EVENT_COLS)})")
    with closing(_connect()) as c:
        c.executemany(q, rows)
        c.commit()


def scoreboard(run_id=None, source=None):
    """Summary for one run, or for the most recent run of a given source."""
    with closing(_connect()) as c:
        if run_id is None and source:
            run_id = c.execute("SELECT MAX(id) FROM runs WHERE source = ?", (source,)).fetchone()[0]
        if run_id is None:
            return {"n": 0}
        r = dict(c.execute(SUMMARY_SQL, (run_id,)).fetchone())
    r["run_id"] = run_id
    r["n_matched"] = int(r["n_matched"] or 0)
    r["n_unmatched"] = r["n"] - r["n_matched"]
    return r


def list_runs(limit=20):
    with closing(_connect()) as c:
        return [dict(x) for x in c.execute(
            "SELECT id, source, label, created_at FROM runs ORDER BY id DESC LIMIT ?", (limit,))]
