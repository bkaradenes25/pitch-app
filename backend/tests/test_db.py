import sqlite3

import numpy as np
import pytest


def ev(**kw):
    base = dict(game_pk=1, pitcher_id=1, pitcher_name="X", balls=0, strikes=0, rec_pitch="FF",
                rec_prob=0.6, actual_pitch="FF", actual_prob=0.6, actual_rank=1,
                matched=0, top3=1, actual_success=0)
    return {**base, **kw}


def test_scoreboard_math(modules):
    db = modules.db
    rid = db.start_run("backtest", "4 pitches")
    db.log_events(rid, [
        ev(matched=1, top3=1, actual_success=1, actual_prob=0.6),
        ev(matched=0, top3=1, actual_success=0, actual_prob=0.5),
        ev(matched=0, top3=1, actual_success=1, actual_prob=0.4),
        ev(matched=0, top3=0, actual_success=0, actual_prob=0.5),
    ])
    sb = db.scoreboard(run_id=rid)
    assert sb["n"] == 4
    assert sb["top1_pct"] == pytest.approx(0.25)
    assert sb["top3_pct"] == pytest.approx(0.75)
    assert sb["avg_actual_prob"] == pytest.approx(0.5)
    assert sb["success_matched"] == pytest.approx(1.0)
    assert sb["success_unmatched"] == pytest.approx(1 / 3)
    assert (sb["n_matched"], sb["n_unmatched"]) == (1, 3)


def test_empty_scoreboard(modules):
    assert modules.db.scoreboard(source="backtest") == {"n": 0}


def test_scoreboard_uses_latest_run_of_source(modules):
    db = modules.db
    first = db.start_run("backtest", "old")
    db.log_events(first, [ev()])
    second = db.start_run("backtest", "new")
    db.log_events(second, [ev(), ev()])
    assert db.scoreboard(source="backtest")["n"] == 2


def test_numpy_values_are_stored_as_plain_integers(modules):
    db = modules.db
    rid = db.start_run("replay")
    db.log_events(rid, [ev(game_pk=np.int64(5), matched=np.int64(1), rec_prob=np.float64(0.7))])
    c = sqlite3.connect(db.DB_PATH)
    assert c.execute("SELECT typeof(game_pk), typeof(matched), typeof(rec_prob) FROM pitch_events"
                     ).fetchone() == ("integer", "integer", "real")


def test_null_outcomes_are_ignored_in_success_rates(modules):
    """Live pitches have no outcome; averages must skip NULLs rather than treat them as 0."""
    db = modules.db
    rid = db.start_run("live")
    db.log_events(rid, [ev(matched=1, actual_success=None)])
    assert db.scoreboard(run_id=rid)["success_matched"] is None


def test_list_runs_newest_first(modules):
    db = modules.db
    db.start_run("replay", "a")
    db.start_run("live", "b")
    assert [r["label"] for r in db.list_runs()] == ["b", "a"]
