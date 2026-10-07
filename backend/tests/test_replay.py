import numpy as np
import pytest

REC = {"recommendations": [{"pitch_type": "SL", "success_prob": 0.6},
                           {"pitch_type": "FF", "success_prob": 0.5},
                           {"pitch_type": "CH", "success_prob": 0.4},
                           {"pitch_type": "CU", "success_prob": 0.3}]}


def test_evaluate_exact_match(modules):
    ev = modules.replay.evaluate({"pitch_type": "SL", "success": 1}, REC)
    assert (ev["matched"], ev["top3"], ev["actual_rank"]) == (1, 1, 1)
    assert ev["rec_pitch"] == "SL" and ev["actual_success"] == 1


def test_evaluate_third_choice(modules):
    ev = modules.replay.evaluate({"pitch_type": "CH", "success": 0}, REC)
    assert (ev["matched"], ev["top3"], ev["actual_rank"]) == (0, 1, 3)
    assert ev["actual_prob"] == pytest.approx(0.4)


def test_evaluate_fourth_choice_is_not_top3(modules):
    ev = modules.replay.evaluate({"pitch_type": "CU", "success": 0}, REC)
    assert (ev["matched"], ev["top3"]) == (0, 0)


def test_evaluate_pitch_not_in_arsenal(modules):
    ev = modules.replay.evaluate({"pitch_type": "KN", "success": 1}, REC)
    assert ev["actual_rank"] is None and ev["actual_prob"] is None
    assert (ev["matched"], ev["top3"]) == (0, 0)


@pytest.mark.parametrize("row", [{"pitch_type": "FF"}, {"pitch_type": "FF", "success": float("nan")}])
def test_evaluate_missing_outcome_is_none(modules, row):
    assert modules.replay.evaluate(row, REC)["actual_success"] is None


def test_state_of_casts_numpy_types(modules):
    row = dict(stand="L", p_throws="R", on_base=np.int64(1), balls=np.int64(2), strikes=np.int64(1),
               outs_when_up=np.int64(0), inning=np.int64(7))
    s = modules.replay.state_of(row)
    assert s["balls"] == 2 and all(type(s[k]) is int for k in ("on_base", "balls", "strikes", "inning"))


# ---- live logging: pair the call made BEFORE a pitch with the pitch that followed ----
def _prev(state, n=0, batter=7):
    return {"batter": batter, "pitcher": 1, "n": n, "rec": REC, "state": state}


def _snap(types, batter=7):
    return {"pitches_this_at_bat": [{"type": t} for t in types], "batter_id": batter,
            "pitcher_id": 1, "pitcher_name": "Test Pitcher"}


def _count(modules, rid):
    return modules.db.scoreboard(run_id=rid)["n"]


def test_live_logs_one_new_pitch(modules, state):
    rid = modules.db.start_run("live")
    modules.replay.log_live_pitch(rid, _prev(state), _snap(["SL"]), 99)
    sb = modules.db.scoreboard(run_id=rid)
    assert sb["n"] == 1 and sb["top1_pct"] == 1.0


@pytest.mark.parametrize("prev_n,types,batter", [
    (0, ["SL", "FF"], 7),   # two pitches arrived between polls: can't pair them reliably
    (0, ["SL"], 8),         # different batter
    (0, [], 7),             # nothing new
])
def test_live_skips_ambiguous_updates(modules, state, prev_n, types, batter):
    rid = modules.db.start_run("live")
    modules.replay.log_live_pitch(rid, _prev(state, prev_n), _snap(types, batter), 99)
    assert _count(modules, rid) == 0


def test_live_skips_when_no_previous_call(modules):
    rid = modules.db.start_run("live")
    modules.replay.log_live_pitch(rid, None, _snap(["SL"]), 99)
    assert _count(modules, rid) == 0
