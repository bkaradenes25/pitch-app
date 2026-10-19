import numpy as np
import pandas as pd
import pytest

from shared import GRID_POINTS, count_bucket


def test_recommendations_are_ranked_and_limited(modules, state):
    out = modules.recommender.recommend(1, state, batter_id=7, top_n=2)
    recs = out["recommendations"]
    assert len(recs) == 2
    assert recs[0]["success_prob"] >= recs[1]["success_prob"]


def test_only_arsenal_pitches_are_recommended(modules, state):
    out = modules.recommender.recommend(1, state, top_n=10)
    assert {r["pitch_type"] for r in out["recommendations"]} == {"FF", "SL", "CH"}
    assert out["used_default_arsenal"] is False


def test_grid_includes_out_of_zone_cells_and_best_cell_is_consistent(modules, state):
    for r in modules.recommender.recommend(1, state, top_n=10)["recommendations"]:
        assert len(r["grid"]) == len(GRID_POINTS) == 25
        assert max(abs(c["plate_x"]) for c in r["grid"]) > 0.83      # outside the zone, horizontally
        assert min(c["plate_z"] for c in r["grid"]) < 1.5            # below the zone
        assert all(0.0 <= c["prob"] <= 1.0 for c in r["grid"])
        best = max(r["grid"], key=lambda c: c["prob"])
        assert r["best_cell_prob"] == pytest.approx(best["prob"], abs=1e-4)
        assert (r["best_location"]["plate_x"], r["best_location"]["plate_z"]) == (best["plate_x"], best["plate_z"])


def test_detail_false_skips_the_grid(modules, state):
    r = modules.recommender.recommend(1, state, detail=False)["recommendations"][0]
    assert "grid" not in r and "best_location" not in r and 0 <= r["success_prob"] <= 1


def test_unknown_pitcher_falls_back_to_default_arsenal(modules, state):
    out = modules.recommender.recommend(999999, state)
    assert out["used_default_arsenal"] is True and out["pitcher_cluster"] == -1
    assert len(out["recommendations"]) == 5


def test_hitter_cluster_lookup(modules, state):
    assert modules.recommender.recommend(1, state, batter_id=7)["hitter_cluster"] == 3
    assert modules.recommender.recommend(1, state, batter_id=12345)["hitter_cluster"] == -1
    assert modules.recommender.recommend(1, state)["hitter_cluster"] == -1


def test_missing_state_fields_raise_instead_of_silently_zero_filling(modules, state):
    del state["balls"]
    with pytest.raises(ValueError, match="balls"):
        modules.recommender.recommend(1, state)


def test_input_state_is_not_mutated(modules, state):
    before = dict(state)
    modules.recommender.recommend(1, state)
    assert state == before


def test_list_pitchers_sorted_by_name(modules):
    names = [p["name"] for p in modules.recommender.list_pitchers()]
    assert names == sorted(names) and len(names) == 2


# ---- command-realistic scoring ----------------------------------------------------------
def test_feature_matrix_matches_training_encoding(modules, state):
    """Guards against train/serve skew: hand-built matrix == get_dummies + reindex."""
    rc = modules.recommender
    pts, xs, zs = ["FF", "SL", "CH", "SI"], [0.1, -0.5, 0.9, 0.0], [2.5, 3.0, 1.2, 2.0]
    got = rc._matrix(state, 2, 3, pts, xs, zs)
    ref = pd.DataFrame({**{k: state[k] for k in rc.REQUIRED_STATE}, "pitch_type": pts,
                        "plate_x": xs, "plate_z": zs, "cluster": 2, "hitter_cluster": 3})
    ref = (pd.get_dummies(ref, columns=["pitch_type", "stand", "p_throws"]).astype(float)
           .reindex(columns=rc.X_COLUMNS, fill_value=0.0))
    pd.testing.assert_frame_equal(got.astype(float), ref, check_dtype=False)


class FakeModel:
    """Success depends only on how far the pitch is from the middle horizontally."""
    def predict_proba(self, X):
        p = np.clip(0.5 - 0.1 * np.abs(X["plate_x"].to_numpy()), 0, 1)
        return np.column_stack([1 - p, p])


def test_score_is_the_mean_over_the_pitchers_real_locations(modules, state, monkeypatch):
    rc = modules.recommender
    monkeypatch.setattr(rc, "MODEL", FakeModel())
    real = np.array([[0.0, 2.5]] * 15 + [[1.0, 2.5]] * 10)      # 15 centered, 10 at x=1
    monkeypatch.setattr(rc, "LOCATIONS", {"pitchers": {"1": {"FF": {"all": real}}}, "league": {}})
    out = rc.recommend(1, state, top_n=10, detail=False)["recommendations"]
    ff = next(r for r in out if r["pitch_type"] == "FF")
    assert ff["success_prob"] == pytest.approx((15 * 0.5 + 10 * 0.4) / 25, abs=1e-4)
    assert (ff["locations_source"], ff["n_samples"]) == ("pitcher", 25)


def test_location_fallback_order(modules):
    rc = modules.recommender
    assert rc.pick_locations(1, "FF", "ahead")[1] == "pitcher_count"
    assert rc.pick_locations(1, "FF", "even")[1] == "pitcher"        # no 'even' bucket stored
    assert rc.pick_locations(1, "SL", "ahead")[1] == "pitcher"
    assert rc.pick_locations(2, "FF", "even")[1] == "league"         # pitcher 2 has no samples
    assert rc.pick_locations(1, "CH", "even") == (None, "grid")      # nothing anywhere


def test_without_a_locations_file_everything_falls_back_to_the_grid(modules, state, monkeypatch):
    rc = modules.recommender
    monkeypatch.setattr(rc, "LOCATIONS", {"pitchers": {}, "league": {}})
    recs = rc.recommend(1, state, top_n=10)["recommendations"]
    assert {r["locations_source"] for r in recs} == {"grid"} and len(recs) == 3


def test_count_bucket():
    assert [count_bucket(0, 2), count_bucket(1, 1), count_bucket(3, 0), count_bucket(0, 0)] == \
           ["ahead", "even", "behind", "even"]


# ---- batter vs pitch category + pitch quality ---------------------------------------------------
NEW_COLS = ["bat_whiff_vs_cat", "bat_xwoba_vs_cat", "pq_velo", "pq_spin", "pq_pfx_x", "pq_pfx_z", "pq_ext"]


class ColumnModel:
    """Success probability = a chosen feature column (so we can see exactly which value reached the model)."""
    def __init__(self, col, scale=1.0, offset=0.0):
        self.col, self.scale, self.offset = col, scale, offset

    def predict_proba(self, X):
        p = np.clip(self.offset + self.scale * X[self.col].to_numpy(), 0, 1)
        return np.column_stack([1 - p, p])


def test_matrix_places_the_new_feature_columns_and_matches_get_dummies(modules, state):
    rc = modules.recommender
    pts, xs, zs = ["FF", "SL", "CH"], [0.1, -0.5, 0.9], [2.5, 3.0, 1.2]
    feats = {"bat_whiff_vs_cat": [.1, .2, .3], "bat_xwoba_vs_cat": [.4, .5, .6], "pq_velo": [95, 85, 84],
             "pq_spin": [2400, 2600, 1700], "pq_pfx_x": [-.5, .3, -1.0], "pq_pfx_z": [1.3, .2, .6],
             "pq_ext": [6.6, 6.2, 6.4]}
    got = rc._matrix(state, 2, 3, pts, xs, zs, feats)
    ref = pd.DataFrame({**{k: state[k] for k in rc.REQUIRED_STATE}, "pitch_type": pts,
                        "plate_x": xs, "plate_z": zs, "cluster": 2, "hitter_cluster": 3, **feats})
    ref = (pd.get_dummies(ref, columns=["pitch_type", "stand", "p_throws"]).astype(float)
           .reindex(columns=rc.X_COLUMNS, fill_value=0.0))
    pd.testing.assert_frame_equal(got.astype(float), ref, check_dtype=False)


def test_the_batter_changes_the_ranking_through_his_profile_against_each_pitch_category(modules, state, monkeypatch):
    rc = modules.recommender
    monkeypatch.setattr(rc, "MODEL", ColumnModel("bat_whiff_vs_cat"))
    probs = lambda b: {r["pitch_type"]: r["success_prob"]
                       for r in rc.recommend(1, state, batter_id=b, top_n=10, detail=False)["recommendations"]}
    # batter 7 whiffs 18% vs fastballs, 40% vs breaking balls, 30% vs offspeed
    assert probs(7) == pytest.approx({"FF": .18, "SL": .40, "CH": .30}, abs=1e-4)
    # no batter / unknown batter -> league rate (.25) for every category
    assert probs(None) == pytest.approx({"FF": .25, "SL": .25, "CH": .25}, abs=1e-4)
    assert probs(123456) == pytest.approx({"FF": .25, "SL": .25, "CH": .25}, abs=1e-4)


def test_batter_category_missing_from_his_profile_falls_back_to_league(modules):
    rc = modules.recommender
    assert rc.batter_vs(7, "other") == (.25, .35)             # his profile has no 'other'
    assert rc.batter_vs(7, "fastballs") == (.18, .42)


def test_pitch_quality_reaches_the_model_and_falls_back_per_column(modules, state, monkeypatch):
    rc = modules.recommender
    monkeypatch.setattr(rc, "MODEL", ColumnModel("pq_velo", scale=.01))
    out = {r["pitch_type"]: r for r in rc.recommend(1, state, top_n=10, detail=False)["recommendations"]}
    assert out["FF"]["success_prob"] == pytest.approx(.951, abs=1e-4)      # this pitcher's own FF velocity
    assert out["SL"]["success_prob"] == pytest.approx(.85, abs=1e-4)
    assert out["CH"]["success_prob"] == pytest.approx(.84, abs=1e-4)       # pitcher has none -> league CH|R
    assert [r["pitch_type"] for r in rc.recommend(1, state, detail=False)["recommendations"]] == ["FF", "SL", "CH"]
    # extension is None for his SL, so only that column falls back to the league value
    q, src = rc.pitch_stuff(1, "SL", "R")
    assert q["pq_velo"] == 85.0 and q["pq_ext"] == 6.4 and src == "pitcher"
    assert rc.pitch_stuff(1, "CH", "R")[1] == "league"
    assert rc.pitch_stuff(1, "XX", "R") == ({k: None for k in q}, None)


def test_stuff_is_returned_for_display_in_inches(modules, state):
    out = {r["pitch_type"]: r for r in modules.recommender.recommend(1, state, top_n=10)["recommendations"]}
    ff = out["FF"]["stuff"]
    assert ff == {"velo": 95.1, "spin": 2300.0, "ext": 6.4, "source": "pitcher", "ivb_in": 15.6, "hb_in": -6.0}
    assert out["CH"]["stuff"]["source"] == "league"


def test_batter_info_in_the_response(modules, state):
    rc = modules.recommender
    b = rc.recommend(1, state, batter_id=7)["batter"]
    assert (b["name"], b["known"], b["stand"], b["switch"]) == ("Test Batter", True, "L", False)
    assert b["vs"]["fastballs"] == {"whiff": .18, "xwoba_con": .42, "swings": 300}
    unknown = rc.recommend(1, state, batter_id=555)["batter"]
    assert unknown["known"] is False and unknown["name"] == "Batter 555" and unknown["vs"] == {}
    assert rc.recommend(1, state)["batter"] is None


def test_list_batters_is_sorted_and_names_fall_back_to_the_id(modules):
    out = modules.recommender.list_batters()
    assert [b["batter_id"] for b in out] == [8, 7]                 # "Batter 8" sorts before "Test Batter"
    assert out[0] == {"batter_id": 8, "name": "Batter 8", "stand": "R", "switch": True}
    assert out[1]["name"] == "Test Batter" and out[1]["switch"] is False


def test_pitchers_list_carries_throwing_hand_when_known(modules):
    by_id = {p["pitcher_id"]: p for p in modules.recommender.list_pitchers()}
    assert by_id[1]["throws"] == "R" and by_id[2]["throws"] is None


def test_an_older_model_without_profile_files_still_works(modules, state, monkeypatch):
    """No batter/pitch profiles (and unknown values reach the model as NaN) must degrade, not crash."""
    rc = modules.recommender
    monkeypatch.setattr(rc, "BATTERS", {"league": {}, "batters": {}})
    monkeypatch.setattr(rc, "PITCH_Q", {"pitchers": {}, "league": {}})
    out = rc.recommend(1, state, batter_id=7)
    assert len(out["recommendations"]) == 3
    assert all(r["stuff"] is None and 0 <= r["success_prob"] <= 1 for r in out["recommendations"])
    assert out["batter"]["known"] is False and rc.list_batters() == []
