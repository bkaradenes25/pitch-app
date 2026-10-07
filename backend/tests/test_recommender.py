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
