import pandas as pd
import pytest

import analyze


def frame(cells):
    """cells: (pitcher, matched, successes, n) -> one row per pitch, all in the same count."""
    rows = []
    for pitcher, matched, wins, n in cells:
        rows += [dict(pitcher_id=pitcher, balls=0, strikes=0, matched=matched, actual_success=int(i < wins),
                      bucket="even", game_pk=1) for i in range(n)]
    return pd.DataFrame(rows)


def test_within_pitcher_comparison_removes_a_skill_confound():
    # G is a good pitcher (75%) who matches the model a lot; P is weaker (25%) and rarely matches.
    # Neither pitcher does better when matching, but pooling them makes matching look helpful.
    df = frame([("G", 1, 6, 8), ("G", 0, 3, 4), ("P", 1, 1, 4), ("P", 0, 2, 8)])
    pooled, _ = analyze.stratified_diff(df, ["balls", "strikes"])
    within, k = analyze.stratified_diff(df, ["pitcher_id", "balls", "strikes"])
    assert pooled == pytest.approx(0.1667, abs=1e-3)
    assert within == pytest.approx(0.0, abs=1e-9) and k == 2


def test_true_effect_is_recovered_within_pitcher():
    df = frame([("A", 1, 3, 4), ("A", 0, 2, 4), ("B", 1, 2, 4), ("B", 0, 1, 4)])   # +25 points for both
    assert analyze.stratified_diff(df, ["pitcher_id", "balls", "strikes"])[0] == pytest.approx(0.25)


def test_returns_nan_when_only_one_group_exists():
    est, k = analyze.stratified_diff(frame([("A", 1, 1, 4)]), ["pitcher_id"])
    assert est != est and k == 0     # NaN


def test_bootstrap_resamples_games_and_returns_intervals():
    rows = []
    for game in range(10):
        for m in (0, 1):
            rows += [dict(pitcher_id="A", balls=0, strikes=0, bucket="even", game_pk=game, matched=m,
                          actual_success=int(i % 2 == 0)) for i in range(6)]
    ci = analyze.bootstrap(pd.DataFrame(rows), {"x": ["pitcher_id"]}, n_boot=20)
    lo, hi = ci["x"]
    assert lo <= hi and lo == pytest.approx(0.0) and hi == pytest.approx(0.0)   # identical groups -> no effect
