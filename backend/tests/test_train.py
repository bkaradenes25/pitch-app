"""Tests for the data-prep rules (the 'success' definition is the heart of the model)."""
import numpy as np
import pandas as pd
import pytest

train = pytest.importorskip("train")   # skipped if xgboost/scipy aren't installed


def raw(rows):
    base = dict(game_type="R", type="B", description="ball", estimated_woba_using_speedangle=np.nan,
                pitch_type="FF", on_1b=np.nan, on_2b=np.nan, on_3b=np.nan, game_date="2026-04-01",
                game_pk=1, at_bat_number=1, pitch_number=1, stand="R", p_throws="R", balls=0,
                strikes=0, outs_when_up=0, inning=1, plate_x=0.0, plate_z=2.5)
    return pd.DataFrame([{**base, **r, "pitch_number": i + 1} for i, r in enumerate(rows)])


def test_success_label_definition():
    df = raw([
        dict(description="called_strike", type="S"),                                   # -> 1
        dict(description="swinging_strike", type="S"),                                # -> 1
        dict(description="ball"),                                                     # -> 0
        dict(description="hit_into_play", type="X", estimated_woba_using_speedangle=0.20),  # weak contact -> 1
        dict(description="hit_into_play", type="X", estimated_woba_using_speedangle=0.33),  # -> 0
        dict(description="hit_into_play", type="X", estimated_woba_using_speedangle=0.55),  # -> 0
    ])
    assert train.prepare(df)["success"].tolist() == [1, 1, 0, 1, 0, 0]


def test_balls_in_play_without_expected_woba_are_dropped():
    df = raw([dict(description="hit_into_play", type="X"), dict(description="ball")])
    assert len(train.prepare(df)) == 1


def test_on_base_flag():
    out = train.prepare(raw([dict(on_2b=123.0), dict()]))
    assert out["on_base"].tolist() == [1, 0]


def test_game_type_filter():
    df = raw([dict(game_type="R"), dict(game_type="D")])
    assert len(train.prepare(df)) == 1
    assert train.prepare(df, game_types=("D",))["game_type"].tolist() == ["D"]


def test_output_is_chronological():
    df = raw([dict(game_date="2026-05-02"), dict(game_date="2026-04-01"), dict(game_date="2026-04-15")])
    dates = train.prepare(df)["game_date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == sorted(dates)


def test_hitter_profile_rates_and_min_pa(monkeypatch):
    monkeypatch.setattr(train, "MIN_PA", 4)
    rows = []
    for ev in ["strikeout", "walk", "single", "strikeout"]:
        rows.append(dict(batter=1, description="hit_into_play" if ev == "single" else "ball",
                         events=ev, launch_speed=100.0 if ev == "single" else np.nan,
                         launch_angle=10.0 if ev == "single" else np.nan,
                         launch_speed_angle=6 if ev == "single" else np.nan,
                         estimated_woba_using_speedangle=0.5 if ev == "single" else np.nan))
    rows.append(dict(batter=2, description="ball", events="walk", launch_speed=np.nan,
                     launch_angle=np.nan, launch_speed_angle=np.nan,
                     estimated_woba_using_speedangle=np.nan))
    prof = train.hitter_profile(pd.DataFrame(rows))
    assert list(prof.index) == [1]                 # batter 2 has too few PAs
    assert prof.loc[1, "k_pct"] == pytest.approx(0.5)
    assert prof.loc[1, "bb_pct"] == pytest.approx(0.25)
    assert prof.loc[1, "hard_hit_pct"] == 1.0
