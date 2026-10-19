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


# ---- batter-vs-pitch-category and pitch-quality features ----------------------------------------
import json  # noqa: E402

from shared import MIN_QUALITY_PITCHES, PRIOR_SWINGS, QUALITY_RAW  # noqa: E402


def synth(seed=0, n=6000):
    """A small Statcast-shaped frame: 8 batters, 6 pitchers, 30 game days, all columns train needs."""
    rng = np.random.default_rng(seed)
    pitch_type = rng.choice(["FF", "SL", "CH", "SI"], n, p=[.4, .25, .2, .15])
    desc = rng.choice(["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"], n,
                      p=[.35, .2, .12, .18, .15])
    bip = desc == "hit_into_play"
    day = rng.integers(0, 30, n)
    pitcher = rng.integers(100, 106, n)
    base = {"FF": 94, "SL": 85, "CH": 85, "SI": 93}
    df = pd.DataFrame({
        "game_type": "R", "type": np.where(bip, "X", "S"), "description": desc,
        "estimated_woba_using_speedangle": np.where(bip, rng.uniform(0, 1, n), np.nan),
        "pitch_type": pitch_type, "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan,
        "game_date": (pd.Timestamp("2026-04-01") + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
        "game_pk": day, "at_bat_number": rng.integers(1, 60, n), "pitch_number": rng.integers(1, 8, n),
        "stand": rng.choice(["L", "R"], n), "p_throws": np.where(pitcher % 2 == 0, "R", "L"),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n), "outs_when_up": rng.integers(0, 3, n),
        "inning": rng.integers(1, 10, n), "plate_x": rng.normal(0, .8, n), "plate_z": rng.normal(2.5, .7, n),
        "pitcher": pitcher, "batter": rng.integers(1, 9, n), "player_name": "P",
        "events": np.where(rng.random(n) < .25, rng.choice(["single", "strikeout", "walk"], n), None),
        "launch_speed": np.where(bip, rng.normal(88, 8, n), np.nan),
        "launch_angle": np.where(bip, rng.normal(12, 15, n), np.nan),
        "launch_speed_angle": np.where(bip, rng.integers(1, 7, n), np.nan),
        "release_speed": [base[p] for p in pitch_type] + rng.normal(0, 1, n) + (pitcher - 100) * .3,
        "release_spin_rate": rng.normal(2300, 150, n), "pfx_x": rng.normal(0, .6, n),
        "pfx_z": rng.normal(.8, .4, n), "release_extension": rng.normal(6.4, .3, n)})
    return df


@pytest.fixture()
def small_thresholds(monkeypatch):
    monkeypatch.setattr(train, "MIN_PITCHER_PITCHES", 50)
    monkeypatch.setattr(train, "MIN_PA", 10)


def split(df):
    prepared = train.prepare(df)
    return prepared, (prepared["game_date"] < pd.Timestamp("2026-04-21"))


def test_batter_feature_is_shrunk_toward_league_for_batters_we_barely_know():
    rows = ([dict(batter=1, pitch_cat="fastballs", description="swinging_strike")] * 120 +
            [dict(batter=1, pitch_cat="fastballs", description="foul")] * 280 +      # veteran: 30% whiff
            [dict(batter=2, pitch_cat="fastballs", description="swinging_strike")] +   # one swing, a whiff
            [dict(batter=3, pitch_cat="fastballs", description="foul")] * 400)         # league: 121/801
    d = pd.DataFrame(rows).assign(estimated_woba_using_speedangle=np.nan)
    tab, rates = train.batter_tables(d)
    probe = pd.DataFrame({"batter": [1, 2, 3, 99], "pitch_cat": "fastballs",
                          "description": "ball", "estimated_woba_using_speedangle": np.nan})
    v = train.batter_feature_values(probe, tab, rates)["bat_whiff_vs_cat"]
    league = rates.loc["fastballs", "whiff"]
    assert v[0] == pytest.approx((120 + PRIOR_SWINGS * league) / (400 + PRIOR_SWINGS))
    assert abs(v[1] - league) < .01                      # 1 swing barely moves him off the league rate
    assert v[0] > v[1] > v[2]                            # veteran whiffs more than the two near-league batters
    assert v[3] == pytest.approx(league)                 # never seen -> exactly the league rate


def test_out_of_fold_feature_has_no_trace_of_the_rows_own_outcome():
    """Leave-one-out leaks: within a group the feature is (total - own)/(n-1), so it separates the group's
    rows by their own outcome. The out-of-fold feature must not (rows in the same fold share one value)."""
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame({"batter": 1, "pitch_cat": "fastballs", "game_pk": rng.integers(0, 40, n),
                      "description": rng.choice(["swinging_strike", "foul"], n),
                      "estimated_woba_using_speedangle": np.nan})
    y = (d["description"] == "swinging_strike").to_numpy()
    oof = train.batter_feature_values_oof(d)["bat_whiff_vs_cat"]
    fold = pd.factorize(d["game_pk"])[0] % train.N_FOLDS
    for f in range(train.N_FOLDS):
        assert np.ptp(oof[fold == f]) < 1e-12                  # same value for every row of a fold
    # what leave-one-out would give: perfectly separates whiffs from non-whiffs
    tab, rates = train.batter_tables(d)
    full = train.batter_feature_values(d, tab, rates)["bat_whiff_vs_cat"]
    own = y.astype(float)
    loo = (full * (n + train.PRIOR_SWINGS) - own) / (n - 1 + train.PRIOR_SWINGS)   # (whiffs - own + prior) / (n - 1 + k)
    assert loo[y].max() < loo[~y].min()                        # the leak the old approach had
    # and the out-of-fold value tells you nothing about the outcome beyond its fold
    assert abs(np.corrcoef(oof, y.astype(float))[0, 1]) < 0.05


def test_out_of_fold_rows_never_use_their_own_games():
    d = pd.DataFrame({"batter": 1, "pitch_cat": "fastballs", "game_pk": [0, 0, 0, 1, 2, 3, 4, 5],
                      "description": ["swinging_strike"] * 3 + ["foul"] * 5,
                      "estimated_woba_using_speedangle": np.nan})
    oof = train.batter_feature_values_oof(d, n_folds=2)["bat_whiff_vs_cat"]
    folds = pd.factorize(d["game_pk"])[0] % 2           # game 0,2,4 -> fold 0; games 1,3,5 -> fold 1
    # fold-0 rows (games 0, 2, 4) only see games 1, 3, 5: 0 whiffs in 3 swings
    other = train.shrink(0, 3, d_rate(d, folds != 0), train.PRIOR_SWINGS)
    assert oof[folds == 0] == pytest.approx(other)


def d_rate(d, mask):
    sub = d[mask]
    return (sub["description"] == "swinging_strike").sum() / len(sub)


def test_features_never_see_outcomes_or_stuff_from_after_the_training_period(small_thresholds):
    df, tr = split(synth())
    a, _ = train.add_features(df.copy(), tr)
    future = df.copy()
    rng = np.random.default_rng(1)
    v = ~tr
    future.loc[v, "description"] = rng.choice(["ball", "swinging_strike", "foul"], int(v.sum()))
    future.loc[v, "estimated_woba_using_speedangle"] = np.nan
    future.loc[v, "release_speed"] += 20
    b, _ = train.add_features(future, tr)
    cols = train.FEATURES[10:]                            # cluster, hitter_cluster, batter and quality features
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_training_rows_use_out_of_fold_batter_features_and_later_rows_use_the_full_table(small_thresholds):
    df, tr = split(synth())
    out, ctx = train.add_features(df.copy(), tr)
    full = train.batter_feature_values(df, ctx["batter_tab"], ctx["batter_rates"])
    train_rows = tr.to_numpy()
    assert not np.allclose(out.loc[train_rows, "bat_whiff_vs_cat"], full["bat_whiff_vs_cat"][train_rows])
    assert np.allclose(out.loc[~train_rows, "bat_whiff_vs_cat"], full["bat_whiff_vs_cat"][~train_rows])
    assert out["bat_whiff_vs_cat"].notna().all()


def test_pitch_quality_uses_own_average_then_league_then_nan():
    d = pd.DataFrame({"pitcher": [1] * MIN_QUALITY_PITCHES + [2] * 5 + [3] * MIN_QUALITY_PITCHES,
                      "pitch_type": ["FF"] * (MIN_QUALITY_PITCHES + 5) + ["XX"] * MIN_QUALITY_PITCHES,
                      "p_throws": "R", **{c: 90.0 for c in QUALITY_RAW.values()}})
    d.loc[d["pitcher"] == 1, list(QUALITY_RAW.values())] = 96.0       # pitcher 1: 96; pitcher 2: 90
    own, lg = train.pitch_quality_tables(d)
    probe = pd.DataFrame({"pitcher": [1, 2, 99, 1], "pitch_type": ["FF", "FF", "FF", "ZZ"], "p_throws": "R"})
    v = train.quality_feature_values(probe, own, lg)["pq_velo"]
    league_ff = d[d["pitch_type"] == "FF"]["release_speed"].mean()
    assert v[0] == 96.0                                   # his own FF average
    assert v[1] == pytest.approx(league_ff)               # only 5 FF -> league average
    assert v[2] == pytest.approx(league_ff)               # unknown pitcher
    assert np.isnan(v[3])                                 # pitch type nobody throws


def test_add_features_explains_how_to_fix_old_data_files(small_thresholds):
    df, tr = split(synth())
    with pytest.raises(SystemExit, match="get_data.py"):
        train.add_features(df.drop(columns=["release_speed"]), tr)


def test_profile_files_are_valid_json_and_list_only_batters_with_enough_pitches(small_thresholds, monkeypatch):
    monkeypatch.setattr(train, "MIN_BATTER_PITCHES", 400)
    df, tr = split(synth())
    df = df[~((df["batter"] == 8) & (df.index % 20 != 0))]            # batter 8 almost never seen
    df, tr = df.reset_index(drop=True), tr[df.index].reset_index(drop=True)
    out, ctx = train.add_features(df.copy(), tr)
    d = out[tr]
    b = train.batter_profiles_json(d, ctx["batter_tab"], ctx["batter_rates"])
    p = train.pitch_profiles_json(d, ctx["quality_own"], ctx["quality_league"])
    json.dumps(b, allow_nan=False), json.dumps(p, allow_nan=False)    # no NaN may reach the JSON files
    assert "8" not in b["batters"] and "1" in b["batters"]
    assert set(b["batters"]["1"]["cats"]) <= set(b["league"])
    assert b["batters"]["1"]["stand"] in ("L", "R") and b["batters"]["1"]["switch"] is True   # random stands
    assert {"velo", "spin", "pfx_x", "pfx_z", "ext", "n"} == set(next(iter(p["pitchers"].values()))["FF"])
    assert "FF|R" in p["league"]


def test_serving_lookups_reproduce_the_training_features(small_thresholds, modules, monkeypatch):
    """Train/serve skew guard: what recommender looks up from the saved JSON must equal the feature
    values the model was trained/validated on (for rows outside the training period)."""
    df, tr = split(synth())
    out, ctx = train.add_features(df.copy(), tr)
    b = json.loads(json.dumps(train.batter_profiles_json(out[tr], ctx["batter_tab"], ctx["batter_rates"])))
    p = json.loads(json.dumps(train.pitch_profiles_json(out[tr], ctx["quality_own"], ctx["quality_league"])))
    rc = modules.recommender
    monkeypatch.setattr(rc, "BATTERS", b)
    monkeypatch.setattr(rc, "PITCH_Q", p)
    probe = out[~tr].drop_duplicates(["batter", "pitcher", "pitch_type"]).head(150)
    for _, r in probe.iterrows():
        whiff, xw = rc.batter_vs(int(r["batter"]), r["pitch_cat"])
        q, _ = rc.pitch_stuff(int(r["pitcher"]), r["pitch_type"], r["p_throws"])
        assert whiff == pytest.approx(r["bat_whiff_vs_cat"], abs=1e-4)
        assert xw == pytest.approx(r["bat_xwoba_vs_cat"], abs=1e-4)
        for col in QUALITY_RAW:
            assert q[col] == pytest.approx(r[col], abs=1e-3)
