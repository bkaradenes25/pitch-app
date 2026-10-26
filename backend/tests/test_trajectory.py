import math

import pytest

import live
import trajectory as T

# a plausible four-seam fastball, numbers given at y = 50 ft
FF = dict(x0=-1.6, y0=50.0, z0=6.0, vx0=4.0, vy0=-135.0, vz0=-4.0, ax=-10.0, ay=27.0, az=-17.0)


def plate_of(p):
    t = T.time_to_y(p["y0"], p["vy0"], p["ay"])
    x, y, z = T.position_at(p, t)
    return t, x, y, z


def test_the_ball_is_at_the_front_of_the_plate_when_the_time_is_up():
    t, x, y, z = plate_of(FF)
    assert y == pytest.approx(T.PLATE_Y, abs=1e-9)
    assert 0.35 < t < 0.45                     # a ~92 mph pitch takes about 0.37 s to cover its last 50 feet


def test_the_earlier_root_is_used_not_the_one_after_drag_reverses_the_ball():
    t = T.time_to_y(50.0, -135.0, 27.0)
    assert t < 0.5                                       # the later root (ball turning around) is ~9 s away


@pytest.mark.parametrize("y0,vy0,ay", [(50, 0, 0), (50, 135, 0), (50, -10, 1000)])
def test_a_ball_that_never_arrives_gives_none(y0, vy0, ay):
    assert T.time_to_y(y0, vy0, ay) is None


def test_statcast_row_recovers_the_fifty_foot_starting_point():
    t, plate_x, _, plate_z = plate_of(FF)
    row = {"plate_x": plate_x, "plate_z": plate_z, **{k: FF[k] for k in ("vx0", "vy0", "vz0", "ax", "ay", "az")}}
    got = T.from_statcast(row)
    assert got["x0"] == pytest.approx(FF["x0"], abs=1e-3) and got["z0"] == pytest.approx(FF["z0"], abs=1e-3)
    assert got["y0"] == 50.0 and set(got) == set(T.KEYS)
    # and the recovered path lands exactly where the row says the ball crossed the plate
    _, x, _, z = plate_of(got)
    assert (x, z) == (pytest.approx(plate_x, abs=1e-3), pytest.approx(plate_z, abs=1e-3))


def test_a_release_point_that_makes_sense_comes_out_of_real_looking_numbers():
    got = T.from_statcast({"plate_x": -0.25, "plate_z": 2.49, "vx0": 5.0, "vy0": -128.0, "vz0": -6.0,
                           "ax": -6.0, "ay": 24.0, "az": -20.0})
    assert -4 < got["x0"] < 4 and 4.5 < got["z0"] < 7.5


@pytest.mark.parametrize("missing", ["plate_x", "vx0", "az"])
def test_statcast_row_without_tracking_columns_gives_none(missing):
    row = {"plate_x": 0.1, "plate_z": 2.5, "vx0": 1, "vy0": -130, "vz0": -4, "ax": 1, "ay": 25, "az": -20}
    assert T.from_statcast({k: v for k, v in row.items() if k != missing}) is None
    assert T.from_statcast({**row, missing: float("nan")}) is None
    assert T.from_statcast({}) is None


def test_feed_coordinates_become_the_nine_numbers():
    c = {"x0": -2.0, "y0": 50.0, "z0": 6.1, "vX0": 5.0, "vY0": -130.0, "vZ0": -5.0,
         "aX": -8.0, "aY": 25.0, "aZ": -18.0, "pX": 0.1, "pZ": 2.5}
    assert T.from_feed_coordinates(c) == {"x0": -2.0, "y0": 50.0, "z0": 6.1, "vx0": 5.0, "vy0": -130.0,
                                          "vz0": -5.0, "ax": -8.0, "ay": 25.0, "az": -18.0}
    assert T.from_feed_coordinates({k: v for k, v in c.items() if k != "aZ"}) is None
    assert T.from_feed_coordinates({}) is None and T.from_feed_coordinates(None) is None


def test_live_snapshot_pitches_carry_their_flight_path():
    import json
    from pathlib import Path
    feed = json.loads((Path(__file__).parent / "fixtures" / "feed_midgame_runner_on.json").read_text())
    pitches = live.parse_snapshot(feed)["pitches_this_at_bat"]
    assert all(set(p["traj"]) == set(T.KEYS) for p in pitches)
    for p in pitches:                                   # the path really ends at the recorded plate location
        _, x, _, z = plate_of(p["traj"])
        assert x == pytest.approx(p["plate_x"], abs=2e-3) and z == pytest.approx(p["plate_z"], abs=2e-3)
    assert math.isclose(pitches[0]["traj"]["x0"], -2.1739528977, abs_tol=1e-3)      # real release point


def test_pitches_without_tracking_numbers_have_no_path():
    pitch = live._pitch({"isPitch": True, "details": {"type": {"code": "FF"}},
                         "pitchData": {"coordinates": {"pX": 0.0, "pZ": 2.5}}})
    assert pitch["traj"] is None and pitch["plate_x"] == 0.0
