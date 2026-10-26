"""
trajectory.py -- a pitch's flight path from tracking data.

Both MLB's live feed and Statcast describe a pitch as a constant-acceleration flight:
    position(t) = p0 + v0 * t + 0.5 * a * t^2
with the numbers given where the ball is 50 ft from the plate (y = 50).
Coordinates (feet): x = catcher's right, y = toward the pitcher, z = up, origin at the back tip of
home plate. plate_x / plate_z are measured where the ball crosses y = 17/12 ft (front of the plate).

  live feed : x0, y0, z0, vX0, vY0, vZ0, aX, aY, aZ come with every pitch.
  Statcast  : vx0, vy0, vz0, ax, ay, az (at y = 50) plus plate_x / plate_z at the plate; the 50-ft
              starting position is recovered by running the flight backwards from the plate.
The frontend draws the path from the same nine numbers (src/trajectory.ts).
"""
import math

PLATE_Y = 17 / 12      # front edge of home plate, in feet from its back tip
START_Y = 50.0         # y where Statcast gives velocity and acceleration
KEYS = ("x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az")
_FEED_KEYS = {"x0": "x0", "y0": "y0", "z0": "z0", "vx0": "vX0", "vy0": "vY0", "vz0": "vZ0",
              "ax": "aX", "ay": "aY", "az": "aZ"}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def time_to_y(y0, vy0, ay, y_target=PLATE_Y):
    """Seconds until a ball at y0 moving with (vy0, ay) reaches y_target; None if it never does.
    The earlier positive root is the real arrival (the later one is after drag has reversed the ball)."""
    a, b, c = 0.5 * ay, vy0, y0 - y_target
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        t = -c / b
        return t if t > 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    r = math.sqrt(disc)
    roots = sorted(t for t in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if t > 0)
    return roots[0] if roots else None


def position_at(p, t):
    """(x, y, z) of the pitch described by p (a dict with KEYS) after t seconds."""
    return (p["x0"] + p["vx0"] * t + 0.5 * p["ax"] * t * t,
            p["y0"] + p["vy0"] * t + 0.5 * p["ay"] * t * t,
            p["z0"] + p["vz0"] * t + 0.5 * p["az"] * t * t)


def _finish(p):
    if time_to_y(p["y0"], p["vy0"], p["ay"]) is None:
        return None
    return {k: round(v, 4) for k, v in p.items()}


def from_feed_coordinates(coords):
    """Pitch flight from a live-feed pitchData.coordinates dict; None if any number is missing."""
    vals = {k: _num((coords or {}).get(src)) for k, src in _FEED_KEYS.items()}
    return None if any(v is None for v in vals.values()) else _finish(vals)


def from_statcast(row):
    """Pitch flight from a Statcast row (dict-like); None if the tracking columns are missing."""
    got = {k: _num(row.get(k)) for k in ("plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az")}
    if any(v is None for v in got.values()):
        return None
    t = time_to_y(START_Y, got["vy0"], got["ay"])
    if t is None:
        return None
    p = {"y0": START_Y, **{k: got[k] for k in ("vx0", "vy0", "vz0", "ax", "ay", "az")}}
    p["x0"] = got["plate_x"] - got["vx0"] * t - 0.5 * got["ax"] * t * t
    p["z0"] = got["plate_z"] - got["vz0"] * t - 0.5 * got["az"] * t * t
    return _finish({k: p[k] for k in KEYS})
