"""shared.py -- constants/helpers used by both the recommender and the data-building scripts."""

# Candidate target locations (feet), catcher's view. Includes out-of-zone cells so the model
# can recommend chase pitches (the zone itself is roughly x in [-0.83, 0.83], z in [1.5, 3.5]).
GRID_X = [-1.4, -0.7, 0.0, 0.7, 1.4]
GRID_Z = [1.0, 1.8, 2.5, 3.2, 3.9]
GRID_POINTS = [(x, z) for x in GRID_X for z in GRID_Z]

MIN_PITCHER_SAMPLES = 25   # min real locations needed to trust a pitcher-specific distribution


def count_bucket(balls, strikes):
    """Pitchers locate differently when ahead (waste/chase), even, or behind (must throw strikes)."""
    if strikes > balls:
        return "ahead"
    if balls > strikes:
        return "behind"
    return "even"


# ---- pitch categories (used for batter-vs-pitch-type features) -------------------------------
FASTBALLS = ["FF", "FT", "SI", "FC", "FA"]
BREAKING = ["SL", "CU", "KC", "KN", "ST", "SB", "SV", "CS"]
OFFSPEED = ["FS", "CH", "EP", "SC"]
PITCH_CATS = ["fastballs", "breaking_balls", "offspeed", "other"]


def pitch_category(pitch_type):
    if pitch_type in FASTBALLS:
        return "fastballs"
    if pitch_type in BREAKING:
        return "breaking_balls"
    if pitch_type in OFFSPEED:
        return "offspeed"
    return "other"


# ---- model features built from per-player profiles -------------------------------------------
# Batter vs pitch category: whiff rate (whiffs / swings) and xwOBA on contact, each shrunk toward the
# league rate for that category so a hitter with few pitches seen sits near league average.
BATTER_FEATURES = ["bat_whiff_vs_cat", "bat_xwoba_vs_cat"]
PRIOR_SWINGS = 150      # weight of the league whiff rate, in swings
PRIOR_BATTED_BALLS = 60  # weight of the league xwOBA-on-contact, in batted balls
MIN_BATTER_PITCHES = 200  # batters below this are not listed in the picker (they get league values)

# Pitch quality: what THIS pitcher's version of this pitch type looks like (train-period averages).
# Feature name -> raw Statcast column.
QUALITY_RAW = {"pq_velo": "release_speed", "pq_spin": "release_spin_rate", "pq_pfx_x": "pfx_x",
               "pq_pfx_z": "pfx_z", "pq_ext": "release_extension"}
QUALITY_FEATURES = list(QUALITY_RAW)
MIN_QUALITY_PITCHES = 30  # fewer pitches of that type -> use the league average for (type, hand)


def shrink(successes, trials, prior_rate, prior_weight):
    """Empirical-Bayes style estimate: (s + w*p) / (n + w). Works on numbers and numpy arrays."""
    return (successes + prior_weight * prior_rate) / (trials + prior_weight)
