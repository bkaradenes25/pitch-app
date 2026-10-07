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
