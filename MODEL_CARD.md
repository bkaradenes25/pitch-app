# Model card: pitch outcome model and pitch recommender

## Summary
An XGBoost classifier predicts the probability that a single pitch has a **good outcome for the pitcher**, given the
game situation, the pitch type, and where the pitch crosses the plate. A recommender then ranks the pitch types a
pitcher throws by their expected success. Everything is trained on the 2026 MLB regular season (Statcast).

**Intended use:** an educational / portfolio project that demonstrates an end-to-end ML application and an honest
evaluation. **Not intended** for in-game decisions, scouting conclusions, or betting.

## Data
- Source: MLB Statcast via Baseball Savant, downloaded with `pybaseball` (`backend/get_data.py`).
- All pitchers (no qualifying list). Regular season only for training; postseason games are held out.
- Chronological split by game date: train through 2026-08-03, validation 2026-08-03 to 2026-08-31, test from
  2026-08-31 (data through 2026-09-27).

## Label ("good outcome")
A pitch is a success if it is a called strike or a swinging strike, **or** a ball in play with expected wOBA of .300
or less (weak contact). Balls, fouls and hard contact count as non-successes. Base rate: about 37%.

## Features
Pitch type, batter side, pitcher hand, runners on (any), balls, strikes, outs, inning, plate location (`plate_x`,
`plate_z`), a **pitcher cluster** (k-means, k=4, on pitch-category mix) and a **hitter cluster** (k-means, k=5, on
rates derived from the same data: K%, BB%, whiff%, hard-hit%, barrel%, exit velocity, launch angle, bat tracking
when present). Both clusters are built from training-period pitches only. Players with too little data get
cluster -1.

## Model
`XGBClassifier` (hist trees), tuned with `RandomizedSearchCV` and time-series cross-validation:
max_depth 7, learning_rate 0.023, n_estimators 393, subsample 0.85, colsample_bytree 0.85.

## Predictive performance (held-out, later in the season)
| | AUC | Log loss | Brier |
|---|---|---|---|
| Cross-validation | 0.781 | | |
| Validation | 0.783 | 0.529 | 0.181 |
| Test | 0.780 | 0.531 | 0.182 |
| Test, always predict the average | | | 0.234 |

Results are stable across cross-validation, validation and test, so there is no sign of overfitting or time drift.
Calibration was checked only in aggregate (below), not with a reliability curve.

## What drives the accuracy (ablation, test set)
| Features used | AUC | Brier |
|---|---|---|
| All features | 0.780 | 0.182 |
| No location (`plate_x`, `plate_z`) | 0.572 | 0.230 |
| No pitch type | 0.772 | 0.185 |
| Location + count only | 0.768 | 0.186 |
| Pitch type + count only | 0.570 | 0.231 |
| Count only | 0.563 | 0.231 |

**Location carries nearly all of the model's accuracy** (that is mostly "in the strike zone or not"). Pitch type adds
a small amount (about +0.009 AUC). The recommender depends on that smaller signal, so it should be read with
modest confidence.

## How the recommender scores pitches
- **Score:** for each pitch type in the pitcher's arsenal, the model's *average* success probability over locations
  sampled from where that pitcher actually throws it in this kind of count (ahead / even / behind), falling back to
  league-wide locations. Command-realistic, and comparable across pitch types.
- **Heatmap:** success probability if the pitch is located at an exact spot on a 5x5 grid that includes out-of-zone cells.
- **History:** the first version scored each pitch by the best cell of an in-zone-only grid and recommended a fastball
  65% of the time (matched pitches did 1.6 points *worse*). Scoring over real locations and adding out-of-zone cells
  fixed this (fastball share fell to 8.6%) without retraining.

## Backtest on pitches the model never saw
113,063 pitches (postseason games plus the regular-season slice after 2026-08-31). For every pitch the game state
before the pitch is rebuilt, the recommender makes its call, and the call is compared with the pitch actually thrown
and its real outcome.

| Check | Result |
|---|---|
| Average model score for the pitch thrown vs the real good-outcome rate | 0.370 vs about 0.374 (aggregate calibration looks good) |
| Model's top pick equals the pitch thrown | 24.1% (random pick 24.2%; pitcher's most-used pitch 39.9%) |
| Thrown pitch is in the model's top 3 | 68.3% |
| Model score: its top pick vs the pitch thrown | 0.417 vs 0.370 |

The model's calls are **not** a prediction of what pitchers throw (match rate is at chance), they answer a different
question: which pitch is most likely to work.

**Did pitchers who threw the model's top pick do better?** Difference in good-outcome rate, matched minus not matched,
with 95% intervals from a bootstrap that resamples whole games (200 resamples):

| Comparison | Difference | 95% interval |
|---|---|---|
| Within the same count | +2.91 points | +2.28 to +3.59 |
| Same pitcher, count bucket | +3.20 points | +2.40 to +3.93 |
| Same pitcher, exact count | +2.75 points | +1.83 to +3.35 |

The difference is positive in all 12 counts and does not shrink when each pitcher is compared with himself, so it is
not just "good pitchers match more".

Recommended top-pick mix vs pitches thrown: FF 8.6% vs 31.3%, SI 25.5% vs 16.7%, SL 23.3% vs 12.6%, CH 7.2% vs 11.1%,
FC 7.3% vs 8.3%, ST 11.7% vs 8.0%, CU 11.6% vs 6.7%, FS 2.5% vs 3.2%, KC 1.6% vs 1.3%.

## Limitations
- **Association, not causation.** Pitchers choose pitches using information the data does not contain (scouting
  reports, sequence, how a pitch feels that day). Matching the model going with better outcomes does not prove that
  following it would improve results.
- **No pitch-quality features.** The model does not see velocity, spin or movement, so pitch-type effects are
  cluster-level averages, not pitcher-specific. A pitcher with a weak slider can still be told to throw it.
- **The label shapes the output.** Counting weak contact and called strikes as success probably favors sinkers and
  sliders; that is a property of the definition, not proof about strategy.
- **Location model is coarse.** Typical locations are split only by count bucket, not by batter, sequence or game state.
- **No batter-specific or sequence effects.** Batters are represented by a cluster; previous pitches in the at-bat are
  not used. Game context is limited to inning, outs and whether anyone is on base.
- **One season, one data source.** Players with little data are mapped to cluster -1 (an out-of-distribution value).
- **Evaluation is partial.** The pitcher's actual choice is not ground truth, the outcome comparison cannot control for
  everything, and aggregate calibration does not guarantee per-situation calibration.

## Ideas for next steps
Per-pitcher pitch quality (velocity, movement), batter-specific profiles, pitch-sequence features, calibration curves,
and a within-batter comparison in the backtest.

## Reproduce
```bash
cd backend
python get_data.py && python train.py && python build_locations.py
python -u backtest.py && python analyze.py --boot 200 && python ablation.py
```
