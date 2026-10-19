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

**v2 additions (code in place; metrics below are still v1 until the pipeline is re-run):**
- *Batter vs pitch category:* the batter's whiff rate and xwOBA on contact against fastballs / breaking balls /
  offspeed, shrunk toward the league rate (prior weight 150 swings / 60 batted balls). Training rows get out-of-fold values (5
  folds by game, so a row's feature comes from other games only). A first version used leave-one-out and scored worse
  on the test set, because within a group that feature leaks the row's own outcome.
- *Pitch quality:* this pitcher's training-period average velocity, spin, horizontal and vertical movement and
  extension for that pitch type, falling back to the league average for (pitch type, handedness).
- `tests/test_train.py` checks that nothing from after the training period reaches any feature and that the saved
  profile files reproduce the training-time feature values (train/serve parity).

## Model
`XGBClassifier` (hist trees), tuned with `RandomizedSearchCV` and time-series cross-validation:
max_depth 7, learning_rate 0.023, n_estimators 393, subsample 0.85, colsample_bytree 0.85.

## Predictive performance (held-out, later in the season; model v1 numbers, v2 test AUC / Brier is 0.783 / 0.181 per the ablation below)
| | AUC | Log loss | Brier |
|---|---|---|---|
| Cross-validation | 0.781 | | |
| Validation | 0.783 | 0.529 | 0.181 |
| Test | 0.780 | 0.531 | 0.182 |
| Test, always predict the average | | | 0.234 |

Results are stable across cross-validation, validation and test, so there is no sign of overfitting or time drift.
Calibration was checked only in aggregate (below), not with a reliability curve.

## What drives the accuracy (ablation, test set, model v2)
Every variant is retrained with the same hyperparameters and the same chronological split. Compare rows only within
this table: hyperparameters are re-tuned on every `train.py` run, which alone moved an unchanged variant by about
0.002 AUC between two runs.

| Features used | AUC | Brier |
|---|---|---|
| All features | 0.7827 | 0.1809 |
| Previous model (clusters only, no batter-vs-pitch, no pitch quality) | 0.7803 | 0.1818 |
| Previous + batter vs pitch type | 0.7802 | 0.1819 |
| Previous + pitch quality | 0.7826 | 0.1808 |
| All, minus batter vs pitch type | 0.7826 | 0.1808 |
| All, minus pitch quality | 0.7802 | 0.1819 |
| No location (`plate_x`, `plate_z`) | 0.5754 | 0.2300 |
| No pitch type | 0.7827 | 0.1808 |
| Location + count only | 0.7683 | 0.1864 |
| Pitch type + count only | 0.5698 | 0.2305 |
| Count only | 0.5631 | 0.2311 |
| Always predict the average | | 0.2340 |

- **Location carries nearly all of the accuracy** (mostly "in the strike zone or not"): without it AUC falls to 0.575.
- **Pitch quality adds a small gain** (about +0.002 AUC, -0.001 Brier), the same size as the run-to-run noise but in
  the same direction in two separate runs. It also makes pitch type redundant: with velocity, spin and movement in the
  model, removing the pitch-type column costs nothing (before, it cost about 0.009 AUC).
- **Batter vs pitch type adds nothing** beyond the hitter cluster (0.7802 vs 0.7803). It is kept because it makes the
  recommendations respond to the batter chosen at no cost in accuracy, not because it improves accuracy.
- A first version of the batter features used leave-one-out encoding and *reduced* test AUC by about 0.009, because
  that encoding leaks each row's own outcome into its feature. Out-of-fold encoding fixed it (see Features).

The recommender depends on the smaller pitch-level signal, so its output should be read with modest confidence.

## How the recommender scores pitches
- **Score:** for each pitch type in the pitcher's arsenal, the model's *average* success probability over locations
  sampled from where that pitcher actually throws it in this kind of count (ahead / even / behind), falling back to
  league-wide locations. Command-realistic, and comparable across pitch types.
- **Heatmap:** success probability if the pitch is located at an exact spot on a 5x5 grid that includes out-of-zone cells.
- **History:** the first version scored each pitch by the best cell of an in-zone-only grid and recommended a fastball
  65% of the time (matched pitches did 1.6 points *worse*). Scoring over real locations and adding out-of-zone cells
  fixed this (fastball share fell to 8.6%) without retraining.

## Backtest on pitches the model never saw
118,503 pitches (postseason games plus the regular-season slice after 2026-08-31). For every pitch the game state
before the pitch is rebuilt, the recommender makes its call, and the call is compared with the pitch actually thrown
and its real outcome.

| Check | Result |
|---|---|
| Average model score for the pitch thrown vs the real good-outcome rate | 0.369 vs about 0.374 (aggregate calibration looks good) |
| Model's top pick equals the pitch thrown | 25.8% (random pick 24.2%; pitcher's most-used pitch 39.9%) |
| Thrown pitch is in the model's top 3 | 69.2% |
| Model score: its top pick vs the pitch thrown | 0.416 vs 0.369 |

The model's calls are **not** a prediction of what pitchers throw (match rate is close to chance), they answer a different
question: which pitch is most likely to work.

**Did pitchers who threw the model's top pick do better?** Difference in good-outcome rate, matched minus not matched,
with 95% intervals from a bootstrap that resamples whole games (200 resamples):

| Comparison | Difference | 95% interval |
|---|---|---|
| Within the same count | +2.38 points | +1.61 to +3.00 |
| Same pitcher, count bucket | +2.50 points | +1.77 to +3.22 |
| Same pitcher, exact count | +2.18 points | +1.39 to +2.78 |

The difference is positive in 10 of 12 counts (3-2 is about zero; 3-0 is negative, but only 422 matched pitches) and
does not shrink when each pitcher is compared with himself, so it is not just "good pitchers match more". The v2
numbers are a little smaller than v1 (about +2.9 points, 113,063 pitches), but the intervals overlap and the held-out
slice is larger, so this backtest does not show v2 beating v1.

Recommended top-pick mix vs pitches thrown: FF 11.6% vs 31.4%, SI 19.5% vs 16.7%, SL 25.7% vs 12.5%, CH 7.6% vs 11.0%,
FC 6.2% vs 8.2%, ST 13.9% vs 8.1%, CU 10.6% vs 6.6%, FS 2.3% vs 3.2%, KC 2.0% vs 1.4%.

## Limitations
- **Association, not causation.** Pitchers choose pitches using information the data does not contain (scouting
  reports, sequence, how a pitch feels that day). Matching the model going with better outcomes does not prove that
  following it would improve results.
- **Pitch quality is averaged (v2).** v1 had no velocity, spin or movement. v2 uses each pitcher's average per pitch
  type, not the quality of the pitch actually thrown, so a pitcher's off day is invisible.
- **The label shapes the output.** Counting weak contact and called strikes as success probably favors sinkers and
  sliders; that is a property of the definition, not proof about strategy.
- **Location model is coarse.** Typical locations are split only by count bucket, not by batter, sequence or game state.
- **Limited batter and no sequence effects.** v2 adds each batter's results by pitch category, not by location or
  count; previous pitches in the at-bat are not used. Game context is limited to inning, outs and whether anyone is on base.
- **One season, one data source.** Players with little data are mapped to cluster -1 (an out-of-distribution value).
- **Evaluation is partial.** The pitcher's actual choice is not ground truth, the outcome comparison cannot control for
  everything, and aggregate calibration does not guarantee per-situation calibration.

## Ideas for next steps
Pitch-sequence features, a run-value label, calibration curves, location models that depend on batter and base-out
state, and a within-batter comparison in the backtest.

## Reproduce
```bash
cd backend
python get_data.py && python train.py && python build_locations.py && python fetch_names.py
python -u backtest.py && python analyze.py --boot 200 && python ablation.py
```
