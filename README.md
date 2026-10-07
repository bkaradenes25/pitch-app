# ⚾ Pitch Recommender

![CI](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml/badge.svg)
<!-- Add a screenshot or GIF at docs/demo.gif, then uncomment: ![Demo](docs/demo.gif) -->

A full-stack app that recommends the next pitch from the game situation **and checks honestly whether those
recommendations hold up** on pitches the model has never seen.

- **Sandbox:** pick a pitcher, count, outs, inning, handedness and runners; see ranked pitch types and a strike-zone
  heatmap (including out-of-zone cells).
- **Replay & Scoreboard:** replay a held-out game pitch by pitch and compare the model's call with the pitch actually
  thrown and how it turned out, with a running scoreboard.
- **Live games:** stream today's MLB games (MLB Stats API) with a recommendation for every at-bat.

## Results at a glance
| | |
|---|---|
| Outcome model, held-out test AUC / Brier | 0.780 / 0.182 (baseline Brier 0.234) |
| What drives accuracy | Location (AUC 0.78 → 0.57 without it); pitch type adds about 0.009 |
| Backtest | 113,063 unseen pitches |
| Pitchers who threw the model's top pick vs not | **+2.9 points** better good-outcome rate (positive in all 12 counts), +2.8 to +3.2 when comparing each pitcher with himself |
| Caveat | An association, not proof of cause; no velocity/spin/movement features |

Full details, ablation and limitations: **[MODEL_CARD.md](MODEL_CARD.md)**.

## Architecture
```
Statcast (pybaseball) -> monthly Parquet -> train.py (XGBoost + k-means clusters) -> artifacts/
                                                                                      |
MLB Stats API (live feed) -> live.py ----------------------------> FastAPI  <---------+
                                                                    |   \-> SQLite (runs, pitch events, scoreboard)
                                                                    +-> React + TypeScript UI (served by the same app)
```
Python 3.12 · FastAPI · Pydantic · XGBoost · scikit-learn · pandas · SQLite · React 18 · TypeScript · Vite ·
pytest · Vitest · ESLint · Docker · GitHub Actions.

## Quick start
**Docker (one command)**
```bash
docker compose up --build        # http://localhost:8000
```
**Without Docker**
```bash
cd frontend && npm install && npm run build && cp -r dist ../backend/static
cd ../backend && pip install -r requirements-serve.txt && python -m uvicorn app:app    # http://localhost:8000
```
**Development (hot reload)**
```bash
cd backend && pip install -r requirements.txt -r requirements-dev.txt && python -m uvicorn app:app --reload
cd frontend && npm install && npm run dev                                            # http://localhost:5173
```
The repo ships the trained model (`backend/artifacts/`) and a few held-out demo games (`backend/demo_data/`), so
Sandbox and Replay work out of the box. The Live tab needs internet access and a game in progress.

## Rebuilding the model from scratch
```bash
cd backend
python get_data.py          # download the Statcast season by month (cached, resumable)
python train.py             # train on the regular season; postseason stays held out
python build_locations.py   # real pitch locations per pitcher/pitch type/count (training period only)
python -u backtest.py       # score every held-out pitch into SQLite (add --every 10 for a quick run)
python analyze.py           # baselines, count-adjusted and within-pitcher comparison, bootstrap intervals
python ablation.py          # which feature groups drive the accuracy
python summary.py && python make_demo_data.py   # bundle results and demo games for deployment
```

## How pitches are scored
- **Ranking score:** the model's average success probability over locations sampled from where *that pitcher* really
  throws the pitch in this kind of count (ahead / even / behind). Comparable across pitch types and not inflated by
  picking the best cell.
- **Heatmap:** success probability if the pitch is located at an exact spot (5x5 grid incl. out-of-zone cells).

## Engineering notes
- Chronological train/validation/test split; clusters and location samples built from training-period data only.
- Missing model inputs raise an error instead of being silently zero-filled; unknown players map to cluster -1.
- Input validation with Pydantic; typed API responses on the frontend (`frontend/src/types.ts`).
- Backend tests (pytest) cover the database, scoring logic, label rules, feed parsing and API; frontend tests (Vitest)
  cover the components. CI runs ESLint, the TypeScript check, both test suites, a production build, and a smoke test of
  the Docker container on every push and pull request.

## Project structure
```
backend/   app.py (API) · recommender.py · replay.py · db.py · live.py · shared.py
           train.py · get_data.py · build_locations.py · backtest.py · analyze.py · ablation.py
           artifacts/ (trained model + lookups) · demo_data/ · tests/
frontend/  src/{components,views}/ · types.ts · api.ts · tests next to the code
Dockerfile · docker-compose.yml · .github/workflows/ci.yml · MODEL_CARD.md
```

## Data and disclaimer
Pitch data: MLB Statcast via Baseball Savant, retrieved with `pybaseball`. The repo includes only a small sample
(`backend/demo_data`); the full dataset is not redistributed, so check Baseball Savant / MLB terms before reusing it.
This is an educational project, not affiliated with MLB or any team, and not intended for in-game decisions or betting.
