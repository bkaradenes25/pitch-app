"""
app.py -- FastAPI backend.   Run:  python -m uvicorn app:app --reload
Docs UI at http://127.0.0.1:8000/docs
"""
import asyncio
import json
import os
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
import live
import recommender
import replay
import summary

db.init_db()
app = FastAPI(title="Pitch Recommender")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class GameState(BaseModel):
    stand: Literal["L", "R"]
    p_throws: Literal["L", "R"]
    on_base: int = Field(ge=0, le=1)
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    outs_when_up: int = Field(ge=0, le=2)
    inning: int = Field(ge=1, le=20)


class RecommendRequest(BaseModel):
    pitcher_id: int
    batter_id: Optional[int] = None
    state: GameState


class RunRequest(BaseModel):
    source: Literal["replay", "live", "backtest"] = "replay"
    label: Optional[str] = None


class StepRequest(BaseModel):
    index: int = Field(ge=0)
    run_id: int


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/pitchers")
def pitchers():
    return recommender.list_pitchers()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return recommender.recommend(req.pitcher_id, req.state.model_dump(), req.batter_id)


# ---- database-backed evaluation -------------------------------------------------
@app.post("/runs")
def create_run(req: RunRequest):
    return {"run_id": db.start_run(req.source, req.label)}


@app.get("/runs")
def runs():
    return db.list_runs()


@app.get("/scoreboard")
def scoreboard(run_id: Optional[int] = None, source: Optional[str] = None):
    sb = db.scoreboard(run_id=run_id, source=source)
    if source == "backtest" and run_id is None and not sb.get("n"):
        sb = summary.load_summary() or sb      # bundled results when no local backtest run exists
    return sb


@app.get("/replay/games")
def replay_games():
    try:
        return replay.list_games()
    except FileNotFoundError as e:
        raise HTTPException(503, f"Statcast data not found. Run get_data.py first. ({e})")


@app.post("/replay/{game_pk}/step")
def replay_step(game_pk: int, req: StepRequest):
    try:
        return replay.step(game_pk, req.index, req.run_id)
    except IndexError:
        raise HTTPException(404, "pitch index out of range")


# ---- live MLB data ---------------------------------------------------------------
@app.get("/live/games")
def live_games():
    try:
        return live.todays_games()
    except requests.RequestException as e:
        raise HTTPException(502, f"MLB API error: {e}")


@app.get("/live/{game_pk}")
def live_snapshot(game_pk: int):
    try:
        snap = live.snapshot(game_pk)
    except (requests.RequestException, KeyError) as e:
        raise HTTPException(502, f"Could not read live feed: {e}")
    snap["recommendation"] = recommender.recommend(
        snap["pitcher_id"], snap["state"], snap["batter_id"])
    return snap


@app.get("/live/{game_pk}/stream")
async def live_stream(game_pk: int, interval: int = 5):
    """SSE: pushes a new recommendation when the game state changes, and logs
    (call made before the pitch) vs (pitch actually thrown) to the database."""
    async def gen():
        run_id = db.start_run("live", f"game {game_pk}")
        last_key, prev = None, None
        while True:
            try:
                snap = await asyncio.to_thread(live.snapshot, game_pk)
                n = len(snap["pitches_this_at_bat"])
                key = json.dumps([snap["state"], snap["pitcher_id"], snap["batter_id"], n], sort_keys=True)
                if key != last_key:
                    last_key = key
                    rec = recommender.recommend(snap["pitcher_id"], snap["state"],
                                                snap["batter_id"], top_n=20)
                    try:
                        replay.log_live_pitch(run_id, prev, snap, game_pk)
                    except Exception as e:      # logging must never break the stream
                        print("live log error:", e)
                    prev = {"batter": snap["batter_id"], "pitcher": snap["pitcher_id"],
                            "n": n, "rec": rec, "state": snap["state"]}
                    snap["recommendation"] = {**rec, "recommendations": rec["recommendations"][:5]}
                    snap["scoreboard"] = db.scoreboard(run_id=run_id)
                    yield f"data: {json.dumps(snap)}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(interval)
    return StreamingResponse(gen(), media_type="text/event-stream")


def mount_frontend(application, directory):
    """Serve the built React app from the same server (one URL, no CORS). Must run after all routes."""
    if os.path.isdir(directory):
        application.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
        return True
    return False


mount_frontend(app, os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
