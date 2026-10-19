import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

GOOD = {"pitcher_id": 1, "batter_id": 7,
        "state": dict(stand="L", p_throws="R", on_base=1, balls=1, strikes=2, outs_when_up=1, inning=4)}


@pytest.fixture()
def client(modules):
    import app
    importlib.reload(app)
    from fastapi.testclient import TestClient
    return TestClient(app.app)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_pitchers_endpoint(client):
    assert {p["pitcher_id"] for p in client.get("/pitchers").json()} == {1, 2}


def test_recommend_ok(client):
    r = client.post("/recommend", json=GOOD)
    assert r.status_code == 200
    assert r.json()["recommendations"][0]["pitch_type"] in {"FF", "SL", "CH"}


@pytest.mark.parametrize("field,value", [("balls", 5), ("strikes", 3), ("stand", "X"), ("inning", 0)])
def test_recommend_rejects_invalid_state(client, field, value):
    body = {**GOOD, "state": {**GOOD["state"], field: value}}
    assert client.post("/recommend", json=body).status_code == 422


def test_run_and_scoreboard_flow(client, modules):
    rid = client.post("/runs", json={"source": "replay", "label": "t"}).json()["run_id"]
    assert client.get(f"/scoreboard?run_id={rid}").json()["n"] == 0
    modules.db.log_events(rid, [dict(matched=1, top3=1, rec_prob=.6, actual_prob=.6, actual_success=1)])
    sb = client.get(f"/scoreboard?run_id={rid}").json()
    assert sb["n"] == 1 and sb["top1_pct"] == 1.0


def test_runs_rejects_unknown_source(client):
    assert client.post("/runs", json={"source": "hack"}).status_code == 422


def test_live_snapshot_uses_feed_and_recommends(client, monkeypatch, fake_feed):
    import live
    monkeypatch.setattr(live, "snapshot", lambda pk: live.parse_snapshot(fake_feed))
    r = client.get("/live/123")
    assert r.status_code == 200
    assert r.json()["recommendation"]["pitcher_id"] == 1


def test_live_feed_failure_returns_502(client, monkeypatch):
    import live
    def boom(pk): raise live.requests.RequestException("down")
    monkeypatch.setattr(live, "snapshot", boom)
    assert client.get("/live/123").status_code == 502


def test_replay_games_without_any_data_returns_503(client, modules, monkeypatch, tmp_path):
    pytest.importorskip("xgboost")   # replay lazily imports train
    monkeypatch.setattr(modules.replay, "DEMO_DIR", tmp_path)   # no bundled demo games either
    assert client.get("/replay/games").status_code == 503


def test_batters_endpoint_lists_profiled_batters(client):
    out = client.get("/batters").json()
    assert [b["batter_id"] for b in out] == [8, 7]
    assert out[1] == {"batter_id": 7, "name": "Test Batter", "stand": "L", "switch": False}


def test_pitchers_endpoint_includes_throwing_hand(client):
    by_id = {p["pitcher_id"]: p for p in client.get("/pitchers").json()}
    assert by_id[1]["throws"] == "R" and by_id[2]["throws"] is None


def test_recommend_returns_batter_and_stuff(client):
    body = client.post("/recommend", json=GOOD).json()
    assert body["batter"]["name"] == "Test Batter" and body["batter"]["known"] is True
    assert all("stuff" in r for r in body["recommendations"])
    no_batter = client.post("/recommend", json={k: v for k, v in GOOD.items() if k != "batter_id"}).json()
    assert no_batter["batter"] is None


def test_live_game_with_no_play_yet_returns_409_not_502(client, monkeypatch):
    import json
    from pathlib import Path

    import live
    pre = json.loads((Path(__file__).parent / "fixtures" / "feed_pregame_sample.json").read_text())
    monkeypatch.setattr(live, "snapshot", lambda pk: live.parse_snapshot(pre))
    r = client.get("/live/123")
    assert r.status_code == 409 and "not have started" in r.json()["detail"]
