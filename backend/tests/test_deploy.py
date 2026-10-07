import importlib
import json

import pytest

DATA_MISSING = "no data"


def _no_data():
    raise SystemExit(DATA_MISSING)


# ---- bundled backtest summary ----
def test_summary_roundtrip(modules, monkeypatch, tmp_path):
    import summary
    monkeypatch.setattr(summary, "ART", str(tmp_path))
    assert summary.load_summary() is None
    db = modules.db
    rid = db.start_run("backtest", "2 pitches")
    db.log_events(rid, [dict(matched=1, top3=1, rec_prob=.5, actual_prob=.4, actual_success=1),
                        dict(matched=0, top3=0, rec_prob=.6, actual_prob=.3, actual_success=0)])
    assert summary.write_summary()["n"] == 2
    assert summary.load_summary()["n"] == 2


def test_summary_requires_a_backtest_run(modules, monkeypatch, tmp_path):
    import summary
    monkeypatch.setattr(summary, "ART", str(tmp_path))
    with pytest.raises(SystemExit):
        summary.write_summary()


# ---- replay falls back to bundled demo data ----
def test_replay_uses_demo_data_when_full_data_is_missing(modules, monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    train = pytest.importorskip("train")
    import pandas as pd
    monkeypatch.setattr(train, "load_pitches", _no_data)
    pd.DataFrame({"game_pk": [1, 2]}).to_parquet(tmp_path / "demo.parquet")
    monkeypatch.setattr(modules.replay, "DEMO_DIR", tmp_path)
    assert list(modules.replay._load_raw()["game_pk"]) == [1, 2]


def test_replay_without_any_data_raises_file_not_found(modules, monkeypatch, tmp_path):
    train = pytest.importorskip("train")
    monkeypatch.setattr(train, "load_pitches", _no_data)
    monkeypatch.setattr(modules.replay, "DEMO_DIR", tmp_path)   # empty folder
    with pytest.raises(FileNotFoundError):
        modules.replay._load_raw()


# ---- one server for API + UI ----
@pytest.fixture()
def api(modules):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import app
    importlib.reload(app)
    from fastapi.testclient import TestClient
    return app, TestClient(app.app)


def test_frontend_is_served_next_to_the_api(api, tmp_path):
    app, client = api
    (tmp_path / "index.html").write_text("<div id='root'></div>")
    assert app.mount_frontend(app.app, str(tmp_path)) is True
    assert "root" in client.get("/").text
    assert client.get("/health").json() == {"ok": True}      # API routes still win over the static mount


def test_missing_frontend_folder_is_skipped(api, tmp_path):
    app, _ = api
    assert app.mount_frontend(app.app, str(tmp_path / "nope")) is False


def test_backtest_scoreboard_falls_back_to_the_bundled_summary(api, monkeypatch, tmp_path):
    import summary
    _, client = api
    monkeypatch.setattr(summary, "ART", str(tmp_path))
    (tmp_path / "backtest_summary.json").write_text(json.dumps({"n": 99, "top1_pct": 0.24}))
    assert client.get("/scoreboard?source=backtest").json()["n"] == 99


def test_a_local_backtest_run_wins_over_the_bundled_summary(api, modules, monkeypatch, tmp_path):
    import summary
    _, client = api
    monkeypatch.setattr(summary, "ART", str(tmp_path))
    (tmp_path / "backtest_summary.json").write_text(json.dumps({"n": 99}))
    rid = modules.db.start_run("backtest", "1 pitch")
    modules.db.log_events(rid, [dict(matched=1, top3=1, rec_prob=.5, actual_prob=.5, actual_success=1)])
    assert client.get("/scoreboard?source=backtest").json()["n"] == 1
