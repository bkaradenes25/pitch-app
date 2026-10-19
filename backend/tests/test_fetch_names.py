import json

import pytest

import fetch_names


class Resp:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


def fake_api(known, calls):
    def get(url, params=None, timeout=None):
        calls.append(params["personIds"].split(","))
        return Resp({"people": [{"id": int(i), "fullName": known[i]} for i in params["personIds"].split(",") if i in known]})
    return get


def test_ids_are_requested_in_chunks_and_names_collected(monkeypatch):
    calls = []
    known = {str(i): f"Player {i}" for i in range(250)}
    monkeypatch.setattr(fetch_names.requests, "get", fake_api(known, calls))
    out = fetch_names.fetch_names(range(250), chunk=100)
    assert [len(c) for c in calls] == [100, 100, 50]
    assert out["0"] == "Player 0" and len(out) == 250


def test_ids_the_api_does_not_know_are_simply_absent(monkeypatch):
    monkeypatch.setattr(fetch_names.requests, "get", fake_api({"1": "A"}, []))
    assert fetch_names.fetch_names([1, 2]) == {"1": "A"}


def test_a_failed_chunk_is_skipped_not_fatal(monkeypatch):
    n = {"calls": 0}

    def get(url, params=None, timeout=None):
        n["calls"] += 1
        if n["calls"] == 1:
            raise fetch_names.requests.ConnectionError("down")
        return Resp({"people": [{"id": 5, "fullName": "Five"}]})

    monkeypatch.setattr(fetch_names.requests, "get", get)
    assert fetch_names.fetch_names([1, 2, 5], chunk=2) == {"5": "Five"}


def test_people_without_a_name_are_ignored(monkeypatch):
    monkeypatch.setattr(fetch_names.requests, "get", lambda *a, **k: Resp({"people": [{"id": 1}, {"id": 2, "fullName": "Two"}]}))
    assert fetch_names.fetch_names([1, 2]) == {"2": "Two"}


def test_main_only_fetches_names_it_does_not_have_and_merges(monkeypatch, tmp_path):
    (tmp_path / "batter_profiles.json").write_text(json.dumps({"league": {}, "batters": {"1": {}, "2": {}, "3": {}}}))
    (tmp_path / "batter_names.json").write_text(json.dumps({"1": "Already Saved"}))
    calls = []
    monkeypatch.setattr(fetch_names, "ART", str(tmp_path))
    monkeypatch.setattr(fetch_names.requests, "get", fake_api({"2": "Two", "3": "Three"}, calls))
    fetch_names.main()
    assert calls == [["2", "3"]]
    assert json.loads((tmp_path / "batter_names.json").read_text()) == {"1": "Already Saved", "2": "Two", "3": "Three"}
