"""
fetch_names.py -- looks up batter names for the Sandbox batter picker.

    python fetch_names.py        # after train.py; needs internet (MLB Stats API, free)

Statcast pitch data only carries the PITCHER's name, so batter names come from the MLB Stats API
people endpoint. Writes artifacts/batter_names.json ({"660271": "Shohei Ohtani", ...}).
Safe to re-run: names already saved are kept, only missing ones are fetched. If the network is
down the picker still works and shows "Batter <id>".
"""
import json
import os

import requests

from live import BASE

ART = os.environ.get("ARTIFACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))
CHUNK = 100


def fetch_names(ids, chunk=CHUNK):
    """{id: full name} for the given MLB player ids; ids that fail to resolve are simply absent."""
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        try:
            r = requests.get(f"{BASE}/v1/people", params={"personIds": ",".join(map(str, part))}, timeout=15)
            r.raise_for_status()
            people = r.json().get("people", [])
        except (requests.RequestException, ValueError) as e:
            print(f"  could not fetch ids {part[0]}..{part[-1]}: {e}")
            continue
        for p in people:
            if "id" in p and p.get("fullName"):
                out[str(p["id"])] = p["fullName"]
    return out


def main():
    profiles = json.load(open(os.path.join(ART, "batter_profiles.json")))
    path = os.path.join(ART, "batter_names.json")
    have = json.load(open(path)) if os.path.exists(path) else {}
    todo = [b for b in profiles["batters"] if b not in have]
    print(f"{len(have)} names saved, {len(todo)} to fetch")
    have.update(fetch_names(todo))
    json.dump(have, open(path, "w"), indent=0)
    missing = [b for b in profiles["batters"] if b not in have]
    print(f"Saved {len(have)} names -> {path}" + (f" ({len(missing)} still missing)" if missing else ""))


if __name__ == "__main__":
    main()
