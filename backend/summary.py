"""
summary.py -- bundle the latest backtest result so a deployed copy can show it without the database.

    python summary.py      # writes artifacts/backtest_summary.json from the latest 'backtest' run (commit it)
"""
import json
import os

import db

ART = os.environ.get("ARTIFACT_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))


def _path():
    return os.path.join(ART, "backtest_summary.json")


def write_summary():
    sb = db.scoreboard(source="backtest")
    if not sb.get("n"):
        raise SystemExit("No backtest run found. Run backtest.py first.")
    with open(_path(), "w") as f:
        json.dump(sb, f, indent=2)
    return sb


def load_summary():
    try:
        with open(_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    print("Saved", _path(), write_summary())
