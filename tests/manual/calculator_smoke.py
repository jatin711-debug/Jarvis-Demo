"""Exercise Calculator across process restarts; requires interactive Windows desktop."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def task(db: Path, *args: str) -> dict:
    process = subprocess.run(
        [sys.executable, "-m", "jarvis.tasks", "--db", str(db), *args],
        text=True, capture_output=True, check=True,
    )
    return json.loads(process.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-calculator-") as root:
        db = Path(root) / "tasks.sqlite3"
        paused = task(db, "start-calculator", "--pause-after", "2")
        assert paused["status"] == "paused", paused
        assert paused["next_step"] == "enter_one", paused
        resumed = task(db, "resume", paused["id"])
        assert resumed["status"] == "completed", resumed
        assert resumed["evidence"][-1]["display"] == "3", resumed
        repeated = task(db, "start-calculator")
        assert repeated["status"] == "completed", repeated
        assert repeated["evidence"][-1]["display"] == "3", repeated
        print("PASS: paused, restarted, resumed, repeated; Calculator displayed 3")


if __name__ == "__main__":
    main()
