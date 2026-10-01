"""Opt-in Windows workflow smoke across two real Jarvis task processes."""
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


async def main():
    title = f"Jarvis Cua Smoke {uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="jarvis-cua-") as directory:
        output = Path(directory) / "smoke.txt"
        process = subprocess.Popen([
            "powershell.exe", "-NoProfile", "-STA", "-File",
            str(Path(__file__).with_name("cua_fixture.ps1")),
            "-OutputPath", str(output), "-WindowTitle", title,
        ], creationflags=subprocess.CREATE_NO_WINDOW)
        started = time.monotonic()
        db_path = Path(directory) / "tasks.sqlite3"

        async def command(*args):
            process_result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "jarvis.tasks", "--db", str(db_path), *args],
                capture_output=True, text=True, timeout=60,
            )
            if process_result.returncode not in (0, 3):  # 3 = ended in needs_review
                raise RuntimeError(process_result.stderr or process_result.stdout)
            return json.loads(process_result.stdout)

        try:
            paused = await command(
                "start-fixture", "--window-title", title, "--output", str(output),
                "--text", "Jarvis Cua verified native desktop workflow", "--pause-after", "2",
            )
            for _ in range(30):
                if paused["current_step"] == 2:
                    break
                await asyncio.sleep(0.25)
                paused = await command("resume", paused["id"], "--pause-after", "2")
            assert paused["status"] == "paused" and paused["current_step"] == 2, paused

            # A second process reopens SQLite, rediscovers the window, and saves.
            final = await command("resume", paused["id"])
            assert final["status"] == "completed", final
            assert output.read_text(encoding="utf-8-sig") == "Jarvis Cua verified native desktop workflow"
            print(f"PASS: CLI task process restart and exact save verified in {time.monotonic()-started:.2f}s")
        finally:
            process.terminate()
            await asyncio.to_thread(process.wait, timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
