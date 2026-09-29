# First durable task workflow

Jarvis now has a text entry point for ordinary requests:

```powershell
python -m jarvis --config config.lfm.yaml --text "How are you doing?"
```

It starts the configured LLM and tools but does not load microphone, wake word,
STT, or TTS. Dangerous tools require an exact `yes` in an interactive terminal;
they are denied if stdin is redirected. This one-shot agent turn is not yet a
resumable task. The persistent task command below is a separate first workflow.

`python -m jarvis.tasks` manages records in `~/.jarvis/tasks.sqlite3` by default.
Use `--db PATH` to direct it to a test database. Its commands are `list`,
`status TASK_ID`, `cancel TASK_ID`, and `resume TASK_ID`.

The first supported workflow controls only an existing, uniquely named
`Jarvis Cua Smoke ...` test window created by `tests/manual/cua_fixture.ps1`.
To run it manually, keep the fixture open in one PowerShell window:

```powershell
$fixtureTitle = 'Jarvis Cua Smoke local1234'
$outputPath = Join-Path $env:TEMP 'jarvis-task-smoke.txt'
powershell.exe -NoProfile -STA -File tests\manual\cua_fixture.ps1 -WindowTitle $fixtureTitle -OutputPath $outputPath
```

In another window, with Cua installed:

```powershell
python -m jarvis.tasks start-fixture --window-title 'Jarvis Cua Smoke local1234' --output "$env:TEMP\jarvis-task-smoke.txt" --text 'hello from Jarvis'
```

The command prints the task ID, status, next step, and evidence. If the fixture
has not opened yet, status is `paused`; run `python -m jarvis.tasks resume ID`
after it opens. `--pause-after 2` stops at a verified step boundary so you can
test resume after process restart. `status ID` shows the event history without
starting Cua or the model. `cancel ID` makes the task terminal.

The workflow journals intent before text entry and clicking Save. On restart
it re-discovers the exact window and checks observed text or output file bytes.
If an attempted action has an uncertain outcome and its postcondition is not
present, the task enters `needs_review` without repeating it. It never
overwrites an existing file with different contents. The output file is the
exact path supplied by the user; use a fresh disposable test path for each run.

The automated `python tests/manual/cua_smoke.py` creates its own disposable
fixture and database, pauses after text entry, starts a fresh Cua runtime,
resumes, and verifies the saved bytes. It requires an interactive Windows
desktop. `pytest` covers complete, paused, cancelled, and uncertain outcomes
without native GUI dependencies.

General goals, voice task commands, durable plans across browser/files/shell,
and retention controls are later roadmap milestones. The ordinary agent now
offers a narrower tool menu per single-surface request; a separate initial
[permission gateway](security.md) checks registered actions and task-CLI scope.
Routing itself is not a security boundary or a cross-app task planner.

## Calculator acceptance workflow

On an interactive Windows desktop with Cua installed, run:

```powershell
python -m jarvis.tasks start-calculator
```

This launches Windows Calculator if needed, selects one exact Calculator
window, clears it, clicks 1 + 2 = through named accessibility controls, and
checks that the display reads 3. The command prints a task ID and evidence.
It does not ask the user for a process or window ID. It preserves checkpoints
in the task database; `--pause-after 2` and `resume ID` test a process restart.
If a click outcome is uncertain, it never blindly repeats that click. An
uncertain Plus is left for review because the display does not change on Plus;
the final result is what confirms that operator. A changed app or missing
control also stops rather than selecting a different target.

Run `python tests/manual/calculator_smoke.py` for a pause/restart/repeat smoke
test. These commands alter the Calculator display and may open Calculator;
they do not touch files outside the test task database.
