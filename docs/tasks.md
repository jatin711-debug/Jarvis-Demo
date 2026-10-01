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
`status TASK_ID`, `cancel TASK_ID`, and `resume TASK_ID`. Progress lines
(`task step verified id=... step=... n=2/4`) go to stderr with the task id
shortened to 8 characters; task input is never logged, only its length. The
JSON result goes to stdout as a single UTF-8 document, even when stdout is a
pipe. The `start-*` commands log `task created id=...` with the full id (the one
`status`, `resume`, and `cancel` take) as soon as the record exists.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Success: the result was printed (a run that completed, paused, or was cancelled; `list`, `status`, `cancel`) |
| 1 | The command failed, for example the desktop backend would not start or another process changed the task |
| 2 | Usage error: bad arguments (`--max-seconds`/`--max-actions` must be positive, `--pause-after` at least 1), an unknown task ID, or an unreadable `--config` |
| 3 | `start-*` or `resume` ran and ended in `needs_review` |
| 130 | Interrupted |

`needs_review` used to share code 2 with usage errors; scripts should now test
for 3. Errors print one `jarvis.tasks: ...` line on stderr, never a traceback.
If the desktop backend fails to start, a task that was still `pending` is marked
`needs_review` with the error (a `startup_failed` event) instead of staying
`pending`; fix the problem and run `resume ID`. A `paused` task is left as it
was.

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

## Step contracts and the engine

Workflows are data. Each step is a `StepContract` (`jarvis/tasks/contracts.py`)
that declares:

| Field | Meaning |
| --- | --- |
| `name`, `target` | Step id and a human-readable statement of what it touches |
| `operation` | `read` (never changes anything) or `mutate` |
| `observe` | A fresh observation or independent read (window state, file bytes) |
| `precondition` | Checked on that observation before dispatch; resolves the exact target |
| `action` | The single dispatch (mutating steps only) |
| `observe_after`, `postcondition` | How the outcome is re-observed and judged |
| `timeout` | Seconds for the whole step, enforced with `asyncio.timeout` |
| `retry` | `reobserve` (read steps), `reconcile`, `adopt_satisfied`; a dispatch is never replayed |
| `evidence` | Builds the proof recorded for the step |

`TaskRunner` runs every step through the same phases and records the phase in
each event's `details`:

1. **inspect**: observe and test the postcondition. If it already holds the step
   is reconciled without acting (`reconciled: true` when a dispatch had been
   journaled, for example after a crash).
2. **act**: test the precondition and the budgets, journal `action_intent`
   (with the attempt) before dispatching, then dispatch once.
3. **verify**: judge the postcondition on a fresh observation or independent
   read, record evidence, and advance.

Stops are explicit. A read step whose target is not there yet (the fixture
window) ends in `paused`. Anything uncertain ends in `needs_review`: an
unmet precondition, an unverified postcondition, a journaled attempt whose
postcondition is not observed (no second dispatch), an exception, or a lost
process. `Plus` in the Calculator workflow sets `reconcile=False` because the
display cannot prove it, and stays `acknowledged` until the final result
verifies it.

## Budgets and timeouts

Every task carries `budget = {max_seconds, max_actions}` and progress counters
`elapsed_seconds`, `actions_taken`, and `started_at`. Defaults come from the
workflow (fixture: 180 s and 4 actions; Calculator: 300 s and 8 actions). Set
them at creation with `store.create_fixture(..., budget=Budget(...))` or
`--max-seconds` / `--max-actions` on `start-*`. `max_seconds` counts time spent
inside runs, accumulated over resumes, not time spent paused.

Budgets are checked before each step and before each dispatch, never during
one, so a run stops in `needs_review` with a clear `error` and the in-flight
action always finishes and is verified first. The overshoot is bounded by one
step timeout. To continue an exhausted task, raise the caps with
`resume ID --max-seconds N --max-actions M` (or `store.set_budget`).

A step that exceeds its `timeout` ends differently by kind. A mutating step goes
to `needs_review` and is never replayed (the action may have landed; resume
re-observes and reconciles). A read step is re-observed up to `retry.reobserve`
times and then `paused`.

## Schema and migration

`SCHEMA_VERSION` is 2. Version 1 databases (no budget fields) are upgraded in
place, inside one transaction, the first time `TaskStore` opens them: each task
gets its workflow's default budget, `elapsed_seconds = 0`, `actions_taken`
counted from its `action_intent` events, and a `schema_migrated` event. Task
`version` numbers, evidence, and journaled attempts are untouched, so a
migrated task resumes with the same no-replay guarantees. A database written by
a newer schema is refused rather than modified.

## Adding a workflow

A new workflow needs a `Workflow` of contracts and a creator; the store, runner,
and `status`/`resume`/`cancel` commands do not change.

```python
from jarvis.tasks.contracts import Budget, Check, StepContract, TaskSpec, Workflow
from jarvis.tasks.store import TaskStore
from jarvis.tasks.workflows import register_workflow

register_workflow(Workflow(
    name="notepad_note",
    steps=(
        StepContract(name="find_window", target="the Notepad window", operation="read",
                     observe=list_windows, postcondition=exactly_one_window,
                     evidence=window_evidence, on_unmet="pause"),
        StepContract(name="type_note", target="the Notepad editor", operation="mutate",
                     observe=observe_editor, postcondition=editor_has_text,
                     precondition=resolve_editor, action=type_note,
                     evidence=lambda ctx, state, phase: {"editor": "matched"},
                     uncertain="Typing may have reached Notepad; no automatic replay"),
    ),
    result="Note typed and verified",
    budget=Budget(max_seconds=120, max_actions=3),
    permissions=lambda task: (("desktop_launch", {"name": "Notepad"}),),
))


def note_spec(text: str) -> TaskSpec:
    return TaskSpec(goal="Type a note in Notepad", scope={"window_title": "Untitled - Notepad"},
                    input={"text": text})


task = TaskStore(path).create("notepad_note", note_spec("hello"))
```

Callables take a `StepContext` (`ctx.task`, `ctx.scope`, `ctx.input`,
`ctx.desktop`) and may be plain or `async`. Postconditions and preconditions
return a `Check`; a precondition passes what it resolved to `action` through
`Check.handle`. Definitions are validated when constructed (unique lower-snake
names, positive timeouts, a mutating step needs an action and is never retried,
the default budget covers the mutating steps), so a bad workflow fails at
import. Give the task's `permissions` callable the gateway checks the scope must
pass. Add a `start-*` CLI command only if the workflow needs a shortcut.
