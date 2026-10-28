# Cua desktop backend

Install with `python -m pip install -e ".[desktop]"`. The desktop extra pins
`cua-driver==0.28.2`, whose Windows bindings were tested. The repository's
`config.yaml` selects `desktop.backend: cua`; configurations without that section
retain the legacy backend. Set `backend: legacy` to disable Cua.

Desktop requests expose six tools: list windows, launch app (which also lists the
app's windows), observe window, click control, type text, and press key. Browser
requests still use Playwright. Routing is a keyword heuristic. Mixed browser/desktop
workflows may need separate requests. UI trees are untrusted input, not instructions.

## What the model sees

Results are short text, not JSON, so a 4B model can read them and a window costs about
a third of the context it used to:

```
window pid=20872 window_id=1181720 "Calculator"
8 Text "Display is 0"
26 Button "One"
32 Button "Seven"
21 Button "Multiply by"
```

The model clicks, types and presses keys by **label** (`desktop_click(pid, window_id,
label="Seven")`) instead of copying snapshot ids and element indexes from one result to the
next. With indexes the 4B model failed "enter 25" and "7 times 8" in Calculator: it clicked
Seven, Eight, Nine... by counting up through the list instead of looking the digit up. A label must match a
control exactly (case and spacing ignored). Anything else is an error that lists the labels,
and nothing is clicked: no partial matches, no guessing. `index` is only for two controls with
the same label. Every action then returns only what changed:

```
clicked 32 Button "Seven"
changed:
8 Text "Display is 7"
(38 other controls unchanged)
```

or `the N listed controls did not change` (an effect the tree does not show, like an expression a
calculator keeps out of its list), or the whole window again when it changed
so much (a dialog opened) that a diff is meaningless. A desktop request gets 16 model steps
instead of 8, since operating an app takes a call per button.

## What is checked

Actions require an exact discovered process/window and a fresh enabled element
from its accessibility snapshot. Snapshots expire after 30 seconds and are
consumed by actions; a label that fails to resolve does not consume it. The driver
still refuses an element token from any other snapshot. Each action returns a fresh
observation; dispatch does not prove task completion. An interrupted/timed-out native
call disables subsequent desktop dispatch until restart: its outcome may be unknown, so
do not replay it.

A desktop approval names the control ("... on pid=4, window_id=9, control Delete"), and the
approval is bound to that label, so it cannot be reused for another control. The durable
task workflows (`jarvis.tasks`) use the same checks through `CuaDesktop.observe` and `act`
with explicit ids and never ask the model to choose a control.

Defaults: background delivery only, telemetry off, update checks off, 30-second
call deadline, 40 controls and depth 6 per observation. There is no automatic
foreground fallback or coordinate guessing. Canvas-only/custom inaccessible UI
is not supported by this adapter. Text entry can replace existing values.
History/result budgets are character bounds, not exact token accounting.

When `security.confirm_dangerous_ops` is enabled, all Cua mutations request voice
confirmation. The existing local config disables this setting; review it before
using real accounts. UI clicks can send messages, submit forms, or delete data:
this adapter is not a security sandbox. Existing shell/file tools and their
permission policy still need a dedicated security audit.

## Verification

Run `python -m pytest -q` and `python -m ruff check .`.
On an interactive Windows desktop, run `python tests/manual/cua_smoke.py`.
The opt-in native smoke opens a uniquely named WinForms fixture, observes controls,
types text, invokes Save, and independently verifies exact file bytes inside a
temporary directory. It terminates only its fixture process and removes its own
temporary file. It does not require Ollama or modify existing apps.

This is a tested integration and initial hardening pass, not production
certification. Full model-driven desktop tasks, actual audio hardware interruption,
long-running soak tests, app accessibility coverage, and security review remain
release gates. The native smoke does not measure LLM accuracy or voice latency.
Ollama must be running with the configured model before full startup can pass.
