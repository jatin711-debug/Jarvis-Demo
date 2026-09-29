# Cua desktop backend

Install with `python -m pip install -e ".[desktop]"`. The desktop extra pins
`cua-driver==0.28.2`, whose Windows bindings were tested. The repository's
`config.yaml` selects `desktop.backend: cua`; configurations without that section
retain the legacy backend. Set `backend: legacy` to disable Cua.

Desktop requests expose six tools: list windows, launch app, observe window,
click control, type text, and press key. Browser requests still use Playwright.
Routing is a keyword heuristic. Mixed browser/desktop workflows may need separate
requests. UI trees are untrusted input, not instructions.

Actions require an exact discovered process/window and a fresh enabled element
from its accessibility snapshot. Snapshots expire after 30 seconds and are
consumed by actions. Each action returns a fresh observation; dispatch does not
prove task completion. An interrupted/timed-out native call disables subsequent
desktop dispatch until restart: its outcome may be unknown, so do not replay it.

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
