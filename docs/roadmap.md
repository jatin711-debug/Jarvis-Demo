# Jarvis roadmap: reliable end-to-end work

Current progress: baseline unit/lint checks, the isolated Cua restart/resume
scenario, the text CLI, and the durable task engine are implemented. Workflows
are declarative step contracts (target, precondition, action, independently
verified postcondition, timeout, retry policy, evidence) run by a generic
engine with inspect/act/verify phases, per-task time and action budgets, and a
versioned, migrating task store. Milestone 2 has begun with a durable
Calculator scenario, a single capability router that offers one tool surface
per request, a permission gateway with an enforced tool classification, and a
standard security profile as the default. Still future work: LLM-driven
planning that creates tasks from open-ended goals, common typed tool outcomes,
complete browser target binding, durable cross-app handoffs, voice task
controls, and retention.

## Goal and operating rules

Jarvis should accept a goal, find the relevant apps and project files, carry out
several dependent steps, recover from ordinary changes, and report what it
verified. Voice is one interface to the same task engine; tasks must also be
testable from text without microphone or speaker hardware.

The assistant may read and inspect within its configured scope. Mutations need
clear target and permission checks. A tool dispatch is evidence of an attempt,
not proof of completion. An action with an uncertain outcome must be inspected
before any retry. The user can inspect progress and stop a task at any step.

## Baseline: reproducible starting point

1. Preserve the current working tree and document the two tested runtimes:
   Ollama/Qwen and global Python/CUDA/LFM. Identify which dependencies belong to
   each profile. Bring the README and CI instructions in line with actual setup.
2. Record baseline pytest/ruff results and a repeatable voice-free Cua smoke run.
   Keep microphone and model tests as separate, opt-in gates.
3. Create a small scenario set with exact postconditions, including: greeting
   without tools; open Calculator and compute 1+2; create, save, and verify a
   file; decline an action; recover from a stale window; and resume after a
   restart. Record completion, unnecessary actions, latency, and failure mode.

Exit: another developer can reproduce the baseline and tell whether a change
improved or regressed those scenarios. Do not claim production readiness from
unit tests or a single GUI smoke test.

## Milestone 1: durable task engine (first implementation milestone)

1. Introduce a task record containing goal, authorized scope, current step,
   status, observations, action attempts, verified outcomes, errors, and a
   concise user-facing progress summary. Persist it locally with a migration
   path and a retention policy; keep task state separate from personal memory.
2. Replace the free-form eight-iteration loop for multi-step requests with
   explicit states: plan, inspect, act, verify, recover/pause, complete. Keep
   short conversational turns on the existing fast path.
3. Provide start, status, cancel, and resume through a text interface first.
   Voice should call the same task API. On restart, load the task and inspect
   live state before continuing; never replay an uncertain mutation blindly.
4. Define step contracts: target, precondition, permitted operation,
   postcondition, timeout, retry policy, and evidence. Cap budgets for time,
   actions, and model context; show useful progress during longer work.

Exit: Jarvis can complete the fixture's discover → type → save → independently
verify sequence, pause between steps, restart, and resume without duplicate
side effects. A declined action ends or pauses the task as appropriate.

## Milestone 2: dependable tools and boundaries

1. Route by task capability instead of greeting and desktop keywords. Offer
   only the tools needed for the current step. Validate tool arguments and
   return the exact corrective schema when a call fails.
2. Give desktop, browser, filesystem, and command tools a common observation
   and result envelope: exact target, source, timestamp, action result,
   independent postcondition, and uncertainty. Use semantic controls first;
   add visual fallback only for interfaces whose controls are inaccessible.
3. Add a central permission gateway for exact paths, commands, apps, and
   external destinations. Distinguish read, local edit, and outward action.
   Confirmation should describe the target and effect, then bind to that
   action; refusal must prevent later automatic retries.
4. Make cross-app handoffs explicit: browser URL/tab, desktop process/window,
   and project directory should survive between steps without asking the user
   for machine-generated identifiers.

Exit: the Calculator scenario succeeds repeatedly; wrong-window, stale-control,
denied-action, and uncertain-dispatch cases stop or recover predictably. The
final answer names verified outcomes and remaining uncertainty.

## Milestone 3: build-project workflow

1. Add a project workspace contract: inspect existing files and git state,
   identify requested change, plan edits, run checks, preview the result, and
   provide a reviewable handoff. Protect unrelated user changes.
2. Prefer direct file/code operations for building software; use desktop UI
   where the application itself is the task or where no better interface exists.
3. Keep an artifact ledger: changed files, tests run, generated outputs, and
   unresolved issues. A task is complete only when its requested acceptance
   conditions are checked or the unmet conditions are reported.

Exit: from a text request, Jarvis can change a small project in a disposable
workspace, run its tests, inspect the result, and report exactly what changed.

## Milestone 4: model routing, memory, and voice

1. Benchmark Qwen and LFM on the same scenario set. Use LFM for tasks it
   completes reliably; route complex planning and recovery to a stronger
   model. Do not infer tool competence from greeting latency.
2. Keep task checkpoints separate from long-term user facts. Compare Mem0
   with TurboSupermemory using identical local embeddings, retrieval budgets,
   and correction/restart cases before selecting a default memory backend.
3. Profile record, STT, first model output, tool work, and TTS separately.
   Barge-in must cancel or pause the active task coherently; speaking should
   not hide unfinished tool calls. Add text-mode tests before audio soak tests.

Exit: routing improves verified completion and latency without increasing
unnecessary or unauthorized actions. Memory benefits are measured on Jarvis
workloads; voice interruption leaves task state consistent.

## Milestone 5: release gates

- Run scenarios across clean Windows setups, restarts, permission denials,
  model failures, inaccessible controls, and longer sessions.
- Maintain traces with sensitive contents redacted: model decision, offered
  tools, validation errors, action IDs, observations, and verification.
- Require scenario pass rates and latency targets agreed from the baseline,
  plus a review of path/command permissions and dependency packaging.
- Keep documentation and CI aligned with the runtime profiles actually used.

## Work order

Begin with the baseline scenarios and text entry point, then implement the
durable task record and one complete Cua fixture workflow. This gives every
later tool, model, and memory change a measurable task to improve. A full agent
framework is optional; the required properties are checkpoints, explicit
state transitions, resumability, and independent verification. LangGraph's
checkpointer/store split is a useful design reference, not a dependency choice.
