# Milestone 2 implementation plan: dependable tools and boundaries

This plan follows [the roadmap](roadmap.md). Capability routing and the
Calculator acceptance task are already implemented; routing is **not** an
authorization boundary. Work proceeds in these slices:

1. **Permission gateway (initial slice implemented).** Classify each built-in tool as read,
   local edit, system control, or external action. At the registry dispatch
   point, validate exact paths, applications, and URL hosts against configured
   scope before confirmation or execution. Show effect, target, and an action
   fingerprint in confirmations. Denial or refusal must end the agent turn.
   Test traversal, symlink escape, missing targets, changed confirmation
   arguments, and disabled-confirmation profiles. The registry gateway and
   task-CLI scope preflight now exist; active-browser destination binding,
   redirect handling, and command isolation still need work. Do not describe
   a Python path check as an OS sandbox; document the remaining race and shell
   risks.
2. **Typed tool outcomes.** Replace string-prefix inference with a structured
   result carrying tool, exact target, source, timestamp, dispatch status,
   independent postcondition, and uncertainty. Adapt desktop, filesystem,
   browser, and shell tools incrementally while retaining spoken summaries.
   A dispatch acknowledgement alone must not be marked verified.
3. **Scoped handoffs.** Persist browser URL/tab, desktop pid/window/snapshot,
   and project directory in task checkpoints. Rediscover stale UI handles and
   re-check permitted targets at each step. Never transfer an approval to a
   different target or automatically retry a refused/uncertain mutation.
4. **Acceptance gates.** Repeat Calculator and fixture restart tests; add
   wrong-window, stale-control, out-of-scope path, denied write, external
   destination, and uncertain-dispatch scenarios. Run unit/lint and live GUI
   smoke separately; report verified outcomes and unresolved uncertainty.

The gateway honors whatever profile the config selects while enforcing any
narrower scopes it specifies. The repository `config.yaml` is now the standard
profile (confirmations on, whitelist on, home directory only); a full-access
profile is an explicit per-machine opt-in in `config.local.yaml`. See
[security](security.md#profiles).
