# Architecture and maintenance boundaries

Jarvis is a local assistant with two execution paths. The conversational path
handles voice or text requests through the agent. The separate task CLI runs
declared, resumable workflows; it is not yet the general voice task engine.

## Conversational request

1. `jarvis.core.orchestrator.Orchestrator` starts services, manages voice
   sessions and conversation history, and shuts services down. Text requests
   use the same agent without loading voice engines.
2. `jarvis.agent.routing` selects a capability surface. `ReActAgent` prepares
   context and offered tools, calls the model, and executes its tool requests.
3. `ToolRegistry` validates calls and invokes the permission gateway before a
   tool handler runs. Browser, desktop, and system tools are separate
   registries. The configured policy applies to both text and voice requests.
4. `jarvis.agent.web_research` identifies direct web-search requests and
   formats their results. The browser tool retrieves snippets; it does not read
   and synthesize the linked articles in this path.
5. `jarvis.voice.search_monitor` coordinates speech detection, confirmation,
   and cancellation for a voice search. The orchestrator provides the actual
   audio callbacks. Searches and other actions are processed serially.

The LLM client and voice implementations own their respective runtime and
hardware integrations. Heavy libraries are loaded lazily so text-only use and
the unit suite do not require initialized audio hardware or a running model.

## Durable workflows

`jarvis.tasks` contains step contracts, the workflow runner, persistence, and a
text CLI. These workflows inspect, act, and verify against defined
postconditions. The conversational agent does not yet turn arbitrary goals
into durable workflows, and voice cancellation of a search is not a general
pause/resume facility for those tasks. See [tasks](tasks.md) and the
[roadmap](roadmap.md).

## Changing a boundary

- Keep model routing, permission decisions, and tool execution distinct. A
  model tool call is a request, not proof that an action succeeded.
- Keep voice hardware operations behind callbacks or voice interfaces. Include
  tests for cancellation and resource cleanup when changing asynchronous work.
- Preserve the text path and the lightweight import contract in
  `tests/unit/test_imports.py` when adding dependencies.
- Check outcomes with automated tests and, for hardware or browser changes,
  separately report what was verified against real devices and services.

## Release gaps

The test suite alone does not establish production readiness. Outstanding work
includes reliable voice interruption across devices, bounded and usable spoken
search responses, broader end-to-end task evaluation, and recovery of
long-running work. Release automation, dependency reproducibility, security
review, and installation on clean target machines also need verification.
Routine conversational events avoid recording transcripts, replies, and tool
argument values. Exception messages, third-party logs, model-server logs, and
existing log files may still contain sensitive data; assess access and retention
before deployment. See [logging](logging.md).
