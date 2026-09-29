"""Generic durable workflow engine.

Each step of a registered workflow runs through explicit phases:

* inspect: take a fresh observation and check whether the postcondition
  already holds; if so, reconcile without acting.
* act: check the precondition and budgets, journal ``action_intent`` before
  dispatching, then dispatch.
* verify: judge the postcondition on a fresh observation, then advance.

Anything uncertain stops in ``needs_review`` (a journaled dispatch is never
replayed); a wait for an absent target stops in ``paused``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from jarvis.agent.permissions import PermissionGateway
from jarvis.tasks.contracts import Check, StepContext, StepContract, StepReview, Workflow
from jarvis.tasks.store import StateConflict, TaskStore
from jarvis.tasks.workflows import get_workflow
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

_TERMINAL = frozenset({"completed", "cancelled"})


async def _settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _check(value: Any, contract: StepContract, what: str) -> Check:
    if not isinstance(value, Check):
        raise TypeError(f"Step {contract.name} {what} must return a Check, got {type(value).__name__}")
    return value


def _short(task: dict[str, Any]) -> str:
    return task["id"][:8]


def _line(text: str) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= 240 else collapsed[:237] + "..."


class TaskRunner:
    def __init__(self, store: TaskStore, desktop: Any,
                 permission_gateway: PermissionGateway | None = None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.desktop = desktop
        self.permission_gateway = permission_gateway
        self.clock = clock

    async def run(self, task_id: str, *, pause_after: int | None = None) -> dict[str, Any]:
        if pause_after is not None and pause_after < 1:
            raise ValueError("pause_after must be positive")
        task = self.store.get(task_id)
        workflow = get_workflow(task["workflow"])
        if task["status"] in _TERMINAL:
            return task
        return await _Execution(self, workflow, task, pause_after).run(task)


class _Execution:
    """One ``TaskRunner.run`` call: its clock, phase, and verified-step count."""

    def __init__(self, runner: TaskRunner, workflow: Workflow, task: dict[str, Any],
                 pause_after: int | None) -> None:
        self.store = runner.store
        self.desktop = runner.desktop
        self.gateway = runner.permission_gateway
        self.clock = runner.clock
        self.workflow = workflow
        self.pause_after = pause_after
        self.base_elapsed = float(task["elapsed_seconds"])
        self.began = self.clock()
        self.verified = 0
        self.phase = "start"

    def elapsed(self) -> float:
        return self.base_elapsed + (self.clock() - self.began)

    def _step_name(self, task: dict[str, Any]) -> str:
        index = task["current_step"]
        return self.workflow.steps[index].name if index < len(self.workflow.steps) else "-"

    def _record(self, task: dict[str, Any], kind: str, *, details: dict[str, Any] | None = None,
                **changes: Any) -> dict[str, Any]:
        return self.store.record(task, kind, details=details,
                                 elapsed_seconds=round(self.elapsed(), 3), **changes)

    def _halt(self, task: dict[str, Any], status: str, kind: str, reason: str, *,
              details: dict[str, Any] | None = None) -> dict[str, Any]:
        step = self._step_name(task)
        halted = self._record(task, kind, status=status, error=reason,
                              details={"step": step, "phase": self.phase, **(details or {})})
        logger.info("task %s id=%s step=%s phase=%s reason=%s", status, _short(task), step,
                    self.phase, _line(reason))
        return halted

    def _budget_reason(self, task: dict[str, Any], *, dispatching: bool = False) -> str | None:
        budget = task["budget"]
        elapsed = self.elapsed()
        if elapsed > budget["max_seconds"]:
            return (f"Time budget exhausted: {elapsed:.1f}s used of "
                    f"{budget['max_seconds']:g}s; raise the budget to continue")
        if dispatching and task["actions_taken"] >= budget["max_actions"]:
            return (f"Action budget exhausted: {task['actions_taken']} of "
                    f"{budget['max_actions']} mutating actions used; raise the budget to continue")
        return None

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        steps = self.workflow.steps
        if self.gateway is not None and self.workflow.permissions is not None:
            self.phase = "permission"
            for name, arguments in self.workflow.permissions(task):
                decision = self.gateway.inspect(name, dict(arguments))
                if not decision.allowed:
                    return self._halt(task, "needs_review", "permission_denied", decision.reason,
                                      details={"tool": name, "target": decision.target})
        resuming = task["status"] != "pending"
        self.phase = "start"
        task = self._record(task, "resumed" if resuming else "started", status="running",
                            error=None, started_at=task["started_at"] or datetime.now(UTC).isoformat(),
                            details={"step": self._step_name(task), "phase": "start"})
        if resuming:
            logger.info("task resumed id=%s workflow=%s step=%s n=%d/%d", _short(task),
                        task["workflow"], self._step_name(task), task["current_step"], len(steps))
        else:
            logger.info("task started id=%s workflow=%s steps=%d input_len=%d", _short(task),
                        task["workflow"], len(steps),
                        len(json.dumps(task["input"], ensure_ascii=False, default=str)))
        try:
            while task["current_step"] < len(steps):
                self.phase = "budget"
                reason = self._budget_reason(task)
                if reason:
                    return self._halt(task, "needs_review", "budget_exhausted", reason)
                task = await self._run_step(task, steps[task["current_step"]])
                if task["status"] != "running":
                    return task
                self.verified += 1
                if self.pause_after is not None and self.verified >= self.pause_after:
                    logger.info("task paused id=%s step=%s reason=pause_after n=%d",
                                _short(task), self._step_name(task), self.verified)
                    return self._record(task, "paused_at_boundary", status="paused",
                                        details={"next_step": task["current_step"],
                                                 "phase": "boundary"})
            return task
        except StateConflict:
            raise
        except asyncio.CancelledError:
            current = self.store.get(task["id"])
            if current["status"] not in _TERMINAL:
                self.phase = "interrupt"
                self._halt(current, "needs_review", "interrupted",
                           "Execution interrupted; inspect outcome before resuming")
            raise
        except StepReview as exc:
            return self._halt_current(task, "needs_review", str(exc))
        except Exception as exc:
            logger.debug("task step exception id=%s", _short(task), exc_info=True)
            return self._halt_current(task, "needs_review", str(exc) or type(exc).__name__,
                                      kind="step_error")

    def _halt_current(self, task: dict[str, Any], status: str, reason: str, *,
                      kind: str = "needs_review") -> dict[str, Any]:
        current = self.store.get(task["id"])
        if current["status"] in _TERMINAL:
            return current
        return self._halt(current, status, kind, reason)

    async def _run_step(self, task: dict[str, Any], contract: StepContract) -> dict[str, Any]:
        reobserved = 0
        while True:
            self.phase = "inspect"
            try:
                async with asyncio.timeout(contract.timeout) as deadline:
                    return await self._execute(task, contract)
            except TimeoutError:
                if not deadline.expired():
                    raise
            current = self.store.get(task["id"])
            limit = f"{contract.timeout:g}s"
            if contract.operation == "mutate":
                if current["attempt"] is not None:
                    reason = (f"Step {contract.name} timed out after {limit} during {self.phase}; "
                              "the action may have taken effect, so it is not replayed")
                else:
                    reason = (f"Step {contract.name} timed out after {limit} during {self.phase} "
                              "before any action was dispatched")
                return self._halt(current, "needs_review", "step_timeout", reason)
            if reobserved < contract.retry.reobserve:
                reobserved += 1
                task = current
                logger.debug("task reobserve id=%s step=%s attempt=%d", _short(task),
                             contract.name, reobserved + 1)
                continue
            return self._halt(current, "paused", "step_timeout",
                              f"Step {contract.name} timed out after {limit}; nothing was changed")

    async def _execute(self, task: dict[str, Any], contract: StepContract) -> dict[str, Any]:
        tag = _short(task)
        ctx = StepContext(task, self.desktop)
        logger.debug("task phase id=%s step=%s phase=inspect", tag, contract.name)
        observed = await _settle(contract.observe(ctx))
        post = _check(await _settle(contract.postcondition(ctx, observed)), contract, "postcondition")
        attempted = task["attempt"] is not None
        if post.held and (contract.operation == "read" or (
                contract.retry.reconcile if attempted else contract.retry.adopt_satisfied)):
            return self._advance(task, contract, observed, phase="inspect",
                                 reconciled=attempted and contract.operation == "mutate")
        if contract.action is None:
            reason = post.reason or f"Step {contract.name} postcondition not met"
            if contract.on_unmet == "pause":
                return self._halt(task, "paused", "waiting", reason)
            return self._halt(task, "needs_review", "needs_review", reason)
        if attempted:
            reason = contract.uncertain
            if contract.retry.reconcile and post.reason:
                reason = f"{reason}; {post.reason}"
            return self._halt(task, "needs_review", "needs_review", reason)

        self.phase = "act"
        logger.debug("task phase id=%s step=%s phase=act", tag, contract.name)
        pre = (_check(await _settle(contract.precondition(ctx, observed)), contract, "precondition")
               if contract.precondition else Check(True, handle=observed))
        if not pre.held:
            return self._halt(task, "needs_review", "needs_review",
                              pre.reason or f"Step {contract.name} precondition not met")
        reason = self._budget_reason(task, dispatching=True)
        if reason:
            return self._halt(task, "needs_review", "budget_exhausted", reason)
        task = self._record(task, "action_intent", attempt={"step": contract.name},
                            actions_taken=task["actions_taken"] + 1,
                            details={"step": contract.name, "phase": "act",
                                     "target": contract.target,
                                     "action_n": task["actions_taken"] + 1})
        logger.debug("task dispatch id=%s step=%s action_n=%d", tag, contract.name,
                     task["actions_taken"])
        ctx = StepContext(task, self.desktop)
        result = await _settle(contract.action(ctx, pre.handle))

        self.phase = "verify"
        logger.debug("task phase id=%s step=%s phase=verify", tag, contract.name)
        after = await _settle(contract.observe_after(ctx, observed, result)
                              if contract.observe_after else contract.observe(ctx))
        post = _check(await _settle(contract.postcondition(ctx, after)), contract, "postcondition")
        if not post.held:
            return self._halt(task, "needs_review", "needs_review",
                              f"Step {contract.name} postcondition not verified: {post.reason}")
        return self._advance(task, contract, after, phase="verify", reconciled=False)

    def _advance(self, task: dict[str, Any], contract: StepContract, observed: Any, *,
                 phase: str, reconciled: bool) -> dict[str, Any]:
        details = dict(contract.evidence(StepContext(task, self.desktop), observed, phase))
        if reconciled:
            details["reconciled"] = True
        steps = self.workflow.steps
        index = task["current_step"]
        statuses = list(task["step_status"])
        statuses[index] = "acknowledged" if contract.confirmed_by else "verified"
        for earlier, step in enumerate(steps[:index]):
            if step.confirmed_by == contract.name and statuses[earlier] == "acknowledged":
                statuses[earlier] = "verified"
        done = index + 1 == len(steps)
        task = self._record(
            task, "step_verified", details={"step": contract.name, "phase": phase, **details},
            current_step=index + 1, step_status=statuses,
            evidence=[*task["evidence"], {"step": contract.name, **details}], attempt=None,
            status="completed" if done else "running",
            result=self.workflow.result if done else None)
        logger.info("task step verified id=%s step=%s n=%d/%d%s", _short(task), contract.name,
                    index + 1, len(steps), " reconciled=true" if reconciled else "")
        if done:
            logger.info("task completed id=%s workflow=%s elapsed_s=%.1f actions=%d",
                        _short(task), task["workflow"], task["elapsed_seconds"],
                        task["actions_taken"])
        return task
