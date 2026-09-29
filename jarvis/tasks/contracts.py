"""Declarative step contracts for durable task workflows.

A workflow is data: an ordered tuple of ``StepContract`` objects. The engine in
``jarvis.tasks.runner`` owns sequencing, journaling, timeouts, and budgets; a
contract only states what a step touches and how to observe, act on, and prove
it. Definitions are validated when they are constructed, so a malformed
workflow fails at import or registration rather than mid-run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Operation = Literal["read", "mutate"]
Unmet = Literal["review", "pause"]

MAX_STEP_TIMEOUT = 3600.0
MAX_BUDGET_SECONDS = 86400.0
MAX_BUDGET_ACTIONS = 1000
DESKTOP_LIMITS = frozenset({"max_elements", "max_depth"})
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


class StepReview(RuntimeError):
    """Raised by a step callable to stop the task for review with this reason."""


@dataclass(frozen=True)
class Check:
    """Outcome of a precondition or postcondition.

    ``handle`` is what a precondition resolved for the action to use (for
    example the exact control and snapshot); it is passed to ``action``.
    """

    held: bool
    reason: str = ""
    handle: Any = None


@dataclass(frozen=True)
class Budget:
    """Caps for one task: run time across resumes and mutating actions."""

    max_seconds: float = 300.0
    max_actions: int = 10

    def __post_init__(self) -> None:
        seconds = self.max_seconds
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or not 0 < seconds <= MAX_BUDGET_SECONDS):
            raise ValueError(f"max_seconds must be in (0, {MAX_BUDGET_SECONDS:g}]")
        actions = self.max_actions
        if isinstance(actions, bool) or not isinstance(actions, int) \
                or not 0 <= actions <= MAX_BUDGET_ACTIONS:
            raise ValueError(f"max_actions must be an integer in [0, {MAX_BUDGET_ACTIONS}]")

    def as_state(self) -> dict[str, Any]:
        return {"max_seconds": float(self.max_seconds), "max_actions": self.max_actions}

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> Budget:
        return cls(max_seconds=float(value["max_seconds"]), max_actions=int(value["max_actions"]))


DEFAULT_BUDGET = Budget()


@dataclass(frozen=True)
class RetryPolicy:
    """How the engine may repeat work. A dispatched mutation is never replayed.

    ``reobserve``: extra observation attempts after a read-only step times out.
    ``reconcile``: after an interrupted dispatch, a postcondition seen in a
    fresh observation completes the step. Turn off when an observation cannot
    prove the operation happened (an operator that leaves the display alone).
    ``adopt_satisfied``: with no journaled attempt, a postcondition that
    already holds completes the step without acting. Turn off when a held
    postcondition would only be a coincidence.
    """

    reobserve: int = 0
    reconcile: bool = True
    adopt_satisfied: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.reobserve, bool) or not isinstance(self.reobserve, int) \
                or not 0 <= self.reobserve <= 5:
            raise ValueError("retry.reobserve must be an integer in [0, 5]")
        if not isinstance(self.reconcile, bool) or not isinstance(self.adopt_satisfied, bool):
            raise ValueError("retry.reconcile and retry.adopt_satisfied must be booleans")


@dataclass(frozen=True)
class StepContext:
    """What a step callable can see: the persisted task and the desktop backend."""

    task: Mapping[str, Any]
    desktop: Any

    @property
    def scope(self) -> Mapping[str, Any]:
        return self.task["scope"]

    @property
    def input(self) -> Mapping[str, Any]:
        return self.task["input"]


# Callables may be plain functions or coroutine functions unless noted.
Observer = Callable[[StepContext], Awaitable[Any]]
Precondition = Callable[[StepContext, Any], Check | Awaitable[Check]]
Postcondition = Callable[[StepContext, Any], Check | Awaitable[Check]]
Action = Callable[[StepContext, Any], Awaitable[Any]]
AfterObserver = Callable[[StepContext, Any, Any], Awaitable[Any]]
EvidenceBuilder = Callable[[StepContext, Any, str], Mapping[str, Any]]


def _check_name(kind: str, value: Any) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{kind} name must be lower_snake_case, got {value!r}")


@dataclass(frozen=True)
class StepContract:
    """One step: what it touches, how to observe it, act on it, and prove it.

    ``observe`` takes a fresh observation or independent read (the engine
    calls it first to see whether the postcondition already holds).
    ``precondition`` runs on that observation before anything is dispatched
    and resolves the exact target into ``Check.handle``. ``action`` dispatches
    the mutation. ``observe_after`` produces the observation the postcondition
    is judged on after the dispatch (default: observe again). ``evidence`` is
    synchronous and builds the recorded proof from an observation and the
    phase ("inspect" or "verify").
    """

    name: str
    target: str
    operation: Operation
    observe: Observer
    postcondition: Postcondition
    evidence: EvidenceBuilder
    precondition: Precondition | None = None
    action: Action | None = None
    observe_after: AfterObserver | None = None
    timeout: float = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    uncertain: str = "Previous action has an uncertain outcome; no automatic replay"
    on_unmet: Unmet = "review"
    confirmed_by: str | None = None

    def __post_init__(self) -> None:
        _check_name("Step", self.name)
        label = f"Step {self.name}"
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError(f"{label}: target must say what the step touches")
        if self.operation not in ("read", "mutate"):
            raise ValueError(f"{label}: operation must be 'read' or 'mutate'")
        for attribute in ("observe", "postcondition", "evidence"):
            if not callable(getattr(self, attribute)):
                raise ValueError(f"{label}: {attribute} must be callable")
        for attribute in ("precondition", "action", "observe_after"):
            value = getattr(self, attribute)
            if value is not None and not callable(value):
                raise ValueError(f"{label}: {attribute} must be callable or None")
        timeout = self.timeout
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= MAX_STEP_TIMEOUT):
            raise ValueError(f"{label}: timeout must be in (0, {MAX_STEP_TIMEOUT:g}] seconds")
        if not isinstance(self.retry, RetryPolicy):
            raise ValueError(f"{label}: retry must be a RetryPolicy")
        if self.on_unmet not in ("review", "pause"):
            raise ValueError(f"{label}: on_unmet must be 'review' or 'pause'")
        if self.confirmed_by is not None:
            _check_name("confirmed_by", self.confirmed_by)
        if self.operation == "read":
            for attribute in ("action", "precondition", "observe_after"):
                if getattr(self, attribute) is not None:
                    raise ValueError(f"{label}: a read-only step cannot declare {attribute}")
            if self.confirmed_by is not None:
                raise ValueError(f"{label}: only a mutating step can await confirmation")
        else:
            if self.action is None:
                raise ValueError(f"{label}: a mutating step must declare an action")
            if self.on_unmet != "review":
                raise ValueError(f"{label}: a mutating step stops for review, it cannot pause")
            if self.retry.reobserve:
                raise ValueError(f"{label}: a mutating step is never retried automatically")
            if not isinstance(self.uncertain, str) or not self.uncertain.strip():
                raise ValueError(f"{label}: a mutating step needs an uncertain-outcome message")


PermissionChecks = Callable[[Mapping[str, Any]], Sequence[tuple[str, Mapping[str, Any]]]]


@dataclass(frozen=True)
class Workflow:
    """A named, ordered set of step contracts plus its default budget.

    ``permissions`` lists the permission-gateway checks (tool name, arguments)
    the task's scope must pass before any step runs. ``desktop_minimums`` are
    observation limits (``max_elements``, ``max_depth``) the CLI raises the
    desktop config to for this workflow.
    """

    name: str
    steps: tuple[StepContract, ...]
    result: str
    budget: Budget = DEFAULT_BUDGET
    permissions: PermissionChecks | None = None
    desktop_minimums: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_name("Workflow", self.name)
        object.__setattr__(self, "steps", tuple(self.steps))
        label = f"Workflow {self.name}"
        if not self.steps:
            raise ValueError(f"{label}: at least one step is required")
        if not all(isinstance(step, StepContract) for step in self.steps):
            raise ValueError(f"{label}: steps must be StepContract objects")
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{label}: step names must be unique")
        if not isinstance(self.result, str) or not self.result.strip():
            raise ValueError(f"{label}: result must describe the verified outcome")
        if not isinstance(self.budget, Budget):
            raise ValueError(f"{label}: budget must be a Budget")
        mutating = sum(step.operation == "mutate" for step in self.steps)
        if self.budget.max_actions < mutating:
            raise ValueError(f"{label}: default budget allows {self.budget.max_actions} actions "
                             f"but {mutating} steps mutate")
        for index, step in enumerate(self.steps):
            if step.confirmed_by is not None and step.confirmed_by not in names[index + 1:]:
                raise ValueError(f"{label}: step {step.name} must be confirmed by a later step")
        if self.permissions is not None and not callable(self.permissions):
            raise ValueError(f"{label}: permissions must be callable or None")
        unknown = set(self.desktop_minimums) - DESKTOP_LIMITS
        if unknown or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                          for value in self.desktop_minimums.values()):
            raise ValueError(f"{label}: desktop_minimums allows positive integers for "
                             f"{sorted(DESKTOP_LIMITS)}")

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)


@dataclass(frozen=True)
class TaskSpec:
    """The creator's output: what a new task is for, what it may touch, its input."""

    goal: str
    scope: Mapping[str, Any]
    input: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("A task needs a goal")
        if not isinstance(self.scope, Mapping) or not isinstance(self.input, Mapping):
            raise ValueError("Task scope and input must be mappings")
