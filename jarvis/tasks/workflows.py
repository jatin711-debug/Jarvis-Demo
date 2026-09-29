"""Workflow registry and the two built-in desktop workflows.

Adding a workflow means defining a ``Workflow`` of step contracts, registering
it with ``register_workflow``, and providing a creator that returns a
``TaskSpec`` (see ``fixture_spec``). Neither the store nor the runner changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from jarvis.tasks.contracts import (
    Budget,
    Check,
    RetryPolicy,
    StepContext,
    StepContract,
    StepReview,
    TaskSpec,
    Workflow,
)

WORKFLOWS: dict[str, Workflow] = {}

_READ_TIMEOUT = 30.0
_ACT_TIMEOUT = 60.0
_LAUNCH_TIMEOUT = 90.0
_SAVE_POLLS = 20
_SAVE_POLL_SECONDS = 0.1
_TEXT_CONTROL = "Smoke text"
_SAVE_CONTROL = "Save smoke file"
_CALCULATOR_APPS = frozenset({"ApplicationFrameHost.exe", "CalculatorApp.exe", "Calculator.exe"})
_DISPLAY_PREFIX = "Display is "


def register_workflow(workflow: Workflow, *, replace: bool = False) -> Workflow:
    if not isinstance(workflow, Workflow):
        raise TypeError("register_workflow expects a Workflow")
    if workflow.name in WORKFLOWS and not replace:
        raise ValueError(f"Workflow already registered: {workflow.name}")
    WORKFLOWS[workflow.name] = workflow
    return workflow


def get_workflow(name: str) -> Workflow:
    try:
        return WORKFLOWS[name]
    except KeyError:
        raise ValueError(f"Unsupported workflow: {name}") from None


class _StepNames(Mapping[str, tuple[str, ...]]):
    """Read-only workflow name -> step names view over the live registry."""

    def __getitem__(self, name: str) -> tuple[str, ...]:
        try:
            return WORKFLOWS[name].step_names
        except KeyError:
            raise KeyError(name) from None

    def __iter__(self) -> Iterator[str]:
        return iter(WORKFLOWS)

    def __len__(self) -> int:
        return len(WORKFLOWS)


WORKFLOW_STEPS: Mapping[str, tuple[str, ...]] = _StepNames()


def _titled(ctx: StepContext, listing: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [window for window in listing["windows"]
            if window.get("title") == ctx.scope["window_title"]]


async def _observe_window(ctx: StepContext, apps: frozenset[str] | None = None) -> dict[str, Any]:
    listing = await ctx.desktop.windows()
    if listing.get("truncated"):
        raise RuntimeError("Window list is truncated; cannot prove a unique target")
    matches = _titled(ctx, listing)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact target window; found {len(matches)}")
    if apps is not None and matches[0].get("app_name") not in apps:
        raise RuntimeError("Window title belongs to an unexpected app")
    return await ctx.desktop.observe(matches[0]["pid"], matches[0]["window_id"])


def _control(state: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    matches = [element for element in state["elements"]
               if element.get("label") == label and element.get("enabled") is not False]
    # Cua may return a valid target token while elements_complete is false.
    # Act only on the exact returned control; never infer absent controls
    # from an incomplete tree.
    if state.get("degraded") or state.get("truncated") or len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify one enabled '{label}' control "
            f"(matches={len(matches)}, degraded={state.get('degraded')}, "
            f"truncated={state.get('truncated')}, complete={state.get('elements_complete')})"
        )
    if not state.get("snapshot_id") or matches[0].get("element_index") is None:
        raise RuntimeError("Control lacks a fresh action target")
    return matches[0]


async def _click(ctx: StepContext, handle: tuple[Mapping[str, Any], Mapping[str, Any]]) -> Any:
    state, control = handle
    return await ctx.desktop.act("click", state["pid"], state["window_id"],
                                 state["snapshot_id"], control["element_index"])


def _after(result: Any, message: str) -> Mapping[str, Any]:
    after = result.get("after") if isinstance(result, Mapping) else None
    if not after:
        raise StepReview(message)
    return after


async def _file_state(ctx: StepContext) -> str:
    target = Path(ctx.scope["output_path"])
    wanted = ctx.input["text"]

    def check() -> str:
        if not target.exists():
            return "absent"
        return "match" if target.read_text(encoding="utf-8-sig") == wanted else "mismatch"

    return await asyncio.to_thread(check)


def _file_matches(ctx: StepContext, state: str) -> Check:
    return Check(state == "match", f"output file is {state}")


def _file_evidence(ctx: StepContext, state: str, phase: str) -> dict[str, Any]:
    return {"file": "exact bytes matched", "source": "independent read"}


def fixture_spec(*, window_title: str, output_path: Path, text: str) -> TaskSpec:
    if not window_title.startswith("Jarvis Cua Smoke ") or len(window_title) < 20:
        raise ValueError("This first workflow accepts only a uniquely named Jarvis Cua Smoke fixture")
    target = Path(output_path).resolve()
    if not target.is_absolute() or not target.parent.is_dir():
        raise ValueError("The output directory must already exist")
    if not text or len(text) > 2000:
        raise ValueError("Fixture text must contain 1-2000 characters")
    return TaskSpec(goal="Save text in isolated desktop fixture",
                    scope={"window_title": window_title, "output_path": str(target)},
                    input={"text": text})


def calculator_spec() -> TaskSpec:
    return TaskSpec(goal="Calculate 1+2 in Windows Calculator",
                    scope={"app": "Calculator", "window_title": "Calculator"},
                    input={"expression": "1+2", "expected": "3"})


def _fixture_permissions(task: Mapping[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    path = task["scope"]["output_path"]
    return (("read_file", {"file_path": path}),
            ("write_file", {"file_path": path, "content": task["input"]["text"]}))


def _calculator_permissions(task: Mapping[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    return (("desktop_launch", {"name": task["scope"]["app"]}),)


# desktop_save_fixture ------------------------------------------------------

async def _list_windows(ctx: StepContext) -> Mapping[str, Any]:
    return await ctx.desktop.windows()


def _fixture_window_found(ctx: StepContext, listing: Mapping[str, Any]) -> Check:
    if listing.get("truncated"):
        return Check(False, "Window list is truncated; cannot prove a unique target")
    found = len(_titled(ctx, listing))
    if found != 1:
        return Check(False, f"Expected one exact fixture window; found {found}")
    return Check(True)


async def _observe_editor(ctx: StepContext) -> dict[str, Any]:
    state = await _observe_window(ctx)
    _control(state, _TEXT_CONTROL)
    return state


def _editor_holds_text(ctx: StepContext, state: Mapping[str, Any]) -> Check:
    editors = [element for element in state["elements"] if element.get("label") == _TEXT_CONTROL]
    if len(editors) == 1 and editors[0].get("value") == ctx.input["text"]:
        return Check(True)
    return Check(False, "editor does not show the requested text")


def _editor_target(ctx: StepContext, state: Mapping[str, Any]) -> Check:
    return Check(True, handle=(state, _control(state, _TEXT_CONTROL)))


async def _type_text(ctx: StepContext, handle: tuple[Mapping[str, Any], Mapping[str, Any]]) -> Any:
    state, editor = handle
    return await ctx.desktop.act("type_text", state["pid"], state["window_id"],
                                 state["snapshot_id"], editor["element_index"],
                                 text=ctx.input["text"])


async def _editor_after(ctx: StepContext, state: Mapping[str, Any], result: Any) -> Any:
    return _after(result, "Text dispatch returned no post-action observation")


async def _save_target(ctx: StepContext, file_state: str) -> Check:
    if file_state == "mismatch":
        return Check(False, "Output path already exists with different contents")
    state = await _observe_window(ctx)
    return Check(True, handle=(state, _control(state, _SAVE_CONTROL)))


async def _await_saved_file(ctx: StepContext, before: str, result: Any) -> str:
    state = "absent"
    for _ in range(_SAVE_POLLS):
        state = await _file_state(ctx)
        if state != "absent":
            break
        await asyncio.sleep(_SAVE_POLL_SECONDS)
    return state


FIXTURE_WORKFLOW = register_workflow(Workflow(
    name="desktop_save_fixture",
    steps=(
        StepContract(
            name="discover_window", target="the exact Jarvis Cua Smoke window title",
            operation="read", observe=_list_windows, postcondition=_fixture_window_found,
            evidence=lambda ctx, listing, phase: {"window_title": ctx.scope["window_title"]},
            on_unmet="pause", timeout=_READ_TIMEOUT, retry=RetryPolicy(reobserve=1),
        ),
        StepContract(
            name="enter_text", target=f"'{_TEXT_CONTROL}' control in the fixture window",
            operation="mutate", observe=_observe_editor, postcondition=_editor_holds_text,
            precondition=_editor_target, action=_type_text, observe_after=_editor_after,
            evidence=lambda ctx, state, phase: {"observed_value": "matched"},
            timeout=_ACT_TIMEOUT,
            uncertain="Previous text action has an uncertain outcome; inspect the fixture",
        ),
        StepContract(
            name="save_file", target="the output file via the fixture Save control",
            operation="mutate", observe=_file_state, postcondition=_file_matches,
            precondition=_save_target, action=_click, observe_after=_await_saved_file,
            evidence=_file_evidence, timeout=_ACT_TIMEOUT,
            uncertain="Previous Save action may have reached the app; no automatic replay",
        ),
        StepContract(
            name="verify_file", target="the output file bytes",
            operation="read", observe=_file_state, postcondition=_file_matches,
            evidence=_file_evidence, timeout=_READ_TIMEOUT, retry=RetryPolicy(reobserve=1),
        ),
    ),
    result="File contents verified",
    budget=Budget(max_seconds=180, max_actions=4),
    permissions=_fixture_permissions,
))


# calculator_1_plus_2 -------------------------------------------------------

def _display(state: Mapping[str, Any]) -> str | None:
    shown = [element["label"].removeprefix(_DISPLAY_PREFIX) for element in state["elements"]
             if isinstance(element.get("label"), str)
             and element["label"].startswith(_DISPLAY_PREFIX)]
    return shown[0].replace(",", "").strip() if len(shown) == 1 else None


def _calculator_found(ctx: StepContext, listing: Mapping[str, Any]) -> Check:
    if listing.get("truncated"):
        return Check(False, "Window list truncated; exact Calculator target is uncertain")
    matches = _titled(ctx, listing)
    if not matches:
        return Check(False, "Calculator window not found")
    if len(matches) != 1 or matches[0].get("app_name") not in _CALCULATOR_APPS:
        return Check(False, "Calculator target is ambiguous or not the expected app")
    return Check(True)


def _calculator_absent(ctx: StepContext, listing: Mapping[str, Any]) -> Check:
    if listing.get("truncated"):
        return Check(False, "Window list truncated; exact Calculator target is uncertain")
    if _titled(ctx, listing):
        return Check(False, "Calculator target is ambiguous or not the expected app")
    return Check(True)


async def _launch_calculator(ctx: StepContext, listing: Mapping[str, Any]) -> Any:
    return await ctx.desktop.launch(ctx.scope["app"])


def _calculator_evidence(ctx: StepContext, listing: Mapping[str, Any], phase: str) -> dict[str, Any]:
    window = _titled(ctx, listing)[0]
    return {"target": {"pid": window["pid"], "window_id": window["window_id"],
                       "title": window["title"]},
            "source": "Cua window inventory"}


async def _observe_calculator(ctx: StepContext) -> dict[str, Any]:
    state = await _observe_window(ctx, _CALCULATOR_APPS)
    if (state.get("title") != ctx.scope["window_title"] or state.get("degraded")
            or state.get("truncated")):
        raise StepReview("Calculator observation is incomplete or target changed")
    if _display(state) is None:
        raise StepReview("Cannot read one Calculator display")
    return state


def _calculator_click(name: str, label: str, *, before: str | None, after: str,
                      retry: RetryPolicy = RetryPolicy(), uncertain: str | None = None,
                      confirmed_by: str | None = None, note: str = "") -> StepContract:
    def postcondition(ctx: StepContext, state: Mapping[str, Any]) -> Check:
        shown = _display(state)
        return Check(shown == after, f"observed display "
                     f"{shown if shown is not None else 'unreadable'}, expected {after}")

    def precondition(ctx: StepContext, state: Mapping[str, Any]) -> Check:
        shown = _display(state)
        if before is not None and shown != before:
            return Check(False, f"Calculator changed before {label}: expected display "
                                f"{before}, observed {shown}")
        return Check(True, handle=(state, _control(state, label)))

    async def observe_after(ctx: StepContext, state: Mapping[str, Any], result: Any) -> Any:
        fresh = _after(result, f"{label} dispatch has no trustworthy post-action observation")
        if (fresh.get("pid") != state["pid"] or fresh.get("window_id") != state["window_id"]
                or fresh.get("degraded") or fresh.get("truncated")):
            raise StepReview(f"{label} dispatch has no trustworthy post-action observation")
        return fresh

    def evidence(ctx: StepContext, state: Mapping[str, Any], phase: str) -> dict[str, Any]:
        if phase == "inspect":
            return {"display": _display(state), "source": "fresh Cua observation"}
        return {"display": _display(state), "source": "Cua post-action observation",
                "control": label, "note": note}

    return StepContract(
        name=name, target=f"'{label}' control in the Calculator window", operation="mutate",
        observe=_observe_calculator, postcondition=postcondition, precondition=precondition,
        action=_click, observe_after=observe_after, evidence=evidence, timeout=_ACT_TIMEOUT,
        retry=retry, uncertain=uncertain or f"Prior {label} action uncertain",
        confirmed_by=confirmed_by,
    )


CALCULATOR_WORKFLOW = register_workflow(Workflow(
    name="calculator_1_plus_2",
    steps=(
        StepContract(
            name="discover_calculator", target="the exact Calculator window (launch if absent)",
            operation="mutate", observe=_list_windows, postcondition=_calculator_found,
            precondition=_calculator_absent, action=_launch_calculator,
            evidence=_calculator_evidence, timeout=_LAUNCH_TIMEOUT,
            uncertain="Calculator launch outcome uncertain; do not launch twice",
        ),
        _calculator_click("clear", "Clear", before=None, after="0"),
        _calculator_click("enter_one", "One", before="0", after="1",
                          retry=RetryPolicy(adopt_satisfied=False)),
        _calculator_click(
            "plus", "Plus", before="1", after="1", confirmed_by="equals",
            retry=RetryPolicy(reconcile=False, adopt_satisfied=False),
            uncertain="Plus dispatch uncertain; display cannot prove operation",
            note="Plus is confirmed by dispatch; final arithmetic result verifies sequence"),
        _calculator_click("enter_two", "Two", before="1", after="2",
                          retry=RetryPolicy(adopt_satisfied=False)),
        _calculator_click("equals", "Equals", before="2", after="3",
                          retry=RetryPolicy(adopt_satisfied=False)),
    ),
    result="Calculator displayed 3 after 1+2",
    budget=Budget(max_seconds=300, max_actions=8),
    permissions=_calculator_permissions,
    # Calculator's lower keypad can be omitted by the compact default tree.
    desktop_minimums={"max_elements": 80, "max_depth": 10},
))
