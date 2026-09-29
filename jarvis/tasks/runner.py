"""Small deterministic workflow with checkpoints and independent postconditions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from jarvis.agent.permissions import PermissionGateway
from jarvis.tasks.store import WORKFLOW_STEPS, StateConflict, TaskStore


class TaskRunner:
    def __init__(self, store: TaskStore, desktop: Any,
                 permission_gateway: PermissionGateway | None = None) -> None:
        self.store = store
        self.desktop = desktop
        self.permission_gateway = permission_gateway

    async def run(self, task_id: str, *, pause_after: int | None = None) -> dict[str, Any]:
        if pause_after is not None and pause_after < 1:
            raise ValueError("pause_after must be positive")
        task = self.store.get(task_id)
        if task["workflow"] not in WORKFLOW_STEPS:
            raise ValueError("Unsupported workflow")
        steps = WORKFLOW_STEPS[task["workflow"]]
        if task["status"] in {"completed", "cancelled"}:
            return task
        if self.permission_gateway is not None:
            if task["workflow"] == "desktop_save_fixture":
                checks = (
                    ("read_file", {"file_path": task["scope"]["output_path"]}),
                    ("write_file", {"file_path": task["scope"]["output_path"],
                                    "content": task["input"]["text"]}),
                )
            else:
                checks = (("desktop_launch", {"name": "Calculator"}),)
            for name, arguments in checks:
                decision = self.permission_gateway.inspect(name, arguments)
                if not decision.allowed:
                    return self.store.record(task, "permission_denied", status="needs_review",
                                             error=decision.reason,
                                             details={"tool": name, "target": decision.target})
        task = self.store.record(task, "resumed" if task["status"] != "pending" else "started",
                                 status="running", error=None)
        completed_steps = 0
        try:
            while task["current_step"] < len(steps):
                step = task["current_step"]
                if task["workflow"] == "calculator_1_plus_2":
                    task = await self._calculator_step(task)
                elif step == 0:
                    task = await self._discover(task)
                elif step == 1:
                    task = await self._type(task)
                elif step == 2:
                    task = await self._save(task)
                else:
                    task = await self._verify(task)
                if task["status"] != "running":
                    return task
                completed_steps += 1
                if pause_after is not None and completed_steps >= pause_after:
                    return self.store.record(task, "paused_at_boundary", status="paused",
                                             details={"next_step": task["current_step"]})
            return task
        except StateConflict:
            raise
        except asyncio.CancelledError:
            current = self.store.get(task_id)
            if current["status"] not in {"completed", "cancelled"}:
                self.store.record(current, "interrupted", status="needs_review",
                                  error="Execution interrupted; inspect outcome before resuming")
            raise
        except Exception as exc:
            current = self.store.get(task_id)
            if current["status"] in {"completed", "cancelled"}:
                return current
            return self.store.record(current, "step_error", status="needs_review", error=str(exc),
                                     details={"step": steps[current["current_step"]]})

    def _advance(self, task: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
        statuses = list(task["step_status"])
        calculator_plus = task["workflow"] == "calculator_1_plus_2" and task["current_step"] == 3
        statuses[task["current_step"]] = "acknowledged" if calculator_plus else "verified"
        steps = WORKFLOW_STEPS[task["workflow"]]
        evidence = [*task["evidence"], {"step": steps[task["current_step"]], **details}]
        next_step = task["current_step"] + 1
        if task["workflow"] == "calculator_1_plus_2" and next_step == len(steps):
            # The operator itself leaves the display unchanged. Only the
            # final observed arithmetic result confirms that it was applied.
            statuses[3] = "verified"
        return self.store.record(task, "step_verified", details=details,
                                 current_step=next_step, step_status=statuses,
                                 evidence=evidence, attempt=None,
                                 status="completed" if next_step == len(steps) else "running",
                                 result=("File contents verified" if task["workflow"] == "desktop_save_fixture"
                                         else "Calculator displayed 3 after 1+2")
                                 if next_step == len(steps) else None)

    def _review(self, task: dict[str, Any], reason: str) -> dict[str, Any]:
        return self.store.record(task, "needs_review", status="needs_review", error=reason,
                                 details={"step": WORKFLOW_STEPS[task["workflow"]][task["current_step"]]})

    async def _window(self, task: dict[str, Any]) -> dict[str, Any]:
        listing = await self.desktop.windows()
        if listing.get("truncated"):
            raise RuntimeError("Window list is truncated; cannot prove a unique target")
        matches = [window for window in listing["windows"]
                   if window.get("title") == task["scope"]["window_title"]]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one exact fixture window; found {len(matches)}")
        if (task["workflow"] == "calculator_1_plus_2"
                and matches[0].get("app_name") not in {
                    "ApplicationFrameHost.exe", "CalculatorApp.exe", "Calculator.exe"}):
            raise RuntimeError("Calculator title belongs to an unexpected app")
        return matches[0]

    async def _observe(self, task: dict[str, Any]) -> dict[str, Any]:
        window = await self._window(task)
        return await self.desktop.observe(window["pid"], window["window_id"])

    @staticmethod
    def _control(state: dict[str, Any], label: str) -> dict[str, Any]:
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

    async def _file_state(self, task: dict[str, Any]) -> str:
        target = Path(task["scope"]["output_path"])

        def check() -> str:
            if not target.exists():
                return "absent"
            return "match" if target.read_text(encoding="utf-8-sig") == task["input"]["text"] else "mismatch"

        return await asyncio.to_thread(check)

    async def _discover(self, task: dict[str, Any]) -> dict[str, Any]:
        try:
            window = await self._window(task)
        except RuntimeError as exc:
            return self.store.record(task, "waiting_for_window", status="paused", error=str(exc))
        return self._advance(task, {"window_title": window["title"]})

    async def _type(self, task: dict[str, Any]) -> dict[str, Any]:
        state = await self._observe(task)
        editor = self._control(state, "Smoke text")
        desired = task["input"]["text"]
        if editor.get("value") == desired:
            return self._advance(task, {"observed_value": "matched"})
        if task["attempt"] is not None:
            return self._review(task, "Previous text action has an uncertain outcome; inspect the fixture")
        task = self.store.record(task, "action_intent", attempt={"step": "enter_text"},
                                 details={"step": "enter_text", "target": "Smoke text"})
        result = await self.desktop.act("type_text", state["pid"], state["window_id"],
                                        state["snapshot_id"], editor["element_index"], text=desired)
        after = result.get("after")
        if not after:
            return self._review(task, "Text dispatch returned no post-action observation")
        matches = [element for element in after["elements"] if element.get("label") == "Smoke text"]
        if len(matches) != 1 or matches[0].get("value") != desired:
            return self._review(task, "Text value was not verified after dispatch")
        return self._advance(task, {"observed_value": "matched"})

    async def _save(self, task: dict[str, Any]) -> dict[str, Any]:
        file_state = await self._file_state(task)
        if file_state == "match":
            return self._advance(task, {"file": "exact bytes matched", "source": "independent read"})
        if file_state == "mismatch":
            return self._review(task, "Output path already exists with different contents")
        if task["attempt"] is not None:
            return self._review(task, "Previous Save action may have reached the app; no automatic replay")
        state = await self._observe(task)
        button = self._control(state, "Save smoke file")
        task = self.store.record(task, "action_intent", attempt={"step": "save_file"},
                                 details={"step": "save_file", "target": "Save smoke file"})
        await self.desktop.act("click", state["pid"], state["window_id"],
                               state["snapshot_id"], button["element_index"])
        for _ in range(20):
            file_state = await self._file_state(task)
            if file_state != "absent":
                break
            await asyncio.sleep(0.1)
        if file_state != "match":
            return self._review(task, f"Save outcome not verified: {file_state}")
        return self._advance(task, {"file": "exact bytes matched", "source": "independent read"})

    async def _verify(self, task: dict[str, Any]) -> dict[str, Any]:
        file_state = await self._file_state(task)
        if file_state != "match":
            return self._review(task, f"Final file verification failed: {file_state}")
        return self._advance(task, {"file": "exact bytes matched", "source": "independent read"})

    async def _calculator_step(self, task: dict[str, Any]) -> dict[str, Any]:
        step = task["current_step"]
        if step == 0:
            listing = await self.desktop.windows()
            if listing.get("truncated"):
                return self._review(task, "Window list truncated; exact Calculator target is uncertain")
            matches = [w for w in listing["windows"] if w.get("title") == "Calculator"]
            if not matches:
                if task["attempt"] is not None:
                    return self._review(task, "Calculator launch outcome uncertain; do not launch twice")
                task = self.store.record(task, "action_intent", attempt={"step": "discover_calculator"},
                                         details={"app": "Calculator", "action": "launch"})
                await self.desktop.launch("Calculator")
                listing = await self.desktop.windows()
                if listing.get("truncated"):
                    return self._review(task, "Window list truncated after Calculator launch")
                matches = [w for w in listing["windows"] if w.get("title") == "Calculator"]
            if len(matches) != 1 or matches[0].get("app_name") not in {
                    "ApplicationFrameHost.exe", "CalculatorApp.exe", "Calculator.exe"}:
                return self._review(task, "Calculator target is absent, ambiguous, or not the expected app")
            return self._advance(task, {"target": {"pid": matches[0]["pid"],
                                                    "window_id": matches[0]["window_id"],
                                                    "title": matches[0]["title"]},
                                        "source": "Cua window inventory"})

        state = await self._observe(task)
        if state.get("title") != "Calculator" or state.get("degraded") or state.get("truncated"):
            return self._review(task, "Calculator observation is incomplete or target changed")
        displays = [e["label"].removeprefix("Display is ") for e in state["elements"]
                    if isinstance(e.get("label"), str) and e["label"].startswith("Display is ")]
        if len(displays) != 1:
            return self._review(task, "Cannot read one Calculator display")
        display = displays[0].replace(",", "").strip()
        labels = ("Clear", "One", "Plus", "Two", "Equals")
        expected_before = (None, "0", "0", "1", "1", "2")
        expected_after = (None, "0", "1", "1", "2", "3")
        label = labels[step - 1]
        if task["attempt"] is not None:
            # Plus leaves the same display, so UIA cannot establish that it
            # happened after a lost acknowledgement. Stop rather than replay.
            if step == 3:
                return self._review(task, "Plus dispatch uncertain; display cannot prove operation")
            if display == expected_after[step]:
                return self._advance(task, {"display": display, "source": "fresh Cua observation",
                                            "reconciled": True})
            return self._review(task, f"Prior {label} action uncertain; observed display {display}")
        if step == 1 and display == "0":
            return self._advance(task, {"display": display, "source": "fresh Cua observation"})
        if step != 1 and display != expected_before[step]:
            return self._review(task, f"Calculator changed before {label}: expected display "
                              f"{expected_before[step]}, observed {display}")
        control = self._control(state, label)
        task = self.store.record(task, "action_intent", attempt={"step": WORKFLOW_STEPS[task["workflow"]][step]},
                                 details={"target": "Calculator", "control": label,
                                          "display_before": display})
        result = await self.desktop.act("click", state["pid"], state["window_id"],
                                        state["snapshot_id"], control["element_index"])
        after = result.get("after")
        if (not after or after.get("pid") != state["pid"]
                or after.get("window_id") != state["window_id"]
                or after.get("degraded") or after.get("truncated")):
            return self._review(task, f"{label} dispatch has no trustworthy post-action observation")
        observed = [e["label"].removeprefix("Display is ") for e in after["elements"]
                    if isinstance(e.get("label"), str) and e["label"].startswith("Display is ")]
        if len(observed) != 1 or observed[0].replace(",", "").strip() != expected_after[step]:
            return self._review(task, f"{label} postcondition not verified: {observed}")
        return self._advance(task, {"display": expected_after[step],
                                    "source": "Cua post-action observation",
                                    "control": label,
                                    "note": "Plus is confirmed by dispatch; final arithmetic result verifies sequence"
                                    if step == 3 else ""})
