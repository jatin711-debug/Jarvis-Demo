"""Bounded, window-scoped Cua integration using the pinned Python SDK.

Keep native imports lazy so voice-only installs and CI do not need Cua. The
generic SDK surface preserves delivery_mode and element_token for text/key
actions, which the 0.28.2 generated convenience inputs do not expose.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, NamedTuple

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.config import DesktopConfig
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

_STALE = "Missing or stale observation; call desktop_observe again"


class _Snapshot(NamedTuple):
    at: float
    snapshot_id: str
    tokens: dict[int, str]  # element_index -> action token, for enabled elements only
    elements: tuple[dict[str, Any], ...]  # the controls as the model was shown them


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _forms(text: Any) -> frozenset[str]:
    """A label as written, plus the other way to write a digit key: "7" is also "seven".

    Apps name keypad buttons either way (Windows Calculator says "Seven"). A small model
    copies the digits of "144" reliably but turns them into words badly, so either is accepted.
    """
    norm = _norm(text)
    if len(norm) == 1 and norm.isdecimal():
        return frozenset({norm, _DIGIT_WORDS[int(norm)]})
    if norm in _DIGIT_WORDS:
        return frozenset({norm, str(_DIGIT_WORDS.index(norm))})
    return frozenset({norm})


def _find_control(snapshot: _Snapshot, label: str | None, index: int | None) -> dict[str, Any]:
    """The one enabled control a label (or index) names in this snapshot; never a guess.

    Labels match exactly (ignoring case and spacing; a digit key may be given as a digit or
    a word). An unknown or shared label is an error that lists the candidates, so the model
    corrects itself instead of the wrong control being clicked.
    """
    usable = [e for e in snapshot.elements if e["element_index"] in snapshot.tokens]
    wanted = _norm(label)
    if wanted:
        wanted_forms = _forms(label)
        matches = [e for e in usable if wanted_forms & _forms(e.get("label"))]
        if index is not None:
            matches = [e for e in matches if e["element_index"] == index]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(f'{len(matches)} controls are labeled "{label}": '
                             + "; ".join(_describe_control(e) for e in matches)
                             + ". Repeat the call with index set to one of these numbers.")
        similar = [e for e in usable if _norm(e.get("label"))
                   and (wanted in _norm(e["label"]) or _norm(e["label"]) in wanted)]
        shown = "Similar labels: " if similar else "Labels: "
        names = "; ".join(_describe_control(e) for e in (similar or usable))
        raise ValueError(f'No enabled control is labeled "{label}". {shown}{names[:700]}')
    if index is not None:
        for element in usable:
            if element["element_index"] == index:
                return element
        raise ValueError(f"Control {index} is not enabled or not in the latest observation")
    raise ValueError("Name the control by its label exactly as desktop_observe lists it "
                     "(or give its index)")


def _digit_keys(digits: str | None) -> list[str] | None:
    """The keys of a ``digits`` argument ("144" -> 1, 4, 4); None when there is none."""
    if digits is None or digits == "":
        return None
    text = str(digits).strip()
    if not (text.isascii() and text.isdecimal() and len(text) <= 20):
        raise ValueError("digits must be 1 to 20 characters from 0-9 and nothing else; "
                         "name other keys by label, one call each")
    return list(text)


def _clip(text: Any, limit: int) -> str:
    flat = " ".join(str(text).split()).replace('"', "'")
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


_PLAIN_ACTIONS = frozenset({"invoke", "text"})


def _describe_control(element: dict[str, Any]) -> str:
    """One control as `index Role "label" = "value"`, the form the model reads and answers in."""
    line = f'{element["element_index"]} {element.get("role")} "{_clip(element.get("label") or "", 80)}"'
    value = element.get("value")
    if value not in (None, "") and value != element.get("label"):
        line += f' = "{_clip(value, 200)}"'
    if element.get("enabled") is False:
        line += " (disabled)"
    extra = [a for a in element.get("actions") or [] if a not in _PLAIN_ACTIONS]
    if extra:
        line += " [" + ",".join(extra) + "]"
    return line


def _describe_window(window: dict[str, Any]) -> str:
    line = f'pid={window["pid"]} window_id={window["window_id"]} "{_clip(window.get("title") or "", 80)}"'
    if window.get("app_name"):
        line += f' ({window["app_name"]})'
    if window.get("minimized"):
        line += " minimized"
    return line


def describe_windows(listing: dict[str, Any]) -> str:
    lines = [_describe_window(w) for w in listing["windows"]]
    if any(w.get("minimized") for w in listing["windows"]):
        lines.append("(a minimized window cannot be operated: ask the user to restore it)")
    if listing.get("truncated"):
        lines.append("(more windows exist than are listed)")
    return "windows:\n" + "\n".join(lines) if lines else "windows: none"


def _observation_notes(observation: dict[str, Any]) -> list[str]:
    notes = []
    if observation.get("degraded"):
        notes.append("(degraded tree: unreliable; observe again before acting)")
    if observation.get("truncated") or observation.get("elements_complete") is False:
        notes.append("(list may be incomplete: a control not shown can still exist)")
    return notes


def _has_keypad(elements: list[dict[str, Any]]) -> bool:
    """True when the window has a key for each of eight or more different digits."""
    digits = {d for e in elements if e.get("enabled") is not False
              for form in _forms(e.get("label")) for d in form if form.isdecimal() and len(form) == 1}
    return len(digits) >= 8


def describe_observation(observation: dict[str, Any]) -> str:
    head = (f'window pid={observation["pid"]} window_id={observation["window_id"]} '
            f'"{_clip(observation.get("title") or "", 80)}"')
    lines = [_describe_control(e) for e in observation["elements"]]
    notes = _observation_notes(observation)
    if _has_keypad(observation["elements"]):
        # Where the model decides what to click, so it is read rather than skimmed in a prompt.
        notes.append('(keypad: desktop_click digits="2380" presses those digit keys in order; '
                     "operators and other keys by label, one call each)")
    return "\n".join([head, *lines, *notes])


def _control_state(element: dict[str, Any]) -> tuple:
    return element.get("role"), element.get("label"), element.get("value"), element.get("enabled")


def describe_change(before: tuple[dict[str, Any], ...], after: dict[str, Any]) -> str:
    """What an action changed. The whole window again when too much moved to list sensibly."""
    elements = after["elements"]
    old = {e["element_index"]: _control_state(e) for e in before}
    changed = [e for e in elements if old.get(e["element_index"]) != _control_state(e)]
    if len(elements) != len(before) or len(changed) > len(elements) // 2:
        return "the window changed; its controls now:\n" + describe_observation(after)
    if not changed:
        return (f"the {len(elements)} listed controls did not change; the action may still have worked "
                "if its effect is not shown in them")
    lines = ["changed:", *(_describe_control(e) for e in changed),
             f"({len(elements) - len(changed)} other controls unchanged)", *_observation_notes(after)]
    return "\n".join(lines)


_VERBS = {"click": "clicked", "type_text": "typed into", "press_key": "pressed a key in"}


def describe_action(action: str, result: dict[str, Any]) -> str:
    target = result["target"]
    done = (f'pressed the keys {" ".join(result["pressed"])}' if "pressed" in result
            else f'{_VERBS.get(action, action)} {_describe_control(target)}')
    if "after" not in result:
        return (f'{done}\ncould not re-read the window ({result.get("observation_error")}); '
                "the outcome is unverified. Observe the window again before acting.")
    return (f'{done}\n{describe_change(result["before"], result["after"])}\n'
            "Dispatch alone is not proof of success: check the requested outcome above.")


class CuaDesktop:
    def __init__(self, config: DesktopConfig, driver: Any = None) -> None:
        self.config = config
        self._driver = driver
        self._lock = asyncio.Lock()
        self._snapshots: dict[tuple[int, int], _Snapshot] = {}
        self._known_windows: set[tuple[int, int]] = set()
        self._closed = False
        self._uncertain = False

    async def startup(self) -> None:
        if self._driver is not None:
            return
        os.environ["CUA_DRIVER_RS_TELEMETRY_ENABLED"] = str(self.config.telemetry).lower()
        os.environ["CUA_DRIVER_RS_UPDATE_CHECK"] = str(self.config.update_check).lower()
        try:
            from cua_driver import CuaDriver
        except ImportError as exc:
            raise RuntimeError("Cua is selected but missing. Install: pip install -e .[desktop]") from exc
        self._driver = CuaDriver.create()
        try:
            async with asyncio.timeout(self.config.timeout):
                metadata = await self._driver.metadata()
            if metadata.driver_version != "0.28.2":
                raise RuntimeError("Jarvis requires cua-driver==0.28.2; reinstall the desktop extra.")
            logger.info("desktop ready backend=cua version=%s delivery=%s",
                        metadata.driver_version, self.config.delivery_mode)
        except BaseException:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        self._closed = True
        self._snapshots.clear()
        self._known_windows.clear()
        if self._driver is not None:
            driver, self._driver = self._driver, None
            async with asyncio.timeout(self.config.timeout):
                await driver.shutdown()

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._closed or self._driver is None:
            raise RuntimeError("Desktop backend is not running")
        if self._uncertain:
            raise RuntimeError("A desktop call was interrupted; restart Jarvis before more desktop actions")
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.config.timeout):
                result = await self._driver.call_tool(name, json.dumps(args))
        except (TimeoutError, asyncio.CancelledError):
            # Native work may already have been dispatched. Never replay it or
            # allow another call to race it after cancellation.
            self._uncertain = True
            self._snapshots.clear()
            logger.error("desktop call interrupted name=%s after=%.1fs; blocking further desktop "
                         "actions until restart", name, time.monotonic() - started)
            raise
        logger.debug("desktop call name=%s ms=%.0f error=%s", name,
                     (time.monotonic() - started) * 1000, result.is_error)
        if result.is_error:
            logger.warning("desktop call refused name=%s code=%s", name, result.error_code or "")
            raise RuntimeError(f"Cua refused {name}: {result.error_code or ''} {result.text}")
        if result.structured_json:
            data = json.loads(result.structured_json)
            return data if isinstance(data, dict) else {"items": data}
        return {"text": result.text}

    async def windows(self) -> dict[str, Any]:
        async with self._lock:
            result = await self._call("list_windows", {})
            windows = result.get("windows", [])
            self._known_windows = {
                (w["pid"], w["window_id"]) for w in windows if w.get("pid")
            }
            self._snapshots.clear()
            return {"windows": [
                {k: w.get(k) for k in ("pid", "window_id", "app_name", "title", "minimized")}
                for w in windows[:60]
            ], "truncated": len(windows) > 60}

    async def launch(self, name: str) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("An application name is required")
        async with self._lock:
            self._snapshots.clear()
            result = await self._call("launch_app", {"name": name})
            # Always rediscover after launch. A launcher PID may differ from
            # the process that owns the actual app window.
            return {"launch": result, "next": "Call desktop_windows to select the exact window."}

    async def observe(self, pid: int, window_id: int) -> dict[str, Any]:
        async with self._lock:
            return await self._observe(pid, window_id)

    async def _observe(self, pid: int, window_id: int) -> dict[str, Any]:
        target = (pid, window_id)
        self._snapshots.pop(target, None)
        if target not in self._known_windows:
            raise ValueError("Select pid and window_id from desktop_windows first")
        state = await self._call("get_window_state", {
            "pid": pid, "window_id": window_id,
            "include_screenshot": False, "include_accessibility_tree": True,
            "max_elements": self.config.max_elements, "max_depth": self.config.max_depth,
        })
        if state.get("pid") != pid or state.get("window_id") != window_id:
            raise RuntimeError("Cua returned a different window; no action is permitted")
        elements = state.get("elements") or []
        tokens = {
            e["element_index"]: e["element_token"] for e in elements
            if e.get("element_token") and e.get("enabled") is not False
        }
        snapshot_id = state.get("snapshot_id")
        shown = [{k: e.get(k) for k in
                  ("element_index", "role", "label", "value", "enabled", "actions")}
                 for e in elements]
        if snapshot_id and not state.get("degraded"):
            self._snapshots[target] = _Snapshot(time.monotonic(), snapshot_id, tokens, tuple(shown))
        return {
            "pid": pid, "window_id": window_id, "snapshot_id": snapshot_id,
            "title": state.get("window_title"), "degraded": state.get("degraded", False),
            "truncated": state.get("truncated", False),
            "elements_complete": state.get("elements_complete"),
            "elements": shown,
        }

    async def act(
        self, action: str, pid: int, window_id: int, snapshot_id: str,
        element_index: int, **extra: Any,
    ) -> dict[str, Any]:
        async with self._lock:
            return await self._act(action, pid, window_id, snapshot_id, element_index, **extra)

    async def act_on(
        self, action: str, pid: int, window_id: int, label: str | None = None,
        index: int | None = None, digits: str | None = None, **extra: Any,
    ) -> dict[str, Any]:
        """``act`` on the control a label names in the window's latest observation.

        The model names what it sees instead of copying snapshot ids and indexes between
        calls. The same checks still hold: the observation must be fresh and the control
        enabled and unique, and the driver refuses a token from any other snapshot.
        ``digits`` (clicks only) presses a number's digit keys in order instead.
        """
        keys = _digit_keys(digits)
        if keys and (label or index is not None):
            raise ValueError("Give either digits or a label, not both")
        if keys and action != "click":
            raise ValueError("digits can only be pressed with desktop_click")
        async with self._lock:
            snapshot = self._fresh_snapshot(pid, window_id)
            if keys:
                return await self._press_keys(pid, window_id, snapshot, keys)
            element = _find_control(snapshot, label, index)
            result = await self._act(action, pid, window_id, snapshot.snapshot_id,
                                     element["element_index"], **extra)
            return {**result, "target": element, "before": snapshot.elements}

    def _fresh_snapshot(self, pid: int, window_id: int) -> _Snapshot:
        snapshot = self._snapshots.get((pid, window_id))
        if snapshot is None or time.monotonic() - snapshot.at > self.config.snapshot_ttl:
            raise ValueError(_STALE)
        return snapshot

    async def _press_keys(
        self, pid: int, window_id: int, first: _Snapshot, digits: list[str],
    ) -> dict[str, Any]:
        """Click a number's digit keys in order, each against the observation the last one left.

        A small model copies "144" faithfully but drops or doubles digits when it spells the
        number out as separate calls. Each key is still resolved and dispatched one at a time;
        the first one that cannot be found stops the sequence and the error says how far it got.
        It takes an explicit ``digits`` argument, never a label: a model that passes an element
        index where a label belongs must get an error, not a handful of key presses.
        """
        snapshot, result, element = first, {}, {}
        for pressed, digit in enumerate(digits, start=0):
            try:
                element = _find_control(snapshot, digit, None)
            except ValueError as exc:
                raise ValueError(f'Pressed {"".join(digits[:pressed]) or "nothing"} of '
                                 f'{"".join(digits)}, then: {exc}') from exc
            result = await self._act("click", pid, window_id, snapshot.snapshot_id,
                                     element["element_index"])
            if "after" not in result:
                break  # the window could not be re-read: stop rather than press blind
            if pressed + 1 < len(digits):
                try:
                    snapshot = self._fresh_snapshot(pid, window_id)
                except ValueError as exc:
                    raise ValueError(f'Pressed {"".join(digits[:pressed + 1])} of '
                                     f'{"".join(digits)}, then: {exc}') from exc
        return {**result, "target": element, "before": first.elements,
                "pressed": digits[:pressed + 1]}

    async def _act(
        self, action: str, pid: int, window_id: int, snapshot_id: str,
        element_index: int, **extra: Any,
    ) -> dict[str, Any]:
        target = (pid, window_id)
        observed = self._snapshots.pop(target, None)
        if (observed is None or observed.snapshot_id != snapshot_id
                or time.monotonic() - observed.at > self.config.snapshot_ttl):
            raise ValueError(_STALE)
        token = observed.tokens.get(element_index)
        if token is None:
            raise ValueError("Element was not enabled and present in the latest observation")
        if action not in {"click", "type_text", "press_key"}:
            raise ValueError("Unsupported desktop action")
        result = await self._call(action, {
            "pid": pid, "window_id": window_id, "snapshot_id": snapshot_id,
            "element_token": token, "delivery_mode": self.config.delivery_mode, **extra,
        })
        # Never turn a dispatch acknowledgement into a completion claim.
        # Return fresh evidence, and let the caller check the postcondition.
        try:
            after = await self._observe(pid, window_id)
        except Exception as exc:
            return {"action": result, "verified": False,
                    "observation_error": str(exc), "next": "Rediscover windows; do not repeat blindly."}
        return {"action": result, "after": after,
                "next": "Check the requested outcome in after; dispatch alone is not success."}


def register_desktop_tools(registry: ToolRegistry, desktop: CuaDesktop) -> None:
    """Expose six compact tools instead of the driver's full catalog.

    Results are short text, not JSON: a small local model does better with the window as
    ``index Role "label"`` lines, and a typical action costs tens of tokens instead of the
    ~900 a full JSON tree took. Actions name a control by its label rather than a
    snapshot id and index, which the model had to copy from one result to the next.
    """
    target = {"pid": {"type": "integer"}, "window_id": {"type": "integer"}}
    control = {
        "label": {"type": "string", "description": "The control's label exactly as desktop_observe "
                  'lists it (a digit key may be given as "7" or "Seven")'},
        "index": {"type": "integer", "description": "Only when two controls share a label: the "
                  "number in front of the one you mean"},
    }
    digits = {"digits": {"type": "string", "description": "Instead of a label, on a keypad: the "
                         'digits to press one key each, in order, such as "2380"'}}

    def schema(props: dict, required: tuple[str, ...] | None = None) -> dict:
        return {"type": "object", "properties": props,
                "required": list(props if required is None else required),
                "additionalProperties": False}

    @registry.tool("desktop_windows", "List app windows; use returned pid/window_id only.", schema({}), repeatable=True)
    async def windows() -> str:
        return describe_windows(await desktop.windows())

    @registry.tool("desktop_launch", "Launch a desktop app by name; returns its windows (pid and window_id).",
                   schema({"name": {"type": "string"}}), requires_confirmation=True)
    async def launch(name: str) -> str:
        await desktop.launch(name)
        # Always rediscover after launch. A launcher PID may differ from the
        # process that owns the actual app window.
        return f"launched {name}\n{describe_windows(await desktop.windows())}"

    @registry.tool("desktop_observe", "List the controls of an exact window: index, role, label and value.",
                   schema(target), repeatable=True)
    async def observe(pid: int, window_id: int) -> str:
        return describe_observation(await desktop.observe(pid, window_id))

    # Pressing the same control twice (77, Tab Tab) is normal on a desktop, and every call
    # is bounded by the step limit, so the repeated-call guard does not apply to actions.
    @registry.tool("desktop_click", "Click a control of an observed window by its label. Returns what changed.",
                   schema({**target, **control, **digits}, ("pid", "window_id")),
                   requires_confirmation=True, repeatable=True)
    async def click(pid: int, window_id: int, label: str | None = None, index: int | None = None,
                    digits: str | None = None) -> str:
        return describe_action("click", await desktop.act_on("click", pid, window_id, label, index, digits))

    @registry.tool("desktop_type", "Set the text of an editable control, found by its label; may replace its existing value.",
                   schema({**target, **control, "text": {"type": "string"}}, ("pid", "window_id", "text")),
                   requires_confirmation=True, repeatable=True)
    async def type_text(pid: int, window_id: int, text: str, label: str | None = None,
                        index: int | None = None) -> str:
        return describe_action("type_text", await desktop.act_on("type_text", pid, window_id, label, index,
                                                                 text=text))

    @registry.tool("desktop_key", "Press a key (return, tab, escape, etc.) in a control found by its label.",
                   schema({**target, **control, "key": {"type": "string"}}, ("pid", "window_id", "key")),
                   requires_confirmation=True, repeatable=True)
    async def press_key(pid: int, window_id: int, key: str, label: str | None = None,
                        index: int | None = None) -> str:
        return describe_action("press_key", await desktop.act_on("press_key", pid, window_id, label, index,
                                                                 key=key))
