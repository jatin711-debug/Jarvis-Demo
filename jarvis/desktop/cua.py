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
from typing import Any

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.config import DesktopConfig


class CuaDesktop:
    def __init__(self, config: DesktopConfig, driver: Any = None) -> None:
        self.config = config
        self._driver = driver
        self._lock = asyncio.Lock()
        self._snapshots: dict[tuple[int, int], tuple[float, str, dict[int, str]]] = {}
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
        try:
            async with asyncio.timeout(self.config.timeout):
                result = await self._driver.call_tool(name, json.dumps(args))
        except (TimeoutError, asyncio.CancelledError):
            # Native work may already have been dispatched. Never replay it or
            # allow another call to race it after cancellation.
            self._uncertain = True
            self._snapshots.clear()
            raise
        if result.is_error:
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
        if snapshot_id and not state.get("degraded"):
            self._snapshots[target] = (time.monotonic(), snapshot_id, tokens)
        return {
            "pid": pid, "window_id": window_id, "snapshot_id": snapshot_id,
            "title": state.get("window_title"), "degraded": state.get("degraded", False),
            "truncated": state.get("truncated", False),
            "elements_complete": state.get("elements_complete"),
            "elements": [{k: e.get(k) for k in
                          ("element_index", "role", "label", "value", "enabled", "actions")}
                         for e in elements],
        }

    async def act(
        self, action: str, pid: int, window_id: int, snapshot_id: str,
        element_index: int, **extra: Any,
    ) -> dict[str, Any]:
        async with self._lock:
            target = (pid, window_id)
            observed = self._snapshots.pop(target, None)
            if (observed is None or observed[1] != snapshot_id
                    or time.monotonic() - observed[0] > self.config.snapshot_ttl):
                raise ValueError("Missing or stale observation; call desktop_observe again")
            token = observed[2].get(element_index)
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
    """Expose six compact tools instead of the driver's full catalog."""
    target = {"pid": {"type": "integer"}, "window_id": {"type": "integer"}}
    action = {**target, "snapshot_id": {"type": "string"}, "element_index": {"type": "integer"}}

    def schema(props: dict) -> dict:
        return {"type": "object", "properties": props, "required": list(props),
                "additionalProperties": False}

    def tool_json(data: dict) -> str:
        # Preserve the full accessibility tree within the model's bounded
        # result context. Null fields and JSON whitespace convey no evidence.
        def compact(value):
            if isinstance(value, dict):
                return {key: compact(item) for key, item in value.items()
                        if item is not None and not (key == "actions" and item == ["invoke"])}
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value

        return json.dumps(compact(data), ensure_ascii=False, separators=(",", ":"))

    @registry.tool("desktop_windows", "List app windows; use returned pid/window_id only.", schema({}), repeatable=True)
    async def windows() -> str:
        return tool_json(await desktop.windows())

    @registry.tool("desktop_launch", "Launch a desktop app by name, then list windows.",
                   schema({"name": {"type": "string"}}), requires_confirmation=True)
    async def launch(name: str) -> str:
        return tool_json(await desktop.launch(name))

    @registry.tool("desktop_observe", "Read current controls and values of an exact window.", schema(target), repeatable=True)
    async def observe(pid: int, window_id: int) -> str:
        return tool_json(await desktop.observe(pid, window_id))

    @registry.tool("desktop_click", "Click an observed element. Returns fresh state; verify the outcome.", schema(action), requires_confirmation=True)
    async def click(pid: int, window_id: int, snapshot_id: str, element_index: int) -> str:
        return tool_json(await desktop.act("click", pid, window_id, snapshot_id, element_index))

    @registry.tool("desktop_type", "Set the text of an observed editable element; may replace its existing value.",
                   schema({**action, "text": {"type": "string"}}), requires_confirmation=True)
    async def type_text(pid: int, window_id: int, snapshot_id: str, element_index: int, text: str) -> str:
        return tool_json(await desktop.act("type_text", pid, window_id, snapshot_id, element_index, text=text))

    @registry.tool("desktop_key", "Press a key (return, tab, escape, etc.) in an observed control.",
                   schema({**action, "key": {"type": "string"}}), requires_confirmation=True)
    async def press_key(pid: int, window_id: int, snapshot_id: str, element_index: int, key: str) -> str:
        return tool_json(await desktop.act("press_key", pid, window_id, snapshot_id, element_index, key=key))
