"""
JARVIS Tool Registry
=====================
Register and manage callable tools for the agent.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from jarvis.agent.permissions import PermissionGateway
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

# Results the agent recognizes as "stop this turn" signals.
CANCELLED_MESSAGE = "Action cancelled by the user."
PERMISSION_DENIED_PREFIX = "Permission denied:"

# Callback: (tool_name, args) -> True to proceed, False to cancel
ConfirmCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema for params
    handler: Callable[..., Coroutine[Any, Any, Any]]
    requires_confirmation: bool = False
    tags: list[str] = field(default_factory=list)
    repeatable: bool = False

    def to_ollama_schema(self) -> dict[str, Any]:
        """Convert to Ollama tool calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def validate_arguments(
    tool: ToolDefinition, args: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """
    Validate and coerce tool arguments against the tool's JSON schema.

    - Checks required parameters are present.
    - Coerces int/float/bool from string representations (common LLM output).
    - Unknown parameters are passed through (handlers may accept **kwargs).

    Returns (coerced_args, error_message). error_message is None on success.
    """
    schema = tool.parameters or {}
    props: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    coerced = dict(args)
    missing = [k for k in required if k not in coerced or coerced[k] is None]
    if missing:
        return coerced, f"Missing required parameter(s): {', '.join(missing)} for tool '{tool.name}'"
    if schema.get("additionalProperties") is False:
        unknown = set(coerced) - set(props)
        if unknown:
            return coerced, f"Unknown parameter(s): {', '.join(sorted(unknown))}"

    for key, spec in props.items():
        if key not in coerced:
            continue
        value = coerced[key]
        expected = spec.get("type")
        if value is None:
            continue
        if expected == "integer" and not isinstance(value, int):
            try:
                numeric = float(str(value))
                if not numeric.is_integer():
                    raise ValueError("Fractional integer")
                coerced[key] = int(numeric)
            except (ValueError, TypeError, OverflowError):
                return coerced, f"Parameter '{key}' must be an integer, got {value!r}"
        elif expected == "number" and not isinstance(value, (int, float)):
            try:
                coerced[key] = float(str(value))
            except (ValueError, TypeError):
                return coerced, f"Parameter '{key}' must be a number, got {value!r}"
        elif expected == "boolean" and not isinstance(value, bool):
            low = str(value).strip().lower()
            if low in ("true", "1", "yes"):
                coerced[key] = True
            elif low in ("false", "0", "no"):
                coerced[key] = False
            else:
                return coerced, f"Parameter '{key}' must be a boolean, got {value!r}"
        elif expected == "string" and not isinstance(value, str):
            coerced[key] = str(value)

    missing = [k for k in required if k not in coerced or coerced[k] is None]
    if missing:
        return coerced, (
            f"Missing required parameter(s): {', '.join(missing)} "
            f"for tool '{tool.name}'"
        )
    return coerced, None


class ToolRegistry:
    """
    Registry for agent tools.

    Usage:
        registry = ToolRegistry()

        @registry.tool(
            name="open_app",
            description="Launch an application",
            parameters={...}
        )
        async def open_app(app_name: str) -> str:
            ...
    """

    def __init__(self, confirm_callback: ConfirmCallback | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._confirm_callback = confirm_callback
        self._permission_gateway: PermissionGateway | None = None

    def set_confirm_callback(self, callback: ConfirmCallback | None) -> None:
        """Wire (or clear) the confirmation callback for dangerous tools."""
        self._confirm_callback = callback

    def set_permission_gateway(self, gateway: PermissionGateway | None) -> None:
        """Apply target/effect policy before confirmation and dispatch."""
        self._permission_gateway = gateway

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        requires_confirmation: bool = False,
        tags: list[str] | None = None,
        repeatable: bool = False,
    ) -> Callable:
        """Decorator to register a tool function."""
        def decorator(func: Callable) -> Callable:
            # Ensure the function is async
            if not inspect.iscoroutinefunction(func):
                raise TypeError(f"Tool '{name}' must be an async function")

            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
                requires_confirmation=requires_confirmation,
                tags=tags or [],
                repeatable=repeatable,
            )
            self._tools[name] = tool_def
            logger.debug("tool registered name=%s", name)
            return func
        return decorator

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool directly (non-decorator style)."""
        self._tools[tool_def.name] = tool_def
        logger.debug("tool registered name=%s", tool_def.name)

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Reject invalid calls before asking the user to approve anything.
        kwargs, error = validate_arguments(tool, kwargs)
        if error:
            allowed = list(tool.parameters.get("properties", {}))
            required = tool.parameters.get("required", [])
            logger.warning("tool rejected name=%s reason=invalid_arguments", tool_name)
            return (f"Invalid arguments for {tool_name}: {error}. "
                    f"Allowed parameter names: {allowed}. Required parameter names: {required}. "
                    "Unknown parameters must be removed, not supplied. No action was executed.")

        if self._permission_gateway is not None:
            decision = self._permission_gateway.inspect(
                tool_name, kwargs, tool_requires_confirmation=tool.requires_confirmation)
            if not decision.allowed:
                logger.warning("tool denied name=%s reason=policy", tool_name)
                return f"{PERMISSION_DENIED_PREFIX} {decision.reason}. No action was executed."
            if decision.requires_confirmation:
                if self._confirm_callback is None:
                    return f"{PERMISSION_DENIED_PREFIX} confirmation is unavailable. No action was executed."
                summary = {"effect": decision.effect, "target": decision.target,
                           "action_id": decision.action_id}
                approved = await self._confirm_callback(tool_name, summary)
                if not approved:
                    logger.info("tool cancelled by user name=%s", tool_name)
                    return CANCELLED_MESSAGE
                # Recheck after a slow voice/text approval. A changed scope
                # or symlink target cannot inherit the earlier approval.
                current = self._permission_gateway.inspect(
                    tool_name, kwargs, tool_requires_confirmation=tool.requires_confirmation)
                if (not current.allowed or current.action_id != decision.action_id
                        or current.target != decision.target):
                    return f"{PERMISSION_DENIED_PREFIX} target changed during confirmation. No action was executed."
        elif tool.requires_confirmation and self._confirm_callback is not None:
            approved = await self._confirm_callback(tool_name, kwargs)
            if not approved:
                logger.info("tool cancelled by user name=%s", tool_name)
                return CANCELLED_MESSAGE

        # The agent logs the call and its outcome at INFO; this is registry-level detail.
        logger.debug("tool dispatch name=%s", tool_name)
        try:
            return await tool.handler(**kwargs)
        except Exception as e:
            logger.debug("tool handler raised name=%s error_type=%s", tool_name, type(e).__name__)
            raise

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Ollama's tool calling format."""
        return [t.to_ollama_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
