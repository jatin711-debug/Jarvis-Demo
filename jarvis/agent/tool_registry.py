"""
JARVIS Tool Registry
=====================
Register and manage callable tools for the agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema for params
    handler: Callable[..., Coroutine[Any, Any, Any]]
    requires_confirmation: bool = False
    tags: list[str] = field(default_factory=list)

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

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        requires_confirmation: bool = False,
        tags: list[str] | None = None,
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
            )
            self._tools[name] = tool_def
            logger.debug("Registered tool: %s", name)
            return func
        return decorator

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool directly (non-decorator style)."""
        self._tools[tool_def.name] = tool_def
        logger.debug("Registered tool: %s", tool_def.name)

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        logger.info("🔧 Executing tool: %s(%s)", name, kwargs)
        try:
            result = await tool.handler(**kwargs)
            logger.info("✅ Tool '%s' completed", name)
            return result
        except Exception as e:
            logger.error("❌ Tool '%s' failed: %s", name, e)
            raise

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Ollama's tool calling format."""
        return [t.to_ollama_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
