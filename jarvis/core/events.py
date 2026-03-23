"""
JARVIS Event Bus
=================
Lightweight in-process pub/sub — replaces Redis Pub/Sub.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """
    Simple async event bus for decoupled communication between services.

    Usage:
        bus = EventBus()
        bus.on("user_spoke", handle_speech)
        await bus.emit("user_spoke", text="hello")
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event."""
        self._handlers[event].append(handler)
        logger.debug("Subscribed %s to event '%s'", handler.__name__, event)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event."""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all subscribed handlers concurrently."""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        logger.debug("Emitting '%s' → %d handler(s)", event, len(handlers))
        tasks = [asyncio.create_task(h(**kwargs)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "Handler error for event '%s': %s", event, result,
                    exc_info=result,
                )
