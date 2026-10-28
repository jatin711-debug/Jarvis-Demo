"""Listen for a user's request to cancel an in-flight browser search."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from jarvis.agent.tool_registry import CANCELLED_MESSAGE
from jarvis.utils.logger import get_logger
from jarvis.voice.transcript import is_explicit_yes

logger = get_logger(__name__)
_T = TypeVar("_T")


async def monitor_search(
    operation: Awaitable[_T],
    *,
    input_device: int | None,
    speak: Callable[[str], Awaitable[None]],
    record_confirmation: Callable[[], Awaitable[str | None]],
    note_fault: Callable[[str, Exception], None],
) -> _T | str:
    """Wait for search or speech, cancelling and joining the search when confirmed.

    The first utterance triggers a fresh yes/no prompt; it is not dispatched as
    another action. A search that finishes while confirmation is pending is still
    discarded if the user confirms cancellation.
    """
    from jarvis.voice.implementations import wait_for_speech

    task = asyncio.ensure_future(operation)
    monitor = None
    stop = asyncio.Event()
    try:
        while not task.done():
            monitor = asyncio.create_task(wait_for_speech(input_device, cancel_event=stop))
            done, _ = await asyncio.wait({task, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                break
            try:
                detected = monitor.result()
            except Exception as exc:
                note_fault("search interruption monitor", exc)
                break
            monitor = None
            if not detected:
                break
            await speak("Would you like me to cancel the search? Say yes or no.")
            heard = await record_confirmation()
            if heard and is_explicit_yes(heard):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                logger.info("search cancelled by voice; results discarded")
                return CANCELLED_MESSAGE
        return await task
    finally:
        stop.set()
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
