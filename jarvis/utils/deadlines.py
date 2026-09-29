"""Bound a wait while reporting progress without logging request contents."""
import asyncio

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)


async def wait_with_progress(awaitable, timeout: float, label: str):
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"{label} exceeded {timeout:g}s")
            done, _ = await asyncio.wait({task}, timeout=min(5.0, remaining))
            if done:
                return task.result()
            if loop.time() < deadline:
                logger.info("Still waiting for %s (%.0fs remaining)", label, deadline - loop.time())
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
