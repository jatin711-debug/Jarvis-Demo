import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from jarvis.core.orchestrator import Orchestrator
from jarvis.utils.config import JarvisConfig


async def test_memory_auto_save_is_respected():
    config = JarvisConfig()
    config.memory.auto_save = False
    app = Orchestrator(config)
    app.memory = SimpleNamespace(add_conversation=AsyncMock())
    app._save_conversation("hello", "hi")
    assert not app._memory_tasks
    app.memory.add_conversation.assert_not_called()


async def test_memory_backlog_is_bounded_and_drained(monkeypatch):
    monkeypatch.setattr("jarvis.core.orchestrator.shutdown_browser", AsyncMock())
    config = JarvisConfig()
    config.memory.pending_writes = 1
    app = Orchestrator(config)
    gate = asyncio.Event()

    async def save(*args):
        await gate.wait()

    app.memory = SimpleNamespace(add_conversation=AsyncMock(side_effect=save), shutdown=AsyncMock())
    app._save_conversation("one", "one")
    app._save_conversation("two", "two")
    assert len(app._memory_tasks) == 1
    gate.set()
    await app.shutdown()
    app.memory.add_conversation.assert_awaited_once_with("one", "one")
    app.memory.shutdown.assert_awaited_once()
    assert not app._memory_tasks


async def test_failed_service_does_not_prevent_remaining_cleanup(monkeypatch):
    monkeypatch.setattr("jarvis.core.orchestrator.shutdown_browser", AsyncMock(side_effect=RuntimeError))
    app = Orchestrator(JarvisConfig())
    app.desktop = SimpleNamespace(shutdown=AsyncMock(side_effect=RuntimeError))
    app.llm = SimpleNamespace(shutdown=AsyncMock())
    await app.shutdown()
    app.desktop.shutdown.assert_awaited_once()
    app.llm.shutdown.assert_awaited_once()
