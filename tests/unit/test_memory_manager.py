"""Memory manager: bounded startup, both mem0 result shapes, and failures that never raise."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import types

import pytest

from jarvis.memory import manager as memory_module
from jarvis.memory.manager import MemoryManager
from jarvis.utils.config import LLMConfig, MemoryConfig, VectorStoreConfig

ENTRIES = [{"id": "1", "memory": "likes tea"}, {"id": "2", "memory": "lives in Toronto"}]


class FakeBackend:
    """A mem0.Memory stand-in answering in the 1.x ({"results": [...]}) or the old list shape."""

    def __init__(self, shape: str = "dict") -> None:
        self.shape = shape
        self.entries = list(ENTRIES)
        self.calls: list[tuple] = []
        self.errors: dict[str, Exception] = {}

    def _answer(self, name: str, entries):
        if name in self.errors:
            raise self.errors[name]
        return {"results": entries} if self.shape == "dict" else entries

    def add(self, messages, user_id=None, metadata=None):
        self.calls.append(("add", messages, user_id, metadata))
        if "add" in self.errors:
            raise self.errors["add"]
        return {"results": []}

    def search(self, query, user_id=None, limit=5):
        self.calls.append(("search", query, user_id, limit))
        return self._answer("search", self.entries[:limit])

    def get_all(self, user_id=None):
        self.calls.append(("get_all", user_id))
        return self._answer("get_all", self.entries)


def install_mem0(monkeypatch, from_config):
    module = types.ModuleType("mem0")
    module.Memory = types.SimpleNamespace(from_config=from_config)
    monkeypatch.setitem(sys.modules, "mem0", module)
    return module


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    """CI has no ollama package; the manager only checks that it can be found."""
    monkeypatch.setitem(sys.modules, "ollama", types.ModuleType("ollama"))


@pytest.fixture(autouse=True)
def log_memory(caplog):
    caplog.set_level(logging.INFO, logger="jarvis.memory.manager")


def make_manager(tmp_path, **memory) -> MemoryManager:
    config = MemoryConfig(vector_store=VectorStoreConfig(path=str(tmp_path / "vectors")), **memory)
    return MemoryManager(config, LLMConfig())


async def started(monkeypatch, tmp_path, backend: FakeBackend | None = None) -> MemoryManager:
    backend = backend or FakeBackend()
    install_mem0(monkeypatch, lambda config: backend)
    manager = make_manager(tmp_path)
    await manager.startup()
    assert manager._config.enabled and manager._memory is backend
    return manager


# --- startup ----------------------------------------------------------------

async def test_startup_builds_mem0_from_the_ollama_config(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MEM0_TELEMETRY", "unset-by-test")
    monkeypatch.delenv("MEM0_TELEMETRY")
    seen = {}

    def from_config(config):
        seen["config"] = config
        seen["telemetry"] = os.environ.get("MEM0_TELEMETRY")
        return FakeBackend()

    install_mem0(monkeypatch, from_config)
    manager = make_manager(tmp_path)
    await manager.startup()

    config = seen["config"]
    assert config["llm"]["provider"] == "ollama"
    assert config["embedder"]["config"]["model"] == "nomic-embed-text:latest"
    assert config["vector_store"]["config"]["path"].endswith("vectors")
    assert seen["telemetry"] == "false"  # set before mem0 could phone home
    assert manager._config.enabled and manager._session_id.startswith("session_")
    assert "ready backend=mem0" in caplog.text


async def test_mem0_is_imported_and_built_off_the_event_loop(monkeypatch, tmp_path):
    loop_thread = threading.current_thread()
    threads: dict[str, threading.Thread] = {}

    def from_config(config):
        threads["build"] = threading.current_thread()
        return FakeBackend()

    class SpyModule(types.ModuleType):
        def __getattr__(self, name):  # ``from mem0 import Memory`` resolves the name here
            threads["import"] = threading.current_thread()
            if name == "Memory":
                return types.SimpleNamespace(from_config=from_config)
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "mem0", SpyModule("mem0"))
    manager = make_manager(tmp_path)
    await manager.startup()

    assert manager._memory is not None
    assert threads["import"] is not loop_thread
    assert threads["build"] is not loop_thread


async def test_disabled_by_config_never_loads_mem0(monkeypatch, tmp_path):
    install_mem0(monkeypatch, lambda config: pytest.fail("mem0 must not be built"))
    manager = make_manager(tmp_path, enabled=False)
    await manager.startup()
    assert await manager.get_context("anything") == []


async def test_missing_mem0_disables_memory(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(sys.modules, "mem0", None)  # makes ``import mem0`` raise ImportError
    manager = make_manager(tmp_path)
    await manager.startup()

    assert manager._config.enabled is False and manager._memory is None
    assert "memory disabled reason=mem0_not_installed" in caplog.text
    await manager.add_conversation("hi", "hello")
    assert await manager.get_context("hi") == []


async def test_missing_ollama_disables_memory_before_mem0_can_prompt(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(sys.modules, "ollama", None)
    install_mem0(monkeypatch, lambda config: pytest.fail("mem0 would ask for input() here"))
    manager = make_manager(tmp_path)
    await manager.startup()

    assert manager._config.enabled is False
    assert "reason=ollama_not_installed" in caplog.text


async def test_system_exit_during_startup_disables_memory(monkeypatch, tmp_path, caplog):
    def from_config(config):
        raise SystemExit(1)  # what mem0 does when an optional package is declined

    install_mem0(monkeypatch, from_config)
    manager = make_manager(tmp_path)
    await manager.startup()  # must not raise

    assert manager._config.enabled is False and manager._memory is None
    assert "memory disabled reason=mem0_exited code=1" in caplog.text


async def test_other_startup_errors_disable_memory_and_say_why(monkeypatch, tmp_path, caplog):
    def from_config(config):
        raise RuntimeError("chroma database is locked")

    install_mem0(monkeypatch, from_config)
    manager = make_manager(tmp_path)
    await manager.startup()

    assert manager._config.enabled is False
    assert "memory init failed error=chroma database is locked" in caplog.text


async def test_slow_startup_is_bounded_and_disables_memory(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(memory_module, "STARTUP_TIMEOUT", 0.05)
    release = threading.Event()

    def from_config(config):
        release.wait(10)
        return FakeBackend()

    install_mem0(monkeypatch, from_config)
    manager = make_manager(tmp_path)
    began = time.monotonic()
    try:
        await manager.startup()
        elapsed = time.monotonic() - began
    finally:
        release.set()  # let the abandoned worker finish

    assert elapsed < 5
    assert manager._config.enabled is False and manager._memory is None
    assert "memory disabled reason=startup_timeout limit=0s" in caplog.text
    assert await manager.get_context("hi") == []


# --- result shapes ----------------------------------------------------------

@pytest.mark.parametrize("shape", ["dict", "list"])
async def test_search_and_get_all_accept_both_mem0_shapes(monkeypatch, tmp_path, shape):
    backend = FakeBackend(shape)
    manager = await started(monkeypatch, tmp_path, backend)

    assert await manager.get_context("tea?", limit=1) == ["likes tea"]
    assert await manager.get_context("tea?") == ["likes tea", "lives in Toronto"]
    assert await manager.get_all_memories() == ENTRIES
    assert ("search", "tea?", "default", 1) in backend.calls
    assert ("get_all", "default") in backend.calls


async def test_graph_results_with_relations_are_unwrapped(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.search = lambda **kwargs: {"results": ENTRIES, "relations": [{"source": "a"}]}
    manager = await started(monkeypatch, tmp_path, backend)
    assert await manager.get_context("q") == ["likes tea", "lives in Toronto"]


async def test_entries_without_a_memory_key_are_skipped(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.entries = [{"id": "1"}, {"id": "2", "memory": ""}, "junk", None,
                       {"id": "3", "memory": "likes tea"}]
    manager = await started(monkeypatch, tmp_path, backend)

    assert await manager.get_context("q") == ["likes tea"]
    assert await manager.get_all_memories() == [
        {"id": "1"}, {"id": "2", "memory": ""}, {"id": "3", "memory": "likes tea"}]


@pytest.mark.parametrize("raw", [None, {}, {"results": None}, {"relations": []}, []])
async def test_empty_or_odd_results_mean_no_memories(monkeypatch, tmp_path, raw):
    backend = FakeBackend()
    backend.search = lambda **kwargs: raw
    backend.get_all = lambda **kwargs: raw
    manager = await started(monkeypatch, tmp_path, backend)

    assert await manager.get_context("q") == []
    assert await manager.get_all_memories() == []


async def test_unrecognised_result_type_is_empty_and_warned(monkeypatch, tmp_path, caplog):
    backend = FakeBackend()
    backend.get_all = lambda **kwargs: "not a result"
    manager = await started(monkeypatch, tmp_path, backend)

    assert await manager.get_all_memories() == []
    assert "memory unexpected result type=str" in caplog.text


# --- writes and failures ----------------------------------------------------

async def test_add_conversation_sends_the_exchange_with_session_metadata(monkeypatch, tmp_path):
    backend = FakeBackend()
    manager = await started(monkeypatch, tmp_path, backend)

    await manager.add_conversation("I like tea", "Noted", user_id="sam")

    ((name, messages, user_id, metadata),) = backend.calls
    assert (name, user_id) == ("add", "sam")
    assert messages == [{"role": "user", "content": "I like tea"},
                        {"role": "assistant", "content": "Noted"}]
    assert metadata["session_id"] == manager._session_id and metadata["timestamp"]


async def test_backend_failures_return_empty_and_log_the_error(monkeypatch, tmp_path, caplog):
    backend = FakeBackend()
    backend.errors = {"add": OSError("disk full"), "search": ValueError("embedder offline"),
                      "get_all": KeyError("collection")}
    manager = await started(monkeypatch, tmp_path, backend)

    assert await manager.add_conversation("hi", "hello") is None
    assert await manager.get_context("hi") == []
    assert await manager.get_all_memories() == []

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "memory write failed error=disk full" in warnings
    assert "memory search failed error=embedder offline" in warnings
    assert any(w.startswith("memory list failed error=") and "collection" in w for w in warnings)
    assert manager._config.enabled  # a failed call does not switch memory off


# --- shutdown ---------------------------------------------------------------

async def test_calls_after_shutdown_do_nothing(monkeypatch, tmp_path, caplog):
    backend = FakeBackend()
    manager = await started(monkeypatch, tmp_path, backend)
    await manager.shutdown()
    await manager.shutdown()  # idempotent

    assert await manager.add_conversation("hi", "hello") is None
    assert await manager.get_context("hi") == []
    assert await manager.get_all_memories() == []
    assert backend.calls == []
    assert [r.getMessage() for r in caplog.records].count("memory stopped") == 1


async def test_shutdown_without_startup_is_safe(tmp_path, caplog):
    manager = make_manager(tmp_path, enabled=False)
    await manager.shutdown()
    assert [r.getMessage() for r in caplog.records] == ["memory stopped"]
