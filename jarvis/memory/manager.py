"""
JARVIS Memory Manager
======================
Mem0.ai wrapper for multi-level memory with local Ollama.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.utils.config import expand_path
from jarvis.utils.logger import get_logger

if TYPE_CHECKING:
    from jarvis.utils.config import LLMConfig, MemoryConfig

logger = get_logger(__name__)

# Importing mem0 and opening its vector store can take a long time (heavy imports, and the
# Ollama embedder pulls its model on first use). Past this limit, run without memory.
STARTUP_TIMEOUT = 60.0


def _installed(name: str) -> bool:
    """Whether ``name`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # already imported, but without a __spec__
        return True
    except ImportError:
        return False


def _create_backend(config: dict[str, Any]) -> Any:
    """Import mem0 and build its Memory. Blocking; runs in a worker thread."""
    from mem0 import Memory

    return Memory.from_config(config)


def _entries(raw: Any) -> list[dict[str, Any]]:
    """The memory entries in a mem0 ``search`` or ``get_all`` result.

    mem0 >= 1.0 returns ``{"results": [...]}`` (plus ``"relations"`` with a graph store);
    older releases returned the bare list.
    """
    if isinstance(raw, dict):
        raw = raw.get("results")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning("memory unexpected result type=%s", type(raw).__name__)
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


class MemoryManager:
    """
    Manages JARVIS memory using Mem0.ai.

    Memory levels:
    - User memory: long-term preferences, patterns, facts
    - Session memory: current conversation context
    - Agent memory: task state, planning context
    """

    def __init__(self, memory_config: MemoryConfig, llm_config: LLMConfig) -> None:
        self._config = memory_config
        self._llm_config = llm_config
        self._memory: Any = None  # mem0.Memory instance
        self._user_id = "default"
        self._session_id: str | None = None
        self._stopped = False

    def _disable(self) -> None:
        self._config.enabled = False
        self._memory = None

    def _backend(self) -> Any:
        """The live mem0 instance, or None when memory is disabled, failed, or stopped."""
        return self._memory if self._config.enabled else None

    async def startup(self) -> None:
        """Initialize Mem0 with local Ollama configuration.

        Never raises and never blocks longer than ``STARTUP_TIMEOUT``: if mem0 cannot be
        loaded, memory is disabled and Jarvis carries on without it.
        """
        if not self._config.enabled:
            logger.info("memory disabled by config")
            return

        # Mem0 phones home anonymized telemetry by default; disable it to keep the
        # "nothing leaves this machine" guarantee. Must be set before mem0 is imported.
        os.environ.setdefault("MEM0_TELEMETRY", "false")

        # mem0's Ollama embedder asks "Install it now? [y/N]" on stdin and exits when the
        # package is missing. Check first so a voice session never blocks on that prompt.
        if not _installed("ollama"):
            logger.warning("memory disabled reason=ollama_not_installed (pip install ollama)")
            self._disable()
            return

        vector_path = str(expand_path(self._config.vector_store.path))
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self._llm_config.model,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "ollama_base_url": self._llm_config.base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest",
                    "ollama_base_url": self._llm_config.base_url,
                },
            },
            "vector_store": {
                "provider": self._config.vector_store.provider,
                "config": {
                    "collection_name": self._config.vector_store.collection_name,
                    "path": vector_path,
                },
            },
        }

        try:
            self._memory = await asyncio.wait_for(
                asyncio.to_thread(_create_backend, config), STARTUP_TIMEOUT)
        except TimeoutError:
            # The worker cannot be interrupted; its result, if any, is discarded.
            logger.warning("memory disabled reason=startup_timeout limit=%.0fs", STARTUP_TIMEOUT)
            self._disable()
        except ImportError as e:
            logger.warning("memory disabled reason=mem0_not_installed error=%s", e)
            self._disable()
        except SystemExit as e:
            # mem0 calls sys.exit() when an optional package is missing; that must not
            # take the whole assistant down.
            logger.warning("memory disabled reason=mem0_exited code=%s", e.code)
            self._disable()
        except Exception as e:
            logger.error("memory init failed error=%s", e)
            self._disable()
        else:
            self._session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            logger.info("ready backend=mem0 vectors=%s", vector_path)

    async def shutdown(self) -> None:
        """Clean up memory resources; later reads and writes do nothing."""
        self._memory = None
        if not self._stopped:
            self._stopped = True
            logger.info("memory stopped")

    async def add_conversation(
        self,
        user_input: str,
        assistant_response: str,
        user_id: str | None = None,
    ) -> None:
        """
        Process a conversation exchange — Mem0 auto-extracts memories.
        Runs in a thread pool to avoid blocking the asyncio event loop.
        """
        backend = self._backend()
        if backend is None:
            return

        uid = user_id or self._user_id
        conversation = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]
        metadata = {"timestamp": datetime.now().isoformat(), "session_id": self._session_id}

        try:
            await asyncio.to_thread(backend.add, conversation, user_id=uid, metadata=metadata)
            logger.debug("memory stored user=%s", uid)
        except Exception as e:
            logger.warning("memory write failed error=%s", e)

    async def get_context(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Retrieve relevant context for a query.

        Returns list of memory strings ranked by relevance.
        Runs in a thread pool to avoid blocking the asyncio event loop.
        """
        backend = self._backend()
        if backend is None:
            return []

        uid = user_id or self._user_id
        try:
            raw = await asyncio.to_thread(backend.search, query=query, user_id=uid, limit=limit)
            memories = [str(entry["memory"]) for entry in _entries(raw) if entry.get("memory")]
        except Exception as e:
            logger.warning("memory search failed error=%s", e)
            return []
        logger.debug("memory search hits=%d", len(memories))
        return memories

    async def get_all_memories(
        self, user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories for a user."""
        backend = self._backend()
        if backend is None:
            return []

        uid = user_id or self._user_id
        try:
            return _entries(await asyncio.to_thread(backend.get_all, user_id=uid))
        except Exception as e:
            logger.warning("memory list failed error=%s", e)
            return []
