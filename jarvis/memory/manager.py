"""
JARVIS Memory Manager
======================
Mem0.ai wrapper for multi-level memory with local Ollama.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

from jarvis.utils.logger import get_logger
from jarvis.utils.config import expand_path

if TYPE_CHECKING:
    from jarvis.utils.config import MemoryConfig, LLMConfig

logger = get_logger(__name__)


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

    async def startup(self) -> None:
        """Initialize Mem0 with local Ollama configuration."""
        if not self._config.enabled:
            logger.info("Memory disabled in config")
            return

        try:
            from mem0 import Memory

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

            self._memory = Memory.from_config(config)
            self._session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            logger.info("✅ Mem0 initialized (vectors: %s)", vector_path)

        except ImportError:
            logger.warning("mem0ai not installed — memory disabled")
            self._config.enabled = False
        except Exception as e:
            logger.error("Failed to initialize Mem0: %s", e)
            self._config.enabled = False

    async def shutdown(self) -> None:
        """Clean up memory resources."""
        self._memory = None
        logger.info("Memory manager shut down")

    async def add_conversation(
        self,
        user_input: str,
        assistant_response: str,
        user_id: str | None = None,
    ) -> None:
        """
        Process a conversation exchange — Mem0 auto-extracts memories.
        """
        if not self._config.enabled or not self._memory:
            return

        uid = user_id or self._user_id
        conversation = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]

        try:
            self._memory.add(
                conversation,
                user_id=uid,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self._session_id,
                },
            )
            logger.debug("Memory stored for conversation with user '%s'", uid)
        except Exception as e:
            logger.warning("Failed to store memory: %s", e)

    async def get_context(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Retrieve relevant context for a query.

        Returns list of memory strings ranked by relevance.
        """
        if not self._config.enabled or not self._memory:
            return []

        uid = user_id or self._user_id

        try:
            results = self._memory.search(
                query=query,
                user_id=uid,
                limit=limit,
            )
            memories = [entry["memory"] for entry in results.get("results", [])]
            logger.debug("Retrieved %d memories for query", len(memories))
            return memories
        except Exception as e:
            logger.warning("Memory search failed: %s", e)
            return []

    async def get_all_memories(
        self, user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories for a user."""
        if not self._config.enabled or not self._memory:
            return []

        uid = user_id or self._user_id
        try:
            return self._memory.get_all(user_id=uid)
        except Exception as e:
            logger.warning("Failed to get all memories: %s", e)
            return []
