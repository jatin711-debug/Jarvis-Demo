"""
JARVIS LLM Client
==================
Async Ollama API client with native tool/function calling.
"""

from __future__ import annotations

from typing import Any

import httpx

from jarvis.utils.logger import get_logger
from jarvis.utils.config import LLMConfig

logger = get_logger(__name__)

# Ollama API endpoints
_API_CHAT = "/api/chat"
_API_GENERATE = "/api/generate"
_API_TAGS = "/api/tags"


class LLMClient:
    """
    Async client for Ollama's REST API.

    Supports:
    - Chat completions with message history
    - Tool/function calling (native Ollama support)
    - Health checks
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        """Initialize the HTTP client and verify Ollama connection."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        # Verify Ollama is running
        if await self.health_check():
            logger.info("✅ Ollama connected at %s", self.base_url)
        else:
            logger.error("❌ Cannot reach Ollama at %s", self.base_url)

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            assert self._client is not None
            resp = await self._client.get(_API_TAGS)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            logger.debug("Available models: %s", model_names)
            return True
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to Ollama.

        Args:
            messages: Conversation history [{role, content}, ...]
            tools: Optional tool definitions for function calling
            temperature: Override config temperature
            max_tokens: Override config max_tokens

        Returns:
            Ollama chat response dict with 'message' key
        """
        assert self._client is not None, "Call startup() first"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
                "num_ctx": self.config.context_window,
            },
        }

        if tools:
            payload["tools"] = tools

        logger.debug("LLM request: %d messages, tools=%s",
                      len(messages), bool(tools))

        resp = await self._client.post(_API_CHAT, json=payload)
        resp.raise_for_status()
        result = resp.json()

        logger.debug("LLM response: %d chars",
                      len(result.get("message", {}).get("content", "")))
        return result

    async def generate(self, prompt: str) -> str:
        """Simple single-shot generation (no chat history)."""
        assert self._client is not None, "Call startup() first"

        resp = await self._client.post(_API_GENERATE, json={
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        })
        resp.raise_for_status()
        return resp.json().get("response", "")
