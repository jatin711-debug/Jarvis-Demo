"""
JARVIS LLM Client
==================
Multi-provider LLM client:
- Ollama: local Ollama API (default)
- HuggingFace: Direct Transformers inference (for LFM, etc.)
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

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
    Async LLM client supporting multiple providers.

    Providers:
    - Ollama: local Ollama API (default)
    - HuggingFace: Direct Transformers inference (LFM, etc.)
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider = config.provider.lower()
        self.base_url = config.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._hf_model = None
        self._hf_tokenizer = None

    async def startup(self) -> None:
        """Initialize the LLM client based on provider."""
        if self.provider == "huggingface":
            await self._startup_huggingface()
        else:
            await self._startup_ollama()

    async def _startup_ollama(self) -> None:
        """Initialize Ollama HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        if await self.health_check():
            logger.info("✅ Ollama connected at %s", self.base_url)
        else:
            logger.error("❌ Cannot reach Ollama at %s", self.base_url)

    async def _startup_huggingface(self) -> None:
        """Initialize HuggingFace Transformers model."""
        logger.info("🔄 Loading HuggingFace model: %s", self.config.model)

        loop = asyncio.get_event_loop()

        def _load_model():
            import warnings
            warnings.simplefilter("ignore")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                device_map="auto",
                torch_dtype="bfloat16",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True,
            )
            return model, tokenizer

        self._hf_model, self._hf_tokenizer = await loop.run_in_executor(
            None, _load_model
        )
        logger.info("✅ HuggingFace model loaded: %s", self.config.model)

    async def shutdown(self) -> None:
        """Close the LLM client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._hf_model = None
        self._hf_tokenizer = None

    async def health_check(self) -> bool:
        """Check if provider is running/available."""
        if self.provider == "huggingface":
            return self._hf_model is not None

        try:
            assert self._client is not None
            resp = await self._client.get(_API_TAGS)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            logger.debug("Available models: %s", model_names)
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply chat template for HuggingFace models like LFM."""
        if not self._hf_tokenizer:
            raise RuntimeError("Tokenizer not loaded")

        # Build conversation for LFM's chat format
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append({"role": role, "content": content})

        return self._hf_tokenizer.apply_chat_template(
            formatted,
            add_generation_prompt=True,
            tokenize=False,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: Conversation history [{role, content}, ...]
            tools: Optional tool definitions
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            images: Optional list of base64-encoded images to attach to the last user message

        Returns:
            Response dict with 'message' key containing content
        """
        if self.provider == "huggingface":
            return await self._chat_huggingface(
                messages, temperature, max_tokens
            )
        return await self._chat_ollama(
            messages, tools, temperature, max_tokens, images
        )

    async def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Chat via Ollama API."""
        assert self._client is not None, "Call startup() first"

        # Attach images to the last user message if provided
        msgs = [dict(m) for m in messages]
        if images and msgs:
            # Find last user message and attach images
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i]["images"] = images
                    break

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens or self.config.max_tokens,
                "num_ctx": self.config.context_window,
            },
        }

        # Disable thinking mode for Qwen3.5 (prevents reasoning token overhead)
        if "qwen3.5" in self.config.model.lower():
            payload["think"] = False

        if tools:
            payload["tools"] = tools

        logger.debug("LLM request: %d messages, %d images", len(msgs), len(images or []))

        resp = await self._client.post(_API_CHAT, json=payload)
        resp.raise_for_status()
        result = resp.json()

        logger.debug("LLM response: %d chars",
                      len(result.get("message", {}).get("content", "")))
        return result

    async def _chat_huggingface(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat via HuggingFace Transformers."""
        assert self._hf_model and self._hf_tokenizer

        # Apply chat template
        prompt = self._apply_chat_template(messages)

        loop = asyncio.get_event_loop()

        def _generate():
            inputs = self._hf_tokenizer(
                prompt, return_tensors="pt"
            ).to(self._hf_model.device)

            output = self._hf_model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                max_new_tokens=max_tokens or self.config.max_tokens,
                pad_token_id=self._hf_tokenizer.pad_token_id,
                eos_token_id=self._hf_tokenizer.eos_token_id,
            )

            # Decode only new tokens
            new_tokens = output[0][inputs.input_ids.shape[1]:]
            return self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True)

        content = await loop.run_in_executor(None, _generate)

        logger.debug("HF response: %d chars", len(content))
        return {"message": {"role": "assistant", "content": content}}

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion, yielding text chunks.

        Note: Streaming only for final answer (not tool-call iterations).
        """
        if self.provider == "huggingface":
            async for chunk in self._chat_stream_huggingface(
                messages, temperature, max_tokens
            ):
                yield chunk
        else:
            async for chunk in self._chat_stream_ollama(
                messages, temperature, max_tokens
            ):
                yield chunk

    async def _chat_stream_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream via Ollama API."""
        assert self._client is not None, "Call startup() first"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens or self.config.max_tokens,
                "num_ctx": self.config.context_window,
            },
        }

        # Disable thinking mode for Qwen3.5 (prevents reasoning token overhead)
        if "qwen3.5" in self.config.model.lower():
            payload["think"] = False

        async with self._client.stream("POST", _API_CHAT, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                import json as _json
                try:
                    data = _json.loads(line)
                except Exception:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    async def _chat_stream_huggingface(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream via HuggingFace (simulated - yields full at once for now)."""
        # For now, full generation then yield token by token
        result = await self._chat_huggingface(messages, temperature, max_tokens)
        content = result.get("message", {}).get("content", "")

        # Yield token by token (simple char splitting for demo)
        for char in content:
            yield char
            await asyncio.sleep(0.01)  # Simulate streaming

    async def generate(self, prompt: str) -> str:
        """Simple single-shot generation (no chat history)."""
        if self.provider == "huggingface":
            loop = asyncio.get_event_loop()

            def _generate():
                inputs = self._hf_tokenizer(
                    prompt, return_tensors="pt"
                ).to(self._hf_model.device)
                output = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                )
                return self._hf_tokenizer.decode(
                    output[0], skip_special_tokens=True
                )

            return await loop.run_in_executor(None, _generate)

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
