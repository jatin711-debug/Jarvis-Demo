"""
JARVIS LLM Client
==================
Multi-provider LLM client:
- Ollama: local Ollama API (default)
- HuggingFace: Direct Transformers inference (for LFM, etc.)
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx

from jarvis.utils.config import LLMConfig
from jarvis.utils.logger import get_logger

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
        self._hf_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Initialize the LLM client based on provider."""
        if self.provider == "huggingface":
            await self._startup_huggingface()
        else:
            await self._startup_ollama()

    async def _startup_ollama(self) -> None:
        """Initialize Ollama HTTP client and verify the configured model exists."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.request_timeout, connect=10.0),
        )

        models = await self.list_models()
        if models is None:
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` and try again."
            )

        if not any(self._model_name_matches(m) for m in models):
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"Model '{self.config.model}' is not pulled. "
                f"Run: ollama pull {self.config.model}\n"
                f"Available models: {', '.join(models) or '(none)'}"
            )

        logger.info("✅ Ollama connected at %s (model=%s)", self.base_url, self.config.model)

    def _model_name_matches(self, tag: str) -> bool:
        """Match 'qwen3:4b' against tags like 'qwen3:4b' or 'name:latest'."""
        want = self.config.model
        return tag == want or tag == f"{want}:latest"

    async def list_models(self) -> list[str] | None:
        """Return pulled model names, or None if Ollama is unreachable."""
        try:
            assert self._client is not None
            resp = await self._client.get(_API_TAGS)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception as e:
            logger.warning("Ollama model list failed: %s", e)
            return None

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
                dtype="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=False,
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
        models = await self.list_models()
        if models is None:
            return False
        logger.debug("Available models: %s", models)
        return True

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
                "temperature": self.config.temperature if temperature is None else temperature,
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
        content = "".join([chunk async for chunk in
                           self._chat_stream_huggingface(messages, temperature, max_tokens)])

        logger.debug("HF response: %d chars", len(content))
        return {"message": {"role": "assistant", "content": content}}

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion, yielding text chunks.

        Note: tool_calls arriving during the stream are silently collected;
        use chat_stream_events() if you need them.
        """
        async for event in self.chat_stream_events(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, images=images,
        ):
            if event["type"] == "content":
                yield event["text"]

    async def chat_stream_events(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream a chat completion as structured events.

        Yields {"type": "content", "text": <token>} for each text chunk,
        then one terminal event:
        {"type": "done", "content": <full text>, "tool_calls": [...]}.
        """
        if self.provider == "huggingface":
            full_text = ""
            async with aclosing(self._chat_stream_huggingface(
                messages, temperature, max_tokens
            )) as chunks:
                async for chunk in chunks:
                    full_text += chunk
                    yield {"type": "content", "text": chunk}
            yield {"type": "done", "content": full_text, "tool_calls": []}
            return

        full_text = ""
        tool_calls: list[dict[str, Any]] = []
        async for kind, payload in self._stream_ollama(
            messages, tools, temperature, max_tokens, images
        ):
            if kind == "content":
                full_text += payload
                yield {"type": "content", "text": payload}
            elif kind == "tool_calls":
                tool_calls.extend(payload)
        yield {"type": "done", "content": full_text, "tool_calls": tool_calls}

    async def _stream_ollama(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Raw Ollama stream: yields ("content", token) or ("tool_calls", [...])."""
        assert self._client is not None, "Call startup() first"

        # Attach images to the last user message if provided
        msgs = [dict(m) for m in messages]
        if images and msgs:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i]["images"] = images
                    break

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
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

        async with self._client.stream("POST", _API_CHAT, json=payload) as resp:
            resp.raise_for_status()
            import json as _json
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = _json.loads(line)
                except Exception:
                    continue
                message = data.get("message", {})
                token = message.get("content", "")
                if token:
                    yield ("content", token)
                calls = message.get("tool_calls")
                if calls:
                    yield ("tool_calls", calls)
                if data.get("done", False):
                    break

    async def _chat_stream_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream via Ollama API (text tokens only)."""
        async for kind, payload in self._stream_ollama(
            messages, None, temperature, max_tokens, images
        ):
            if kind == "content":
                yield payload

    async def _chat_stream_huggingface(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream decoded text from a serialized, cooperatively cancellable worker."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        assert self._hf_model is not None and self._hf_tokenizer is not None
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        stop = threading.Event()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                # Preserve tool markers; remove only conversation boundary tokens.
                for marker in ("<|im_end|>", "<|im_start|>", "<|startoftext|>", "<|endoftext|>"):
                    text = text.replace(marker, "")
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, text)

        async with self._hf_lock:
            prompt = self._apply_chat_template(messages)
            streamer = Streamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=False)
            temp = self.config.temperature if temperature is None else temperature

            def generate():
                try:
                    inputs = self._hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._hf_model.device)
                    options = {"do_sample": temp > 0}
                    if temp > 0:
                        options.update(temperature=temp, top_p=self.config.top_p, top_k=self.config.top_k)
                    with torch.inference_mode():
                        self._hf_model.generate(
                            **inputs, **options, streamer=streamer,
                            stopping_criteria=StoppingCriteriaList([Cancelled()]),
                            repetition_penalty=self.config.repetition_penalty,
                            max_new_tokens=max_tokens or self.config.max_tokens,
                            pad_token_id=self._hf_tokenizer.pad_token_id,
                            eos_token_id=self._hf_tokenizer.eos_token_id,
                        )
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            worker = asyncio.create_task(asyncio.to_thread(generate))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop.set()
                await asyncio.shield(worker)

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
