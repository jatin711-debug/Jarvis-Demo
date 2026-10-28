"""
JARVIS LLM Client
==================
Multi-provider LLM client:
- Ollama: local Ollama API (default)
- llama.cpp: a running ``llama-server`` (OpenAI-compatible API, GGUF models)
- HuggingFace: Direct Transformers inference (for LFM, etc.)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any, NoReturn

import httpx

from jarvis.agent.llama_server import LlamaServer
from jarvis.utils.config import LLMConfig
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

# Ollama API endpoints
_API_CHAT = "/api/chat"
_API_TAGS = "/api/tags"

# llama-server (llama.cpp) endpoints
_API_OPENAI_CHAT = "/v1/chat/completions"
_API_HEALTH = "/health"
_API_PROPS = "/props"
_LLAMACPP_READY_WAIT_S = 90.0   # a cold model load can take this long; /health answers 503 meanwhile
_SERVER_EXIT_GRACE_S = 1.0      # a dropped connection can be seen just before the exit is reported


class LLMClient:
    """
    Async LLM client supporting multiple providers.

    Providers:
    - Ollama: local Ollama API (default)
    - llama.cpp: a running llama-server (GGUF models)
    - HuggingFace: Direct Transformers inference (LFM, etc.)
    """

    def __init__(self, config: LLMConfig, log_dir: Path | None = None) -> None:
        self.config = config
        self.provider = config.provider.lower()
        self.base_url = config.base_url.rstrip("/")
        self._log_dir = log_dir or Path(tempfile.gettempdir())
        self._client: httpx.AsyncClient | None = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_lock = asyncio.Lock()
        self._server: LlamaServer | None = None  # llama-server, when Jarvis started it
        self._server_lock = asyncio.Lock()  # one restart at a time
        self._server_vision = False  # llama-server: was it started with a multimodal projector?

    @property
    def server_pid(self) -> int | None:
        """Process id of the llama-server Jarvis launched, or None (not launched by Jarvis)."""
        return self._server.pid if self._server is not None else None

    async def startup(self) -> None:
        """Initialize the LLM client based on provider."""
        if self.provider == "huggingface":
            await self._startup_huggingface()
        elif self.provider == "llamacpp":
            await self._startup_llamacpp()
        else:
            await self._startup_ollama()

    async def _startup_llamacpp(self) -> None:
        """Connect to llama-server, waiting for it to finish loading its model."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.request_timeout, connect=10.0),
        )
        if self.config.server.model and await self._llamacpp_status() is None:
            self._server = LlamaServer(self.config, self._log_dir / "llama-server.log")
            try:
                await self._server.start()
            except RuntimeError as exc:
                await self._abort_startup(str(exc))

        try:
            await self._await_llamacpp_ready()
        except RuntimeError as exc:
            await self._abort_startup(str(exc))

        props = await self._server_props()
        self._server_vision = bool((props.get("modalities") or {}).get("vision"))
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        logger.info("llama-server connected url=%s model=%s n_ctx=%s vision=%s",
                    self.base_url, props.get("model_alias") or self.config.model,
                    n_ctx, self._server_vision)
        if isinstance(n_ctx, int) and n_ctx < self.config.context_window:
            logger.warning("llama-server context (%d) is smaller than llm.context_window (%d); "
                           "start it with -c %d", n_ctx, self.config.context_window,
                           self.config.context_window)

    async def _await_llamacpp_ready(self) -> None:
        """Wait until /health says the model is loaded; RuntimeError says why it cannot be."""
        loop = asyncio.get_running_loop()
        server = self._server
        wait_s = self.config.server.startup_timeout if server else _LLAMACPP_READY_WAIT_S
        deadline = loop.time() + wait_s
        announced = False
        while True:
            status = await self._llamacpp_status()
            if status == 200:
                return
            if server is not None and not server.alive():
                raise RuntimeError(
                    "llama-server exited during startup. Last output:\n" + server.log_tail())
            if status is None and server is None:
                raise RuntimeError(
                    f"Cannot reach llama-server at {self.base_url}. Start it first, e.g.: "
                    "llama-server -m <model.gguf> --port 8080 -ngl 99 -fa on, or set "
                    "llm.server.model so Jarvis starts it (see docs/llamacpp.md)")
            if status not in (None, 503):
                raise RuntimeError(f"llama-server at {self.base_url} is unhealthy (HTTP {status})")
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"llama-server at {self.base_url} is still loading its model after {wait_s:.0f}s")
            if not announced:
                logger.info("waiting for llama-server to load its model url=%s", self.base_url)
                announced = True
            await asyncio.sleep(1.0)

    async def _ensure_llamacpp_server(self) -> None:
        """Restart a llama-server that Jarvis launched and that has since exited.

        A server somebody else started is never touched: if it goes away, requests fail
        and the error says so.
        """
        server = self._server
        if server is None or server.alive():
            return
        async with self._server_lock:
            if self._server is not server or server.alive():
                return  # shut down, or another request already restarted it
            logger.warning("llama-server exited code=%s; restarting. Last output:\n%s",
                           server.exit_code, server.log_tail())
            await server.stop()  # releases the old log handle
            try:
                await server.start()
                await self._await_llamacpp_ready()
            except RuntimeError:
                await server.stop()  # a server stuck while loading must not linger
                raise
            logger.info("llama-server restarted url=%s", self.base_url)

    async def _llamacpp_status(self) -> int | None:
        """HTTP status of llama-server's /health, or None when nothing answers."""
        assert self._client is not None
        try:
            return (await self._client.get(_API_HEALTH)).status_code
        except httpx.HTTPError:
            return None

    async def _server_props(self) -> dict[str, Any]:
        """llama-server's /props, or {} when unavailable (it is informational only)."""
        assert self._client is not None
        try:
            resp = await self._client.get(_API_PROPS)
            resp.raise_for_status()
            props = resp.json()
        except (httpx.HTTPError, ValueError):
            return {}
        return props if isinstance(props, dict) else {}

    async def _abort_startup(self, message: str) -> NoReturn:
        """Release what startup acquired (client, a server we launched) and raise ``message``."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._server is not None:
            await self._server.stop()
            self._server = None
        raise RuntimeError(message)

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

        logger.info("ollama connected url=%s model=%s", self.base_url, self.config.model)

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
            logger.warning("ollama model list failed error=%s", e)
            return None

    async def _startup_huggingface(self) -> None:
        """Initialize HuggingFace Transformers model."""
        logger.info("loading huggingface model=%s (first run downloads weights)", self.config.model)

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
        logger.info("huggingface model loaded model=%s", self.config.model)

    async def shutdown(self) -> None:
        """Close the LLM client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._server is not None:
            await self._server.stop()
            self._server = None
        self._hf_model = None
        self._hf_tokenizer = None

    @property
    def supports_images(self) -> bool:
        """Whether this provider can look at attached images.

        The HF path is text-only; llama-server reads images only when it was
        started with a multimodal projector (``--mmproj``).
        """
        if self.provider == "llamacpp":
            return self._server_vision
        return self.provider != "huggingface"

    async def health_check(self) -> bool:
        """Check if provider is running/available."""
        if self.provider == "huggingface":
            return self._hf_model is not None
        if self.provider == "llamacpp":
            if self._client is None:
                return False
            try:
                return (await self._client.get(_API_HEALTH)).status_code == 200
            except httpx.HTTPError:
                return False
        models = await self.list_models()
        if models is None:
            return False
        logger.debug("ollama models available=%s", models)
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
        if self.provider == "llamacpp":
            return await self._chat_llamacpp(messages, tools, temperature, max_tokens, images)
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
        payload = self._ollama_payload(messages, tools, temperature, max_tokens, images,
                                       stream=False)
        logger.debug("llm request messages=%d images=%d", len(payload["messages"]),
                     len(images or []))

        resp = await self._client.post(_API_CHAT, json=payload)
        resp.raise_for_status()
        result = resp.json()

        logger.debug("llm response chars=%d", len(result.get("message", {}).get("content", "")))
        return result

    def _thinking_mode(self) -> bool | None:
        """An explicit setting wins over the backwards-compatible Qwen3.5 default."""
        if self.config.enable_thinking is not None:
            return self.config.enable_thinking
        return False if "qwen3.5" in self.config.model.lower() else None

    def _ollama_payload(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        images: list[str] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """The /api/chat request body shared by streamed and non-streamed calls."""
        msgs = [dict(m) for m in messages]
        if images:
            # Images belong to the most recent user message.
            for message in reversed(msgs):
                if message.get("role") == "user":
                    message["images"] = images
                    break

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens or self.config.max_tokens,
                "num_ctx": self.config.context_window,
            },
        }
        thinking = self._thinking_mode()
        if thinking is not None:
            payload["think"] = thinking
        if tools:
            payload["tools"] = tools
        return payload

    async def _chat_huggingface(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat via HuggingFace Transformers."""
        content = "".join([chunk async for chunk in
                           self._chat_stream_huggingface(messages, temperature, max_tokens)])

        logger.debug("huggingface response chars=%d", len(content))
        return {"message": {"role": "assistant", "content": content}}

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
        raw_stream = self._stream_llamacpp if self.provider == "llamacpp" else self._stream_ollama
        async with aclosing(raw_stream(
            messages, tools, temperature, max_tokens, images
        )) as stream:
            async for kind, payload in stream:
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
        """Raw Ollama stream: yields ("content", token) or ("tool_calls", [...]).

        Raises when the server rejects the request, reports an error mid-stream,
        or closes the stream before the final ``done`` message: a truncated
        reply must not be mistaken for a complete one.
        """
        assert self._client is not None, "Call startup() first"
        payload = self._ollama_payload(messages, tools, temperature, max_tokens, images,
                                       stream=True)

        async with self._client.stream("POST", _API_CHAT, json=payload) as resp:
            if resp.status_code >= 400:
                # A streamed response has no body yet; read it so the caller can
                # tell "this model does not support tools" from other failures.
                await resp.aread()
            resp.raise_for_status()
            finished = False
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    logger.debug("ollama stream line skipped (not json) chars=%d", len(line))
                    continue
                if data.get("error"):
                    raise RuntimeError(f"Ollama reported an error: {data['error']}")
                message = data.get("message", {})
                token = message.get("content", "")
                if token:
                    yield ("content", token)
                calls = message.get("tool_calls")
                if calls:
                    yield ("tool_calls", calls)
                if data.get("done", False):
                    finished = True
                    break
            if not finished:
                raise httpx.RemoteProtocolError("Ollama closed the stream before the reply finished")

    # ── llama.cpp (llama-server, OpenAI-compatible) ─────────

    async def _chat_llamacpp(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        images: list[str] | None,
    ) -> dict[str, Any]:
        """A whole reply, assembled from the stream (llama-server is only called streamed)."""
        content = ""
        tool_calls: list[dict[str, Any]] = []
        async with aclosing(self._stream_llamacpp(
            messages, tools, temperature, max_tokens, images
        )) as stream:
            async for kind, payload in stream:
                if kind == "content":
                    content += payload
                elif kind == "tool_calls":
                    tool_calls.extend(payload)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        logger.debug("llamacpp response chars=%d tool_calls=%d", len(content), len(tool_calls))
        return {"message": message}

    def _openai_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        images: list[str] | None,
    ) -> dict[str, Any]:
        """The streamed /v1/chat/completions request body."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._openai_messages(messages, images),
            "stream": True,
            "cache_prompt": True,  # reuse the KV cache of the unchanged prompt prefix
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        thinking = self._thinking_mode()
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _openai_messages(messages: list[dict[str, Any]],
                         images: list[str] | None) -> list[dict[str, Any]]:
        """Conversation history in OpenAI shape.

        The agent keeps Ollama-shaped history (tool-call arguments as objects,
        tool results identified by name). OpenAI wants JSON-string arguments and
        a ``tool_call_id`` that pairs each result with its call.
        """
        converted: list[dict[str, Any]] = []
        unanswered: list[str] = []
        issued = 0
        for source in messages:
            message = dict(source)
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    issued += 1
                    unanswered.append(f"call_{issued}")
                    calls.append({
                        "id": unanswered[-1],
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": arguments if isinstance(arguments, str)
                            else json.dumps(arguments),
                        },
                    })
                message["tool_calls"] = calls
            elif message.get("role") == "tool":
                message["tool_call_id"] = unanswered.pop(0) if unanswered else "call_0"
            converted.append(message)

        if images:
            # Images belong to the most recent user message.
            for message in reversed(converted):
                if message.get("role") == "user":
                    message["content"] = [{"type": "text", "text": message.get("content", "")}] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                        for image in images
                    ]
                    break
        return converted

    async def _stream_llamacpp(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """``_stream_llamacpp_once`` that survives the server dying.

        A server Jarvis launched is restarted when it is found dead before a request. If it
        dies while the request is in flight and nothing of the reply has been passed on yet,
        it is restarted and the request is sent once more. Once part of a reply has gone out
        it cannot be replayed, so the error is raised and the next request restarts the server.
        """
        for attempt in (1, 2):
            await self._ensure_llamacpp_server()
            emitted = False
            try:
                async with aclosing(self._stream_llamacpp_once(
                        messages, tools, temperature, max_tokens, images)) as stream:
                    async for item in stream:
                        emitted = True
                        yield item
                return
            except httpx.TransportError as exc:
                if (attempt == 2 or emitted or self._server is None
                        or isinstance(exc, httpx.TimeoutException)
                        or not await self._server.wait_exit(_SERVER_EXIT_GRACE_S)):
                    raise
                logger.warning("llama-server died during the request; retrying once error=%s", exc)

    async def _stream_llamacpp_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Raw llama-server stream: yields ("content", token), then ("tool_calls", [...]) once.

        Tool calls arrive as fragments; they are assembled and emitted in the
        Ollama shape (arguments as an object) when the reply finishes. Like the
        Ollama path, a stream that ends before the finish marker is an error.
        """
        assert self._client is not None, "Call startup() first"
        payload = self._openai_payload(messages, tools, temperature, max_tokens, images)

        pending: dict[int, dict[str, str]] = {}
        finished = False
        async with self._client.stream("POST", _API_OPENAI_CHAT, json=payload) as resp:
            if resp.status_code >= 400:
                await resp.aread()  # so the caller can read the error text
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue  # blank separators and ": keep-alive" comments
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    logger.debug("llamacpp stream line skipped (not json) chars=%d", len(data))
                    continue
                if chunk.get("error"):
                    error = chunk["error"]
                    detail = error.get("message") if isinstance(error, dict) else error
                    raise RuntimeError(f"llama-server reported an error: {detail}")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    token = delta.get("content")
                    if token:
                        yield ("content", token)
                    for fragment in delta.get("tool_calls") or []:
                        slot = pending.setdefault(fragment.get("index", 0),
                                                  {"name": "", "arguments": ""})
                        function = fragment.get("function") or {}
                        if function.get("name") and not slot["name"]:
                            slot["name"] = function["name"]
                        slot["arguments"] += function.get("arguments") or ""
                    if choice.get("finish_reason"):
                        finished = True
        if not finished:
            raise httpx.RemoteProtocolError("llama-server closed the stream before the reply finished")

        calls = []
        for _, slot in sorted(pending.items()):
            if not slot["name"]:
                continue
            try:
                arguments = json.loads(slot["arguments"]) if slot["arguments"].strip() else {}
            except ValueError:
                logger.warning("tool call arguments are not valid JSON tool=%s chars=%d",
                               slot["name"], len(slot["arguments"]))
                arguments = {}
            calls.append({"function": {"name": slot["name"], "arguments": arguments}})
        if calls:
            yield ("tool_calls", calls)

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
