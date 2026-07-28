"""
JARVIS ReAct Agent
===================
Reasoning + Acting loop with two tool-calling paths:

1. NATIVE (preferred): Ollama's tool-calling API — reliable, structured.
2. FALLBACK: text parsing (<|tool_call_start|>[fn(args)] / JSON blocks) for
   models/providers without native support (auto-detected, e.g. HuggingFace).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Maximum tool-calling iterations to prevent infinite loops
MAX_ITERATIONS = 8

# Retries for transient LLM failures / empty responses
_MAX_LLM_RETRIES = 2

# Keywords that trigger vision (screenshot analysis)
_VISION_KEYWORDS = re.compile(
    r"\b(screen|screenshot|what do you see|what's on my screen|"
    r"this window|this dialog|this error|this message|"
    r"look at|can you see|show me|"
    r"click the|where is|find the|locate the|"
    r"button|icon|menu bar|taskbar|desktop)\b",
    re.IGNORECASE,
)

# Fallback tool-call format instructions — only used when native tool
# calling is unavailable (HuggingFace provider or unsupported models).
TOOL_PROMPT_TEMPLATE = """
You have tools. To call one, use ONLY this format:
<|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>

If no tool is needed, reply with plain text.

Tools:
{tool_descriptions}

RULES:
- ONE tool call per response
- NEVER say you did something without calling a tool first
- If the user asks you to do something (search, open, click, navigate), you MUST call the appropriate tool
- After getting a tool result, respond to the user based on that result
- Be concise — responses will be spoken aloud
"""

# Phrases that indicate the model is claiming to have done an action
_ACTION_CLAIM_PATTERNS = re.compile(
    r"\b(has opened|has been opened|i opened|i searched|i navigated|i launched|i started|"
    r"i(?:'ve| have) opened|i(?:'ve| have) searched|i(?:'ve| have) navigated|"
    r"i(?:'ve| have) launched|i(?:'ve| have) started|"
    r"has launched|opened for you|searched for|navigated to|"
    r"calling the .+ tool|has been saved|i(?:'ve| have) saved|i(?:'ve| have) typed|"
    r"calculator has|browser has|chrome has|youtube has|google has)\b",
    re.IGNORECASE,
)

# Markers that may begin an inline (fallback) tool call in streamed text
_STREAM_MARKERS = ("<|", "```json", '{"tool"')


def _build_tool_descriptions(registry: ToolRegistry) -> str:
    """Build compact tool descriptions to minimize prompt tokens."""
    lines = []
    for name in registry.list_tools():
        tool = registry.get(name)
        if tool is None:
            continue
        params = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        parts = []
        for k, v in params.items():
            req = "*" if k in required else ""
            parts.append(f"{k}{req}:{v.get('type','any')}")
        param_str = ", ".join(parts) if parts else ""
        lines.append(f"- {name}({param_str}) — {tool.description}")
    return "\n".join(lines)


# LFM special tokens for tool calling
_LFM_TOOL_CALL_START = "<|tool_call_start|>"
_LFM_TOOL_CALL_END = "<|tool_call_end|>"


def _parse_func_call(call_text: str) -> dict[str, Any] | None:
    """
    Parse a Python-style function call: tool_name(arg="value", arg2=123).
    Positional arguments are returned under "_positional" for the caller
    to map onto the tool's declared parameter order.
    """
    func_match = re.match(r"(\w+)\((.*)\)", call_text.strip(), re.DOTALL)
    if not func_match:
        return None
    tool_name = func_match.group(1)
    args_str = func_match.group(2).strip()
    args: dict[str, Any] = {}
    positional: list[Any] = []

    def _convert(val: str) -> Any:
        if val.startswith('"') and val.endswith('"') and len(val) >= 2:
            # Unescape \" and \\ inside double-quoted strings
            return val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if val.startswith("'") and val.endswith("'") and len(val) >= 2:
            return val[1:-1].replace("\\'", "'")
        low = val.lower()
        if low in ("true",):
            return True
        if low in ("false",):
            return False
        if low in ("null", "none"):
            return None
        try:
            return float(val) if "." in val else int(val)
        except ValueError:
            return val

    # Named args: key="value" / key='value' / key=123 / key=true
    named_pattern = (
        r'(\w+)=(?:("(?:[^"\\]|\\.)*")|(\'(?:[^\'\\]|\\.)*\')|'
        r'(-?\d+\.?\d*)|(true|false|True|False|NULL|null|None))'
    )
    named_spans: list[tuple[int, int]] = []
    for m in re.finditer(named_pattern, args_str):
        key = m.group(1)
        raw = next(g for g in m.groups()[1:] if g is not None)
        args[key] = _convert(raw)
        named_spans.append(m.span())

    # Anything not consumed by a named arg may be a positional value
    remainder = args_str
    for start, end in reversed(named_spans):
        remainder = remainder[:start] + remainder[end:]
    for piece in remainder.split(","):
        piece = piece.strip()
        if piece:
            positional.append(_convert(piece))

    result: dict[str, Any] = {"tool": tool_name, "args": args}
    if positional:
        result["_positional"] = positional
    return result


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """
    Try to extract a tool call from the LLM response.
    Handles: LFM markers, JSON blocks, raw JSON variants, and
    Python-style function calls.
    """
    def _clean(parsed: dict) -> dict:
        parsed["tool"] = re.sub(r"\(.*\)$", "", parsed["tool"]).strip()
        return parsed

    def _from_json(parsed: dict) -> dict | None:
        """Accept {"tool":...,"args":...}, {"name":...,"arguments":...},
        and native {"function": {...}} shapes."""
        if "tool" in parsed:
            return _clean(parsed)
        if "name" in parsed:
            return _clean({
                "tool": parsed["name"],
                "args": parsed.get("arguments", parsed.get("args", {})),
            })
        fn = parsed.get("function")
        if isinstance(fn, dict) and "name" in fn:
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return _clean({"tool": fn["name"], "args": args})
        return None

    # ── LFM format (flexible) ──
    lfm_variants = [
        re.escape(_LFM_TOOL_CALL_START) + r"\[(.*?)\]" + re.escape(_LFM_TOOL_CALL_END),
        re.escape(_LFM_TOOL_CALL_START) + r"(.*?)" + re.escape(_LFM_TOOL_CALL_END),
    ]
    for pattern in lfm_variants:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            call_text = match.group(1).strip()
            parsed = _parse_func_call(call_text)
            if parsed:
                return parsed

    # ── Loose heuristic: any func(args) near tool call markers ──
    if "tool_call" in text.lower():
        func_match = re.search(r"(\w+)\((.*?)\)", text, re.DOTALL)
        if func_match:
            parsed = _parse_func_call(func_match.group(0))
            if parsed:
                return parsed

    # ── JSON code blocks ──
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            parsed = _from_json(json.loads(code_block.group(1)))
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # ── Raw JSON with a tool-ish key ──
    json_match = re.search(r'\{\s*"(?:tool|name|function)"\s*:', text)
    if json_match:
        start = json_match.start()
        brace_count = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        parsed = _from_json(json.loads(text[start:i + 1]))
                        if parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def _normalize_native_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Ollama native tool_calls → [{tool, args}, ...]."""
    normalized = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            normalized.append({"tool": name, "args": args})
    return normalized


class ReActAgent:
    """
    ReAct agent with native tool calling (preferred) and a text-parsing
    fallback. `run()` collects the full response; `run_stream()` streams
    the final answer token-by-token as it is generated (true streaming).
    """

    # Keywords that signal the user wants browser/web tools
    _WEB_KEYWORDS = re.compile(
        r"\b(browse|browser|web|website|search|google|youtube|url|http|navigate|"
        r"open.*site|go to|look up|surf|online|internet|wiki|reddit|github|stackoverflow|"
        r"facebook|twitter|instagram|linkedin|amazon|netflix|spotify)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        browser_tools: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self._browser_tools = browser_tools
        self._base_system_prompt = system_prompt
        # Pre-build descriptions for the fallback (text-prompt) path
        self._system_tool_desc = _build_tool_descriptions(tools)
        self._browser_tool_desc = (
            _build_tool_descriptions(browser_tools) if browser_tools else ""
        )
        # None = native tool support unknown (optimistic), False = disabled
        self._native_tools_ok: bool | None = (llm.provider == "ollama") or None

    def _needs_browser_tools(self, text: str) -> bool:
        """Check if the user message implies web/browser usage."""
        return bool(self._WEB_KEYWORDS.search(text))

    def _get_tool_descriptions(self, include_browser: bool) -> str:
        """Return tool descriptions, optionally including browser tools."""
        if include_browser and self._browser_tool_desc:
            return self._system_tool_desc + "\n" + self._browser_tool_desc
        return self._system_tool_desc

    def _get_ollama_tools(self, include_browser: bool) -> list[dict[str, Any]]:
        """Native tool schemas for the active tool set."""
        schemas = self.tools.get_ollama_tools()
        if include_browser and self._browser_tools:
            schemas += self._browser_tools.get_ollama_tools()
        return schemas

    def _resolve_tool(self, name: str):
        """Look up a tool in system tools first, then browser tools."""
        tool = self.tools.get(name)
        if tool is None and self._browser_tools:
            tool = self._browser_tools.get(name)
        return tool

    async def _execute_tool(self, name: str, **kwargs):
        """Execute a tool from either registry."""
        if self.tools.get(name):
            return await self.tools.execute(name, **kwargs)
        if self._browser_tools and self._browser_tools.get(name):
            return await self._browser_tools.execute(name, **kwargs)
        raise ValueError(f"Unknown tool: {name}")

    def _build_system_prompt(
        self,
        context: list[str] | None = None,
        include_browser: bool = False,
        use_native: bool = True,
    ) -> str:
        """Build the full system prompt with tools and context."""
        prompt = self._base_system_prompt

        # Fallback path needs inline format instructions; native path
        # gets tool schemas via the API and needs no extra tokens.
        if not use_native:
            tool_desc = self._get_tool_descriptions(include_browser)
            if tool_desc:
                prompt += TOOL_PROMPT_TEMPLATE.format(tool_descriptions=tool_desc)

        # Add memory context
        if context:
            memory_block = "\n".join(f"- {c}" for c in context)
            prompt += f"\n\nRelevant memories about this user:\n{memory_block}"

        return prompt

    async def _capture_screenshot(self) -> list[str] | None:
        """Capture the screen, downsize for vision model, and return as base64-encoded PNG."""
        try:
            import pyautogui
            img = pyautogui.screenshot()
            # Downsize to reduce vision encoder latency (~60-70% smaller payload)
            img = img.resize((672, 448))
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            logger.debug("Screenshot captured: %d bytes base64", len(b64))
            return [b64]
        except Exception as e:
            logger.warning("Screenshot capture failed: %s", e)
            return None

    def _needs_vision(self, text: str) -> bool:
        """Check if the user message implies visual context is needed."""
        return bool(_VISION_KEYWORDS.search(text))

    def _map_positional_args(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Map positional args onto the tool's declared parameter order."""
        positional = tool_call.pop("_positional", None)
        if not positional:
            return tool_call
        tool_def = self._resolve_tool(tool_call.get("tool", ""))
        if tool_def is None:
            return tool_call
        declared = list(tool_def.parameters.get("properties", {}).keys())
        for i, value in enumerate(positional):
            if i < len(declared) and declared[i] not in tool_call["args"]:
                tool_call["args"][declared[i]] = value
        return tool_call

    async def _chat_with_retry(self, messages, tools, images):
        """Non-streamed chat with retries for transient failures."""
        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                return await self.llm.chat(messages=messages, tools=tools, images=images)
            except (TimeoutError, httpx.HTTPError) as e:
                if attempt >= _MAX_LLM_RETRIES:
                    raise
                wait = 0.5 * (attempt + 1)
                logger.warning("LLM request failed (%s), retrying in %.1fs", e, wait)
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def _run_impl(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None,
        context: list[str] | None,
        stream: bool,
    ) -> AsyncIterator[str]:
        """
        Core ReAct loop. Yields response tokens (all at once when
        stream=False, incrementally when stream=True).
        """
        # ── Build messages ──
        include_browser = self._needs_browser_tools(user_message)
        if include_browser:
            logger.debug("Browser tools activated for this request")

        images: list[str] | None = None
        if self._needs_vision(user_message):
            logger.info("Vision triggered for: %s", user_message[:80])
            images = await self._capture_screenshot()

        use_native = self._native_tools_ok is True
        system_prompt = self._build_system_prompt(
            context, include_browser=include_browser, use_native=use_native,
        )
        if images:
            system_prompt += (
                "\n\nVISION MODE: A screenshot is attached to the user's message. "
                "Describe or analyze what you see in the image directly. "
                "Do NOT call tools to capture or analyze the image — it is already provided."
            )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        if images:
            messages.append({
                "role": "user",
                "content": f"{user_message}\n\n[I've attached a screenshot. Answer based on what you see in the image.]",
            })
        else:
            messages.append({"role": "user", "content": user_message})

        native_tools = self._get_ollama_tools(include_browser) if use_native else None

        seen_tool_calls: set[tuple[str, str]] = set()
        tool_was_called = False
        claim_retries = 0
        empty_retries = 0
        content = ""

        for iteration in range(MAX_ITERATIONS):
            logger.debug("ReAct iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            # ── Get LLM response (streamed or one-shot) ──
            tool_calls_raw: list[dict[str, Any]] = []
            try:
                if stream:
                    # True streaming: tokens flow to the caller as they
                    # arrive; text that could be a fallback tool call is
                    # held back so markup is never spoken.
                    flushed: list[str] = []
                    pending = ""
                    suspicious = False
                    done_event: dict[str, Any] | None = None
                    async for event in self.llm.chat_stream_events(
                        messages=messages,
                        tools=native_tools,
                        images=images if iteration == 0 else None,
                    ):
                        if event["type"] == "done":
                            done_event = event
                            break
                        token = event["text"]
                        pending += token
                        if suspicious:
                            continue
                        cut = -1
                        for marker in _STREAM_MARKERS:
                            idx = pending.find(marker)
                            if idx != -1:
                                cut = idx if cut == -1 else min(cut, idx)
                        if cut != -1:
                            if cut > 0:
                                flushed.append(pending[:cut])
                                yield pending[:cut]
                            pending = pending[cut:]
                            suspicious = True
                        elif len(pending) > 2:
                            # Flush all but a 2-char tail (split markers)
                            flushed.append(pending[:-2])
                            yield pending[:-2]
                            pending = pending[-2:]
                    if done_event is not None:
                        tool_calls_raw = done_event.get("tool_calls", [])
                    content = "".join(flushed) + pending
                    # Held-back text that isn't a tool call is innocent — flush it
                    if pending and not tool_calls_raw and not _parse_tool_call(content):
                        yield pending
                else:
                    response = await self._chat_with_retry(
                        messages, native_tools, images if iteration == 0 else None,
                    )
                    msg = response.get("message", {})
                    content = msg.get("content", "")
                    tool_calls_raw = msg.get("tool_calls", [])
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text[:300]
                except Exception:
                    pass
                if e.response.status_code == 400 and "tool" in body.lower():
                    # Model doesn't support native tool calling — fall back
                    logger.warning("Native tools unsupported, switching to text parsing")
                    self._native_tools_ok = False
                    native_tools = None
                    messages[0]["content"] = self._build_system_prompt(
                        context, include_browser=include_browser, use_native=False,
                    )
                    continue
                raise

            logger.info("LLM raw (iter %d): %s", iteration + 1, content[:300])

            # ── Handle empty responses with a retry ──
            if not content and not tool_calls_raw:
                empty_retries += 1
                if empty_retries <= _MAX_LLM_RETRIES:
                    logger.warning("Empty LLM response (iter %d), retrying", iteration + 1)
                    await asyncio.sleep(0.5)
                    continue
                logger.warning("Empty LLM response after retries")
                yield "I couldn't generate a response."
                return

            # ── Resolve tool call(s): native first, fallback parse second ──
            parsed_calls: list[dict[str, Any]] = []
            native_call = False
            if tool_calls_raw:
                parsed_calls = _normalize_native_calls(tool_calls_raw)
                native_call = True
            else:
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                fallback = _parse_tool_call(content) or _parse_tool_call(cleaned)
                if fallback:
                    parsed_calls = [self._map_positional_args(fallback)]

            # ── Final answer? ──
            if not parsed_calls:
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if not tool_was_called and claim_retries < 1 and \
                        _ACTION_CLAIM_PATTERNS.search(cleaned):
                    claim_retries += 1
                    logger.warning("Model claimed action without tool call, forcing retry")
                    messages.append({"role": "assistant", "content": content})
                    if native_tools is not None:
                        nudge = (
                            "You said you did something but you did NOT call any tool. "
                            "You MUST use one of the provided tools to perform actions. "
                            "Actually call the tool now."
                        )
                    else:
                        nudge = (
                            "You said you did something but you did NOT call any tool. "
                            "You MUST use a tool to perform actions. Reply with ONLY the "
                            "tool call in the format <|tool_call_start|>[tool_name(arg=\"value\")]<|tool_call_end|>."
                        )
                    messages.append({"role": "user", "content": nudge})
                    continue

                final = cleaned if cleaned else content
                if not stream:
                    yield final
                return

            # ── Execute tool call(s) ──
            if native_call:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls_raw,
                })
            else:
                messages.append({"role": "assistant", "content": content})

            for tool_call in parsed_calls:
                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("args", {})

                # Dedup: same tool+args already executed this run
                call_key = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
                if call_key in seen_tool_calls:
                    logger.warning("Skipping repeated tool call: %s", tool_name)
                    dup_msg = (
                        f"You already called '{tool_name}' with these exact arguments. "
                        "Do NOT repeat it. Either use a DIFFERENT tool, or give your "
                        "final text response now."
                    )
                    if native_call:
                        messages.append({"role": "tool", "name": tool_name, "content": dup_msg})
                    else:
                        messages.append({"role": "user", "content": dup_msg})
                    continue
                seen_tool_calls.add(call_key)

                logger.info("Agent calling tool: %s(%s)", tool_name, tool_args)
                tool_was_called = True

                try:
                    result = await self._execute_tool(tool_name, **tool_args)
                    result_str = str(result)
                except ValueError:
                    result_str = (
                        f"Error: Unknown tool '{tool_name}'. "
                        f"Available: {self.tools.list_tools()}"
                    )
                    logger.warning("Unknown tool: %s", tool_name)
                except Exception as e:
                    result_str = f"Error executing {tool_name}: {e}"
                    logger.error("Tool '%s' failed: %s", tool_name, e)

                if native_call:
                    messages.append({
                        "role": "tool", "name": tool_name, "content": result_str,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool '{tool_name}' returned:\n{result_str}\n\n"
                            "If the user's request is fully completed, give a brief final "
                            "response. If more steps are needed, call the next tool."
                        ),
                    })

        # ── Exhausted iterations ──
        logger.warning("ReAct loop exhausted %d iterations", MAX_ITERATIONS)
        closing = (
            "I've completed what I could. Let me know if you need anything else."
            if tool_was_called
            else (content if content else "I ran into too many steps. Please try again.")
        )
        yield closing

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: list[str] | None = None,
    ) -> str:
        """Process a user message through the ReAct loop. Returns the response."""
        parts: list[str] = []
        async for token in self._run_impl(
            user_message, conversation_history, context, stream=False,
        ):
            parts.append(token)
        return "".join(parts)

    async def run_stream(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Same as run(), but streams the answer token-by-token as it is
        generated — including the final answer after any tool calls.
        """
        async for token in self._run_impl(
            user_message, conversation_history, context, stream=True,
        ):
            yield token

    # ==========================================================================
    # AUTONOMOUS RESEARCH MODE
    # ==========================================================================

    async def research(self, query: str, max_sources: int = 5) -> str:
        """
        Perform autonomous research on a topic.

        The agent will:
        1. Search the web for relevant sources
        2. Browse and read promising pages
        3. Extract key information
        4. Compile a concise summary report

        Args:
            query: The research question or topic
            max_sources: Maximum number of sources to consult

        Returns:
            A research summary with sources cited
        """
        logger.info("Starting research on: %s", query[:80])

        system_prompt = (
            "You are a research analyst. Your goal is to thoroughly investigate "
            "the user's query using web search and browser tools, then compile "
            "a concise, factual summary with citations.\n\n"
            "RESEARCH PROCESS:\n"
            "1. Use web_search to find relevant sources\n"
            "2. Use open_url to read promising pages\n"
            "3. Use read_pdf if PDF documents are found\n"
            "4. Synthesize findings into a brief report\n\n"
            "RULES:\n"
            "- Be factual and cite your sources\n"
            "- Limit to the most relevant information\n"
            "- If search returns no results, say so\n"
            "- Respond with a concise summary suitable for spoken TTS"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research this topic and give me a summary: {query}"},
        ]

        # Use the standard ReAct loop but with research-specific prompt
        self._seen_tool_calls: set[tuple[str, str]] = set()
        sources_consulted = 0

        for iteration in range(MAX_ITERATIONS):
            response = await self.llm.chat(messages=messages)
            content = response.get("message", {}).get("content", "")
            logger.info("Research iter %d: %s", iteration + 1, content[:300])

            if not content:
                return "I couldn't complete the research."

            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            tool_call = _parse_tool_call(content) or _parse_tool_call(cleaned)

            if tool_call is None:
                return cleaned if cleaned else content

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # Track sources
            if tool_name in ("open_url", "read_pdf"):
                sources_consulted += 1
                if sources_consulted > max_sources:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You've consulted {max_sources} sources. "
                            "Please synthesize what you've learned into a final summary now."
                        ),
                    })
                    continue

            # Prevent repeats
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_key in self._seen_tool_calls:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You already consulted that source. "
                        "Please synthesize your findings into a final summary."
                    ),
                })
                continue
            self._seen_tool_calls.add(call_key)

            messages.append({"role": "assistant", "content": content})

            try:
                result = await self._execute_tool(tool_name, **tool_args)
                result_str = str(result)[:4000]  # Limit context length
            except Exception as e:
                result_str = f"Error: {e}"

            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' result:\n{result_str}\n\n"
                    f"Continue researching or provide your final summary."
                ),
            })

        return "Research complete. Here is what I found:\n\n" + (content if content else "")

    # ==========================================================================
    # SEE-THINK-ACT DESKTOP AUTOMATION LOOP
    # ==========================================================================

    async def computer_use(self, goal: str, max_steps: int = 15) -> str:
        """
        Autonomously accomplish a desktop task via screenshot → think → act loop.

        Two-phase approach:
        1. PLANNING: Model creates a step-by-step plan without screenshots
        2. EXECUTION: Each step is executed with screenshot verification

        Args:
            goal: The desktop task to accomplish (e.g., "Open Notepad and type Hello")
            max_steps: Maximum number of action iterations

        Returns:
            A summary of what was accomplished
        """
        logger.info("Computer-use mode: %s", goal[:80])

        import base64
        import io

        import pyautogui

        # ── Phase 1: Planning ──
        plan_prompt = (
            f"Goal: {goal}\n\n"
            "Create a concise step-by-step plan to accomplish this goal. "
            "Each step must use one of these tools:\n"
            "- open_application(app_name) — launch an app by name\n"
            "- click_ui_element(element_name) — click a visible UI element\n"
            "- type_text(text) — type text at current cursor\n"
            "- send_keystrokes(keys) — press keys like 'win+r', 'ctrl+a'\n\n"
            "Format your plan as a numbered list. Be specific with element names."
        )

        plan_resp = await self.llm.chat(messages=[
            {"role": "system", "content": "You are a desktop automation planner."},
            {"role": "user", "content": plan_prompt},
        ])
        plan = plan_resp.get("message", {}).get("content", "")
        logger.info("Computer-use plan:\n%s", plan)

        # Extract steps from plan (look for numbered lines with tool names)
        steps: list[dict[str, Any]] = []
        for line in plan.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try to find tool calls in the plan line
            tc = _parse_tool_call(line)
            if tc is None:
                # Fallback: look for patterns like "open_application(..."
                m = re.search(r"(\w+)\s*\(([^)]*)\)", line)
                if m:
                    tname = m.group(1)
                    targs = m.group(2).strip()
                    args: dict[str, Any] = {}
                    # Named args: param="value" or param='value'
                    for kv in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', targs):
                        args[kv.group(1)] = kv.group(2)
                    for kv in re.finditer(r"(\w+)\s*=\s*'(.*?)'", targs):
                        args[kv.group(1)] = kv.group(2)
                    # Positional args: "value", 'value', or bareword — map to known params
                    if not args and targs:
                        pos = re.search(r'^["\']([^"\']+)["\']$', targs)
                        if pos:
                            val = pos.group(1)
                        else:
                            val = targs  # bareword like notepad
                        param_map = {
                            "open_application": "app_name",
                            "click_ui_element": "element_name",
                            "type_text": "text",
                            "send_keystrokes": "keys",
                            "take_screenshot": "save_path",
                        }
                        if tname in param_map:
                            args[param_map[tname]] = val
                    tc = {"tool": tname, "args": args}
            if tc:
                steps.append(tc)

        if not steps:
            return f"Could not create a plan for: {goal}"

        logger.info("Parsed %d executable steps", len(steps))

        # ── Phase 2: Execution ──
        system_prompt = (
            "You are verifying desktop automation steps. "
            "A screenshot is attached. Confirm if the expected UI state is visible. "
            "If yes, say CONTINUE. If no, say ADJUST: <alternative action>"
        )

        actions_taken: list[str] = []

        for step_idx, step in enumerate(steps[:max_steps]):
            tool_name = step.get("tool", "")
            tool_args = step.get("args", {})

            logger.info("Step %d/%d: %s(%s)", step_idx + 1, len(steps), tool_name, tool_args)

            # Execute the planned action
            try:
                result = await self._execute_tool(tool_name, **tool_args)
                result_str = str(result)[:500]
                actions_taken.append(f"{tool_name}({tool_args}) -> {result_str[:100]}")
            except Exception as e:
                result_str = f"Error: {e}"
                actions_taken.append(f"{tool_name}({tool_args}) -> FAILED: {e}")

            logger.info("Step %d result: %s", step_idx + 1, result_str[:200])

            # Screenshot verification (optional but helpful for debugging)
            if step_idx < len(steps) - 1:
                img = pyautogui.screenshot()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                verify_msg = (
                    f"Goal: {goal}\n"
                    f"Just executed: {tool_name}({tool_args})\n"
                    f"Result: {result_str[:200]}\n\n"
                    "Is the next step ready to execute? Reply CONTINUE or ADJUST: <new action>"
                )

                verify_resp = await self.llm.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": verify_msg, "images": [img_b64]},
                    ]
                )
                verify = verify_resp.get("message", {}).get("content", "").strip().upper()
                logger.debug("Verification: %s", verify)

                if verify.startswith("ADJUST:"):
                    # Parse alternative action
                    alt_text = verify[7:].strip()
                    alt_tc = _parse_tool_call(alt_text)
                    if alt_tc:
                        steps.insert(step_idx + 1, alt_tc)
                        logger.info("Adjusted plan: added %s", alt_tc)

        return (
            f"Task: {goal}\n\n"
            f"Plan executed ({len(actions_taken)} steps):\n"
            + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions_taken))
        )
