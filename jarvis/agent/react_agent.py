"""
JARVIS ReAct Agent
===================
Prompt-based Reasoning + Acting loop that works with ANY LLM model.
Uses structured JSON output parsing instead of native tool calling API.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, AsyncIterator, TYPE_CHECKING

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

if TYPE_CHECKING:
    from jarvis.utils.config import LLMConfig

logger = get_logger(__name__)

# Maximum tool-calling iterations to prevent infinite loops
MAX_ITERATIONS = 5

# Keywords that trigger vision (screenshot analysis)
_VISION_KEYWORDS = re.compile(
    r"\b(screen|screenshot|what do you see|what's on my screen|"
    r"this window|this dialog|this error|this message|"
    r"look at|can you see|show me|"
    r"click the|where is|find the|locate the|"
    r"button|icon|menu bar|taskbar|desktop)\b",
    re.IGNORECASE,
)

# System prompt addition that teaches the LLM how to use tools
TOOL_PROMPT_TEMPLATE = """
You have tools. To call one, use ONLY this format:
<|tool_call_start|>[open_app(app_name="notepad")]<|tool_call_end|>

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


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """
    Try to extract a tool call from the LLM response.
    Handles various formats: raw JSON, markdown code blocks, LFM format,
    and common LLM variations (missing brackets, partial tokens, etc.).

    LFM format: <|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>
    """
    def _clean(parsed: dict) -> dict:
        # Strip trailing () from tool name (LLM sometimes copies from description)
        parsed["tool"] = re.sub(r"\(.*\)$", "", parsed["tool"]).strip()
        return parsed

    def _parse_func_call(call_text: str) -> dict[str, Any] | None:
        """Parse a Python-style function call: tool_name(arg=\"value\", arg2=123)"""
        func_match = re.match(r"(\w+)\((.*)\)", call_text, re.DOTALL)
        if not func_match:
            return None
        tool_name = func_match.group(1)
        args_str = func_match.group(2)
        args = {}
        # Parse arguments: key="value" or key='value' or key=123 or key=true/false
        arg_pattern = r'(\w+)=(?:("(?:[^"\\]|\\.)*")|(\'(?:[^\'\\]|\\.)*\')|(\d+\.?\d*)|(true|false|True|False|NULL|null|None))'
        for m in re.finditer(arg_pattern, args_str):
            key = m.group(1)
            val = m.group(2) or m.group(3) or m.group(4) or m.group(5)
            if val is None:
                continue
            # Strip quotes from string values
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Convert number strings to actual numbers
            try:
                if '.' in str(val):
                    val = float(val)
                else:
                    val = int(val)
            except (ValueError, TypeError):
                pass
            # Convert booleans/nulls
            if isinstance(val, str):
                lower = val.lower()
                if lower in ("true", "True"):
                    val = True
                elif lower in ("false", "False"):
                    val = False
                elif lower in ("null", "none", "NULL", "None"):
                    val = None
            args[key] = val
        return {"tool": tool_name, "args": args}

    # ── LFM format (flexible) ──
    # Try exact tokens first, then looser variants
    lfm_variants = [
        # Exact with square brackets
        (re.escape(_LFM_TOOL_CALL_START) + r"\[(.*?)\]" + re.escape(_LFM_TOOL_CALL_END)),
        # Without square brackets
        (re.escape(_LFM_TOOL_CALL_START) + r"(.*?)" + re.escape(_LFM_TOOL_CALL_END)),
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
        # Look for tool_name(args) anywhere in the text
        func_match = re.search(r"(\w+)\((.*?)\)", text, re.DOTALL)
        if func_match:
            parsed = _parse_func_call(func_match.group(0))
            if parsed:
                return parsed

    # ── JSON code blocks ──
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1))
            if "tool" in parsed:
                return _clean(parsed)
        except json.JSONDecodeError:
            pass

    # ── Raw JSON with tool key ──
    json_match = re.search(r'\{\s*"tool"\s*:', text)
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
                        parsed = json.loads(text[start:i + 1])
                        if "tool" in parsed:
                            return _clean(parsed)
                    except json.JSONDecodeError:
                        pass
                    break

    return None


class ReActAgent:
    """
    A prompt-based ReAct (Reasoning + Acting) agent.

    Works with ANY model — no native tool calling API required.
    The LLM is taught to output structured JSON when it wants to use a tool.

    Loop:
    1. Send user message with tool descriptions in system prompt
    2. If LLM responds with tool-call JSON → execute it, feed results back
    3. Repeat until LLM responds with plain text (final answer)
    4. Return the final response
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
        # Pre-build descriptions for each set
        self._system_tool_desc = _build_tool_descriptions(tools)
        self._browser_tool_desc = (
            _build_tool_descriptions(browser_tools) if browser_tools else ""
        )

    def _needs_browser_tools(self, text: str) -> bool:
        """Check if the user message implies web/browser usage."""
        return bool(self._WEB_KEYWORDS.search(text))

    def _get_tool_descriptions(self, include_browser: bool) -> str:
        """Return tool descriptions, optionally including browser tools."""
        if include_browser and self._browser_tool_desc:
            return self._system_tool_desc + "\n" + self._browser_tool_desc
        return self._system_tool_desc

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

    def _build_system_prompt(self, context: list[str] | None = None,
                              include_browser: bool = False) -> str:
        """Build the full system prompt with tools and context."""
        prompt = self._base_system_prompt

        # Add tool instructions
        tool_desc = self._get_tool_descriptions(include_browser)
        if tool_desc:
            prompt += TOOL_PROMPT_TEMPLATE.format(
                tool_descriptions=tool_desc,
            )

        # Add memory context
        if context:
            memory_block = "\n".join(f"- {c}" for c in context)
            prompt += f"\n\nRelevant memories about this user:\n{memory_block}"

        return prompt

    async def _capture_screenshot(self) -> list[str] | None:
        """Capture the screen and return as base64-encoded PNG."""
        try:
            import pyautogui
            img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            logger.debug("Screenshot captured: %d bytes base64", len(b64))
            return [b64]
        except Exception as e:
            logger.warning("Screenshot capture failed: %s", e)
            return None

    def _needs_vision(self, text: str) -> bool:
        """Check if the user message implies visual context is needed."""
        return bool(_VISION_KEYWORDS.search(text))

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: list[str] | None = None,
    ) -> str:
        """
        Process a user message through the ReAct loop.

        Args:
            user_message: The user's input text
            conversation_history: Prior messages for context
            context: Relevant memories from Mem0

        Returns:
            The agent's final text response
        """
        # Build messages
        messages: list[dict[str, Any]] = []

        # Detect if browser tools are needed
        include_browser = self._needs_browser_tools(user_message)
        if include_browser:
            logger.debug("Browser tools activated for this request")

        # Detect if vision is needed
        images: list[str] | None = None
        if self._needs_vision(user_message):
            logger.info("Vision triggered for: %s", user_message[:80])
            images = await self._capture_screenshot()
            if images:
                logger.info("Screenshot attached to LLM request")

        # System prompt with tools + context
        system_prompt = self._build_system_prompt(context, include_browser=include_browser)
        if images:
            system_prompt += (
                "\n\nVISION MODE: A screenshot is attached to the user's message. "
                "Describe or analyze what you see in the image directly. "
                "Do NOT call tools to capture or analyze the image — it is already provided."
            )
        messages.append({"role": "system", "content": system_prompt})

        # Conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Current user message
        if images:
            # Tell the model to analyze the attached image directly
            messages.append({
                "role": "user",
                "content": f"{user_message}\n\n[I've attached a screenshot. Answer based on what you see in the image.]",
            })
        else:
            messages.append({"role": "user", "content": user_message})

        # ReAct loop
        self._last_tool_call = None  # reset dedup tracker per run
        self._seen_tool_calls: set[tuple[str, str]] = set()  # ALL calls made this run
        tool_was_called = False  # track if any tool was called this run
        for iteration in range(MAX_ITERATIONS):
            logger.debug("ReAct iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = await self.llm.chat(messages=messages, images=images if iteration == 0 else None)
            content = response.get("message", {}).get("content", "")
            thinking = response.get("message", {}).get("thinking", "")
            logger.info("🧠 LLM raw (iter %d): %s", iteration + 1, content[:300])
            if thinking:
                logger.debug("🧠 LLM thinking (iter %d): %s", iteration + 1, thinking[:300])

            if not content:
                logger.warning("⚠️ Empty LLM response (iter %d). Prompt had %d chars, %d messages. Thinking length: %d chars",
                               iteration + 1, len(self._build_system_prompt(context, include_browser=include_browser)), len(messages), len(thinking))
                return "I couldn't generate a response."

            # Strip <think>...</think> tags for parsing (but keep for context)
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Try to parse a tool call from the response
            tool_call = _parse_tool_call(content) or _parse_tool_call(cleaned)

            if tool_call is None:
                # No tool call — but check if model is CLAIMING it did something
                # without actually calling a tool (hallucinated action)
                if not tool_was_called and _ACTION_CLAIM_PATTERNS.search(cleaned):
                    logger.warning("⚠️ Model claimed action without tool call, forcing retry")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You said you did something but you did NOT call any tool. "
                            "You MUST use a tool to perform actions. Reply with ONLY the "
                            "JSON tool call to actually do what the user asked."
                        ),
                    })
                    continue

                # Genuine final answer
                final = cleaned if cleaned else content
                return final

            # Execute the tool
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # Prevent repeated tool calls (same tool+args seen ANY time this run)
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_key in self._seen_tool_calls:
                logger.warning("⚠️ Skipping repeated tool call: %s (already executed this run)", tool_name)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"You already called '{tool_name}' with these exact arguments earlier. "
                        f"Do NOT repeat it. Either use a DIFFERENT tool to continue the task, "
                        f"or give your final text response to the user now."
                    ),
                })
                continue
            self._seen_tool_calls.add(call_key)

            logger.info("🔧 Agent calling tool: %s(%s)", tool_name, tool_args)
            tool_was_called = True

            # Add the assistant's tool-calling message
            messages.append({"role": "assistant", "content": content})

            try:
                result = await self._execute_tool(tool_name, **tool_args)
                result_str = str(result)
            except ValueError as e:
                result_str = f"Error: Unknown tool '{tool_name}'. Available: {self.tools.list_tools()}"
                logger.warning("Unknown tool: %s", tool_name)
            except Exception as e:
                result_str = f"Error executing {tool_name}: {e}"
                logger.error("Tool '%s' failed: %s", tool_name, e)

            # Feed tool result back — allow further tool calls if multi-step
            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned:\n{result_str}\n\n"
                    f"If the user's request is fully completed, give a brief final response. "
                    f"If more steps are needed, call the next tool."
                ),
            })

        # Exhausted iterations — give a summary of what was done
        logger.warning("ReAct loop exhausted %d iterations", MAX_ITERATIONS)
        if tool_was_called:
            return "I've completed what I could. Let me know if you need anything else."
        return content if content else "I ran into too many steps. Please try again."

    async def run_stream(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Same as run(), but streams the FINAL answer token-by-token.

        Tool-call iterations are done internally (non-streamed).
        Once the agent is ready to give its final answer, the last
        iteration is streamed via LLM chat_stream().

        Yields individual tokens as they arrive.
        """
        # Build messages (same as run())
        messages: list[dict[str, Any]] = []
        include_browser = self._needs_browser_tools(user_message)
        if include_browser:
            logger.debug("🌐 Browser tools activated for this request (stream)")
        system_prompt = self._build_system_prompt(context, include_browser=include_browser)
        messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        self._last_tool_call = None
        self._seen_tool_calls: set[tuple[str, str]] = set()
        tool_was_called = False

        for iteration in range(MAX_ITERATIONS):
            logger.debug("ReAct-stream iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            # Non-final iterations: use regular (non-streamed) chat
            response = await self.llm.chat(messages=messages)
            content = response.get("message", {}).get("content", "")
            thinking = response.get("message", {}).get("thinking", "")
            logger.info("🧠 LLM raw (iter %d): %s", iteration + 1, content[:300])
            if thinking:
                logger.debug("🧠 LLM thinking (iter %d): %s", iteration + 1, thinking[:300])

            if not content:
                logger.warning("⚠️ Empty LLM response (stream iter %d). Prompt had %d messages. Thinking length: %d chars",
                               iteration + 1, len(messages), len(thinking))
                yield "I couldn't generate a response."
                return

            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            tool_call = _parse_tool_call(content) or _parse_tool_call(cleaned)

            if tool_call is None:
                # Check hallucination
                if not tool_was_called and _ACTION_CLAIM_PATTERNS.search(cleaned):
                    logger.warning("⚠️ Model claimed action without tool call, forcing retry")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You said you did something but you did NOT call any tool. "
                            "You MUST use a tool to perform actions. Reply with ONLY the "
                            "JSON tool call to actually do what the user asked."
                        ),
                    })
                    continue

                # This IS the final answer — but we already have it from the
                # non-streamed call. Just yield it. (Streaming the final
                # iteration adds complexity for ~100ms gain; we can refine later.)
                final = cleaned if cleaned else content
                yield final
                return

            # Execute tool (same as run())
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_key in self._seen_tool_calls:
                logger.warning("⚠️ Skipping repeated tool call: %s", tool_name)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"You already called '{tool_name}' with these exact arguments. "
                        f"Do NOT repeat it. Give your final text response now."
                    ),
                })
                continue
            self._seen_tool_calls.add(call_key)

            logger.info("🔧 Agent calling tool: %s(%s)", tool_name, tool_args)
            tool_was_called = True
            messages.append({"role": "assistant", "content": content})

            try:
                result = await self._execute_tool(tool_name, **tool_args)
                result_str = str(result)
            except ValueError:
                result_str = f"Error: Unknown tool '{tool_name}'."
            except Exception as e:
                result_str = f"Error executing {tool_name}: {e}"

            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned:\\n{result_str}\\n\\n"
                    f"If the user's request is fully completed, give a brief final response. "
                    f"If more steps are needed, call the next tool."
                ),
            })

        # Exhausted iterations
        logger.warning("ReAct-stream loop exhausted %d iterations", MAX_ITERATIONS)
        if tool_was_called:
            yield "I've completed what I could. Let me know if you need anything else."
        else:
            yield content if content else "I ran into too many steps. Please try again."


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
        tool_was_called = False
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

            tool_was_called = True
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

        import base64, io
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
