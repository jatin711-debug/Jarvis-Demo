"""
JARVIS ReAct Agent
===================
Prompt-based Reasoning + Acting loop that works with ANY LLM model.
Uses structured JSON output parsing instead of native tool calling API.
"""

from __future__ import annotations

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

# System prompt addition that teaches the LLM how to use tools
# Supports both JSON format (Ollama) and LFM format (HuggingFace)
TOOL_PROMPT_TEMPLATE = """
You have tools. To call one, use EITHER format:

# LFM/HuggingFace format (preferred):
<|tool_call_start|>[tool_name(arg="value", arg2=123)]<|tool_call_end|>

# JSON format:
```json
{{"tool": "<name>", "args": {{...}}}}
```

If no tool is needed, reply with plain text.

Tools:
{tool_descriptions}

RULES:
- ONE tool call per response, no extra text around the JSON
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
    r"has launched|is now open|opened for you|searched for|navigated to|"
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
    Try to extract a tool call JSON from the LLM response.
    Handles various formats: raw JSON, markdown code blocks, LFM format.

    LFM format: <|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>
    """
    def _clean(parsed: dict) -> dict:
        # Strip trailing () from tool name (LLM sometimes copies from description)
        parsed["tool"] = re.sub(r"\(.*\)$", "", parsed["tool"]).strip()
        return parsed

    # Try LFM format first: <|tool_call_start|>[tool_name(...) ]<|tool_call_end|>
    if _LFM_TOOL_CALL_START in text:
        pattern = re.escape(_LFM_TOOL_CALL_START) + r"\[(.*?)\]" + re.escape(_LFM_TOOL_CALL_END)
        lfm_match = re.search(pattern, text, re.DOTALL)
        if lfm_match:
            call_text = lfm_match.group(1).strip()
            # Parse Python-style call: tool_name(arg="value", arg2=123)
            func_match = re.match(r"(\w+)\((.*)\)", call_text)
            if func_match:
                tool_name = func_match.group(1)
                args_str = func_match.group(2)
                # Parse arguments
                args = {}
                arg_pattern = r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\d+\.?\d*))'
                for m in re.finditer(arg_pattern, args_str):
                    key = m.group(1)
                    val = m.group(2) or m.group(3) or m.group(4)
                    # Convert number strings to actual numbers
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except (ValueError, TypeError):
                        pass
                    args[key] = val
                return {"tool": tool_name, "args": args}

    # Try to find JSON in code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1))
            if "tool" in parsed:
                return _clean(parsed)
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON with tool key
    json_match = re.search(r'\{\s*"tool"\s*:', text)
    if json_match:
        # Find the matching closing brace
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
            logger.debug("🌐 Browser tools activated for this request")

        # System prompt with tools + context
        system_prompt = self._build_system_prompt(context, include_browser=include_browser)
        messages.append({"role": "system", "content": system_prompt})

        # Conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        # ReAct loop
        self._last_tool_call = None  # reset dedup tracker per run
        self._seen_tool_calls: set[tuple[str, str]] = set()  # ALL calls made this run
        tool_was_called = False  # track if any tool was called this run
        for iteration in range(MAX_ITERATIONS):
            logger.debug("ReAct iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = await self.llm.chat(messages=messages)
            content = response.get("message", {}).get("content", "")
            logger.info("🧠 LLM raw (iter %d): %s", iteration + 1, content[:300])

            if not content:
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
            logger.info("🧠 LLM raw (iter %d): %s", iteration + 1, content[:300])

            if not content:
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
