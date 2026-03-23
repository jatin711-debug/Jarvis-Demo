"""
JARVIS ReAct Agent
===================
Prompt-based Reasoning + Acting loop that works with ANY LLM model.
Uses structured JSON output parsing instead of native tool calling API.
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

if TYPE_CHECKING:
    from jarvis.utils.config import LLMConfig

logger = get_logger(__name__)

# Maximum tool-calling iterations to prevent infinite loops
MAX_ITERATIONS = 5

# System prompt addition that teaches the LLM how to use tools
TOOL_PROMPT_TEMPLATE = """
You have access to the following tools. To use a tool, respond with EXACTLY this JSON format (and nothing else before it):

```json
{{"tool": "<tool_name>", "args": {{<arguments>}}}}
```

When you want to give a final answer to the user (no tool needed), just respond normally with text.

Available tools:
{tool_descriptions}

IMPORTANT RULES:
- Use ONLY ONE tool call per response
- After receiving a tool result, either use another tool or give your final answer
- For potentially dangerous operations, explain what you'll do first
- Always respond in a helpful, concise manner
"""


def _build_tool_descriptions(registry: ToolRegistry) -> str:
    """Build human-readable tool descriptions for the system prompt."""
    lines = []
    for name in registry.list_tools():
        tool = registry.get(name)
        if tool is None:
            continue
        params = tool.parameters.get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'any')} - {v.get('description', '')}"
            for k, v in params.items()
        )
        required = tool.parameters.get("required", [])
        lines.append(
            f"- **{name}**: {tool.description}\n"
            f"  Parameters: {param_str or '(none)'}\n"
            f"  Required: {', '.join(required) if required else '(none)'}"
        )
    return "\n".join(lines)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """
    Try to extract a tool call JSON from the LLM response.
    Handles various formats: raw JSON, markdown code blocks, etc.
    """
    # Try to find JSON in code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1))
            if "tool" in parsed:
                return parsed
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
                            return parsed
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

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self._base_system_prompt = system_prompt
        self._tool_descriptions = _build_tool_descriptions(tools)

    def _build_system_prompt(self, context: list[str] | None = None) -> str:
        """Build the full system prompt with tools and context."""
        prompt = self._base_system_prompt

        # Add tool instructions
        if len(self.tools) > 0:
            prompt += TOOL_PROMPT_TEMPLATE.format(
                tool_descriptions=self._tool_descriptions,
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

        # System prompt with tools + context
        system_prompt = self._build_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})

        # Conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        # ReAct loop
        for iteration in range(MAX_ITERATIONS):
            logger.debug("ReAct iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = await self.llm.chat(messages=messages)
            content = response.get("message", {}).get("content", "")

            if not content:
                return "I couldn't generate a response."

            # Try to parse a tool call from the response
            tool_call = _parse_tool_call(content)

            if tool_call is None:
                # No tool call found — this is the final answer
                # Strip any thinking tags if present
                final = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return final if final else content

            # Execute the tool
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            logger.info("🔧 Agent calling tool: %s(%s)", tool_name, tool_args)

            # Add the assistant's tool-calling message
            messages.append({"role": "assistant", "content": content})

            try:
                result = await self.tools.execute(tool_name, **tool_args)
                result_str = str(result)
            except ValueError as e:
                result_str = f"Error: Unknown tool '{tool_name}'. Available: {self.tools.list_tools()}"
                logger.warning("Unknown tool: %s", tool_name)
            except Exception as e:
                result_str = f"Error executing {tool_name}: {e}"
                logger.error("Tool '%s' failed: %s", tool_name, e)

            # Feed tool result back
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned:\n{result_str}\n\nNow give your final response to the user based on this result.",
            })

        # Exhausted iterations
        logger.warning("ReAct loop exhausted %d iterations", MAX_ITERATIONS)
        return content if content else "I ran into too many steps. Please try again."
