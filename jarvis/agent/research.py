"""Experimental autonomous web research (not wired into the voice loop)."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from jarvis.agent.tool_calls import parse_tool_call, strip_thinking
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_ITERATIONS = 8
_RESULT_CHARS = 4000

_SYSTEM_PROMPT = (
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


async def run_research(
    llm: Any,
    execute_tool: Callable[..., Awaitable[Any]],
    query: str,
    max_sources: int = 5,
) -> str:
    """
    Investigate ``query`` with web tools and return a short cited summary.

    Search, read promising pages, then synthesize; stops after ``max_sources``
    pages and never repeats an identical tool call.
    """
    logger.info("research started query_chars=%d max_sources=%d", len(query), max_sources)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Research this topic and give me a summary: {query}"},
    ]
    seen_calls: set[tuple[str, str]] = set()
    sources_consulted = 0
    content = ""

    for iteration in range(_MAX_ITERATIONS):
        response = await llm.chat(messages=messages)
        content = response.get("message", {}).get("content", "")
        logger.debug("research iteration=%d chars=%d", iteration + 1, len(content))

        if not content:
            return "I couldn't complete the research."

        cleaned = strip_thinking(content)
        tool_call = parse_tool_call(content) or parse_tool_call(cleaned)
        if tool_call is None:
            return cleaned or content

        tool_name = tool_call.get("tool", "")
        tool_args = tool_call.get("args", {})

        if tool_name in ("open_url", "read_pdf"):
            sources_consulted += 1
            if sources_consulted > max_sources:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (f"You've consulted {max_sources} sources. "
                                "Please synthesize what you've learned into a final summary now."),
                })
                continue

        call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
        if call_key in seen_calls:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": ("You already consulted that source. "
                            "Please synthesize your findings into a final summary."),
            })
            continue
        seen_calls.add(call_key)

        messages.append({"role": "assistant", "content": content})
        try:
            result = str(await execute_tool(tool_name, **tool_args))[:_RESULT_CHARS]
        except Exception as e:
            result = f"Error: {e}"

        messages.append({
            "role": "user",
            "content": (f"Tool '{tool_name}' result:\n{result}\n\n"
                        "Continue researching or provide your final summary."),
        })

    return "Research complete. Here is what I found:\n\n" + content
