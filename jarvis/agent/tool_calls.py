"""
Tool-call text handling for the ReAct agent
===========================================
Pure helpers with no I/O: parse tool calls out of model text (LFM markers,
JSON blocks, Python-style calls), normalize native Ollama tool_calls, and
build the compact tool listing used by the text-parsing fallback prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.agent.tool_registry import ToolDefinition

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


# Fallback tool-call format instructions, only used when native tool
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
- Be concise: responses will be spoken aloud
"""


# Phrases that indicate the model is claiming to have done an action
ACTION_CLAIM_PATTERNS = re.compile(
    r"\b(has opened|has been opened|i opened|i searched|i navigated|i launched|i started|"
    r"i(?:'ve| have) opened|i(?:'ve| have) searched|i(?:'ve| have) navigated|"
    r"i(?:'ve| have) launched|i(?:'ve| have) started|"
    r"has launched|opened for you|searched for|navigated to|"
    r"calling the .+ tool|has been saved|i(?:'ve| have) saved|i(?:'ve| have) typed|"
    r"calculator has|browser has|chrome has|youtube has|google has|"
    # Files and folders: "I read the file", "the file was created".
    r"i(?:'ve| have)? (?:read|created|wrote|written|deleted|removed|moved|copied|renamed|listed) "
    r"(?:the|that|your|a|an) (?:file|folder|directory|document)|"
    r"(?:file|folder|directory) (?:has been|was|is now) (?:created|written|deleted|removed|moved|"
    r"copied|renamed))\b",
    re.IGNORECASE,
)


def build_tool_descriptions(tools: Iterable[ToolDefinition]) -> str:
    """Build compact tool descriptions to minimize prompt tokens."""
    lines = []
    for tool in tools:
        params = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        parts = []
        for k, v in params.items():
            req = "*" if k in required else ""
            parts.append(f"{k}{req}:{v.get('type','any')}")
        lines.append(f"- {tool.name}({', '.join(parts)}) - {tool.description}")
    return "\n".join(lines)


# LFM special tokens for tool calling
_LFM_TOOL_CALL_START = "<|tool_call_start|>"
_LFM_TOOL_CALL_END = "<|tool_call_end|>"


def parse_func_call(call_text: str) -> dict[str, Any] | None:
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


def parse_tool_call(text: str) -> dict[str, Any] | None:
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
            parsed = parse_func_call(call_text)
            if parsed:
                return parsed

    # ── Loose heuristic: any func(args) near tool call markers ──
    if "tool_call" in text.lower():
        func_match = re.search(r"(\w+)\((.*?)\)", text, re.DOTALL)
        if func_match:
            parsed = parse_func_call(func_match.group(0))
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


def normalize_native_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Ollama native tool_calls -> [{tool, args}, ...]."""
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

