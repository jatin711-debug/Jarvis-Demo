"""Unit tests for browser/vision intent routing in the ReAct agent."""

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry


class _FakeLLM:
    provider = "ollama"


def _agent() -> ReActAgent:
    return ReActAgent(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        system_prompt="test",
        browser_tools=ToolRegistry(),
    )


class TestBrowserRouting:
    def test_web_intents_activate_browser_tools(self):
        agent = _agent()
        for text in (
            "search youtube for cats",
            "google the weather in Paris",
            "browse to github.com",
            "look up the news online",
            "open the reddit site",
        ):
            assert agent._needs_browser_tools(text) is True, text

    def test_local_commands_do_not(self):
        agent = _agent()
        for text in (
            "what time is it",
            "open notepad",
            "list the files in my downloads",
            "set volume to 30",
            "take a screenshot",
        ):
            assert agent._needs_browser_tools(text) is False, text


class TestVisionRouting:
    def test_visual_intents_trigger_vision(self):
        agent = _agent()
        for text in (
            "what's on my screen",
            "take a screenshot",
            "click the OK button",
            "where is the taskbar",
            "can you see this error message",
        ):
            assert agent._needs_vision(text) is True, text

    def test_non_visual_commands_do_not(self):
        agent = _agent()
        for text in (
            "what time is it",
            "open notepad",
            "search youtube for music",
            "set volume to 50",
        ):
            assert agent._needs_vision(text) is False, text
