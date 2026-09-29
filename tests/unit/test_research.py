"""The experimental research loop searches, reads, then summarizes."""

from unittest.mock import AsyncMock

from jarvis.agent.research import run_research


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, messages=None, **_):
        self.prompts.append(messages)
        return {"message": {"content": self.replies.pop(0)}}


async def test_search_then_summarize():
    llm = ScriptedLLM(['<|tool_call_start|>[web_search(query="cats")]<|tool_call_end|>',
                       "Cats are mammals."])
    execute = AsyncMock(return_value="result page")
    assert await run_research(llm, execute, "cats") == "Cats are mammals."
    execute.assert_awaited_once_with("web_search", query="cats")


async def test_identical_call_is_not_repeated():
    call = '<|tool_call_start|>[open_url(url="http://a.test")]<|tool_call_end|>'
    llm = ScriptedLLM([call, call, "Summary."])
    execute = AsyncMock(return_value="page")
    assert await run_research(llm, execute, "topic") == "Summary."
    assert execute.await_count == 1


async def test_tool_errors_are_reported_to_the_model_not_raised():
    llm = ScriptedLLM(['<|tool_call_start|>[web_search(query="x")]<|tool_call_end|>', "Nothing found."])
    execute = AsyncMock(side_effect=RuntimeError("offline"))
    assert await run_research(llm, execute, "x") == "Nothing found."
    assert "Error: offline" in llm.prompts[1][-1]["content"]


async def test_empty_model_reply_ends_research():
    assert await run_research(ScriptedLLM([""]), AsyncMock(), "x") == "I couldn't complete the research."
