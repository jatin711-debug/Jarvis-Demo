"""Desktop contract tests run without native dependencies or GUI access."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.agent.react_agent import MAX_DESKTOP_ITERATIONS, ReActAgent
from jarvis.agent.routing import Route, Surface
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.desktop.cua import CuaDesktop, describe_observation, register_desktop_tools
from jarvis.utils.config import DesktopConfig


class Driver:
    """A fixture window with one editable control."""

    def __init__(self):
        self.calls = []
        self.generation = 0
        self.degraded = False
        self.minimized = False
        self.shutdown = AsyncMock()

    def current_elements(self):
        return [{"element_index": 0, "element_token": "token", "enabled": True,
                 "role": "edit", "label": "Text", "value": ""}]

    def dispatch(self, name, args):
        """Hook for a driver whose window reacts to actions."""

    async def call_tool(self, name, arguments):
        args = json.loads(arguments)
        self.calls.append((name, args))
        if name == "list_windows":
            data = {"windows": [{"pid": 1, "window_id": 2, "title": "Fixture",
                                 "minimized": self.minimized}]}
        elif name == "get_window_state":
            self.generation += 1
            data = {"pid": 1, "window_id": 2, "snapshot_id": f"s{self.generation}",
                    "degraded": self.degraded, "elements": self.current_elements()}
        else:
            self.dispatch(name, args)
            data = {"dispatched": True}
        return SimpleNamespace(is_error=False, structured_json=json.dumps(data), text="")


class CalculatorDriver(Driver):
    """Digit keys append to a display; Multiply by changes nothing visible; Equals opens a dialog."""

    DIGITS = {"b1": "7", "b2": "8"}

    def __init__(self):
        super().__init__()
        self.display = "0"
        self.dialog = False

    def current_elements(self):
        if self.dialog:
            return [{"element_index": 0, "element_token": "ok", "enabled": True,
                     "role": "Button", "label": "OK", "actions": ["invoke"]}]
        elements = [{"element_index": 0, "element_token": "d", "enabled": True, "role": "Text",
                     "label": f"Display is {self.display}", "actions": ["text"]}]
        for index, label in enumerate(["Seven", "Eight", "Multiply by", "Equals"], 1):
            elements.append({"element_index": index, "element_token": f"b{index}", "enabled": True,
                             "role": "Button", "label": label, "actions": ["invoke"]})
        elements.append({"element_index": 5, "element_token": "x", "enabled": False,
                         "role": "Button", "label": "Clear entry", "actions": ["invoke"]})
        return elements

    def dispatch(self, name, args):
        if name == "click" and args["element_token"] in self.DIGITS:
            self.display = ("" if self.display == "0" else self.display) + self.DIGITS[args["element_token"]]
        if name == "click" and args["element_token"] == "b4":
            self.dialog = True


@pytest.fixture
def desktop():
    return CuaDesktop(DesktopConfig(), driver=Driver())


@pytest.fixture
def calculator():
    return CuaDesktop(DesktopConfig(), driver=CalculatorDriver())


@pytest.fixture
def tools(desktop):
    registry = ToolRegistry()
    register_desktop_tools(registry, desktop)
    return registry


@pytest.fixture
def calc_tools(calculator):
    registry = ToolRegistry()
    register_desktop_tools(registry, calculator)
    return registry


async def observed(desktop):
    await desktop.windows()
    return await desktop.observe(1, 2)


async def refused(registry, name, **args):
    """The message of the error a tool raises; the agent turns it into text for the model."""
    with pytest.raises(ValueError) as caught:
        await registry.execute(name, **args)
    return str(caught.value)


def clicks(desktop):
    return [args["element_token"] for name, args in desktop._driver.calls if name == "click"]


async def test_requires_discovered_window(desktop):
    with pytest.raises(ValueError, match="desktop_windows"):
        await desktop.observe(1, 2)
    assert not desktop._driver.calls


async def test_windows_rejects_extra_ids_with_corrective_schema(desktop, tools):
    error = await tools.execute("desktop_windows", pid=1, window_id=2)
    assert "Unknown parameter" in error
    assert "Required parameter names: []" in error
    assert "must be removed" in error
    assert not desktop._driver.calls
    assert 'pid=1 window_id=2 "Fixture"' in await tools.execute("desktop_windows")


async def test_observe_does_not_require_snapshot_or_element(desktop, tools):
    await tools.execute("desktop_windows")
    error = await tools.execute("desktop_observe", pid=1, window_id=2,
                                snapshot_id="fake", element_index=0)
    assert "Required parameter names: ['pid', 'window_id']" in error
    assert '0 edit "Text"' in await tools.execute("desktop_observe", pid=1, window_id=2)


async def test_launch_returns_the_windows_so_the_model_does_not_ask_again(desktop, tools):
    text = await tools.execute("desktop_launch", name="Calculator")
    assert text.startswith("launched Calculator")
    assert 'pid=1 window_id=2 "Fixture"' in text
    assert [name for name, _ in desktop._driver.calls] == ["launch_app", "list_windows"]


async def test_action_uses_token_and_returns_fresh_evidence(desktop):
    state = await observed(desktop)
    result = await desktop.act("type_text", 1, 2, state["snapshot_id"], 0, text="hello")
    assert result["after"]["snapshot_id"] != state["snapshot_id"]
    name, args = desktop._driver.calls[-2]
    assert name == "type_text"
    assert args["element_token"] == "token"
    assert args["delivery_mode"] == "background"
    assert "element_token" not in json.dumps(result["after"])


@pytest.mark.parametrize("mode", ["wrong", "expired", "degraded", "rediscovered"])
async def test_rejects_invalid_observation(desktop, mode):
    desktop._driver.degraded = mode == "degraded"
    state = await observed(desktop)
    if mode == "expired":
        desktop._snapshots[(1, 2)] = desktop._snapshots[(1, 2)]._replace(at=0)
    if mode == "rediscovered":
        await desktop.windows()
    with pytest.raises(ValueError, match="stale"):
        await desktop.act("click", 1, 2, "wrong" if mode == "wrong" else state["snapshot_id"], 0)
    assert all(name != "click" for name, _ in desktop._driver.calls)


async def test_unknown_element_rejected(desktop):
    state = await observed(desktop)
    with pytest.raises(ValueError, match="Element"):
        await desktop.act("click", 1, 2, state["snapshot_id"], 99)


async def test_cancelled_native_call_disables_further_dispatch(desktop):
    desktop._driver.call_tool = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await desktop.windows()
    with pytest.raises(RuntimeError, match="interrupted"):
        await desktop.windows()
    assert desktop._driver.call_tool.await_count == 1


async def test_timeout_disables_further_dispatch(desktop):
    desktop._driver.call_tool = AsyncMock(side_effect=TimeoutError)
    with pytest.raises(TimeoutError):
        await desktop.windows()
    with pytest.raises(RuntimeError, match="interrupted"):
        await desktop.windows()


async def test_shutdown_is_idempotent(desktop):
    driver = desktop._driver
    await desktop.shutdown()
    await desktop.shutdown()
    driver.shutdown.assert_awaited_once()
    with pytest.raises(RuntimeError, match="not running"):
        await desktop.windows()


async def test_failed_post_observation_is_not_claimed_verified(desktop):
    state = await observed(desktop)
    desktop._observe = AsyncMock(side_effect=RuntimeError("window closed"))
    result = await desktop.act("click", 1, 2, state["snapshot_id"], 0)
    assert result["verified"] is False
    assert "do not repeat" in result["next"]


def test_six_tools_and_selective_routing(desktop, tools):
    assert len(tools) == 6
    assert tools.get("desktop_observe").repeatable
    assert tools.get("desktop_click").requires_confirmation
    agent = ReActAgent(llm=SimpleNamespace(provider="ollama"), tools=ToolRegistry(),
                       system_prompt="test", desktop_tools=tools)
    assert [t.name for t in agent._tools_for(Route(Surface.DESKTOP))] == tools.list_tools()
    assert agent._tools_for(Route(Surface.CHAT)) == []


def test_actions_name_a_control_by_label_not_by_snapshot_ids(tools):
    for name in ("desktop_click", "desktop_type", "desktop_key"):
        schema = tools.get(name).parameters
        assert {"pid", "window_id", "label", "index"} <= set(schema["properties"])
        assert "snapshot_id" not in schema["properties"]
        assert "label" not in schema["required"] and "index" not in schema["required"]
        assert tools.get(name).repeatable  # pressing 7 twice must not be refused as a repeat


async def test_old_style_snapshot_arguments_are_refused_without_acting(desktop, tools):
    await observed(desktop)
    error = await tools.execute("desktop_click", pid=1, window_id=2, snapshot_id="s1", element_index=0)
    assert "Unknown parameter" in error and "snapshot_id" in error
    assert clicks(desktop) == []


async def test_observation_is_compact_text(calculator):
    state = await observed(calculator)
    text = describe_observation(state)
    assert text.splitlines()[0] == 'window pid=1 window_id=2 ""'
    assert '1 Button "Seven"' in text and "(disabled)" in text
    assert "element_token" not in text and "snapshot" not in text
    assert len(text) < 60 * len(state["elements"])  # the JSON form was about 80 characters a control


async def test_long_values_are_clipped(desktop):
    desktop._driver.current_elements = lambda: [
        {"element_index": 0, "element_token": "t", "enabled": True, "role": "edit",
         "label": "Notes", "value": "x" * 5000}]
    text = describe_observation(await observed(desktop))
    assert len(text) < 400 and "…" in text


async def test_click_by_label_resolves_the_control_and_reports_what_changed(calculator, calc_tools):
    await calc_tools.execute("desktop_windows")
    await calc_tools.execute("desktop_observe", pid=1, window_id=2)
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, label="Seven")
    assert clicks(calculator) == ["b1"]
    assert reply.startswith('clicked 1 Button "Seven"')
    assert '0 Text "Display is 7"' in reply
    assert "(5 other controls unchanged)" in reply
    assert "Dispatch alone is not proof" in reply
    # The new snapshot is used for the next label without the model carrying ids around.
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, label="Eight")
    assert clicks(calculator) == ["b1", "b2"] and "Display is 78" in reply


async def test_label_match_ignores_case_and_spacing(calculator, calc_tools):
    await observed(calculator)
    await calc_tools.execute("desktop_click", pid=1, window_id=2, label="  multiply   BY ")
    assert clicks(calculator) == ["b3"]


async def test_an_action_without_visible_effect_says_so(calculator, calc_tools):
    await observed(calculator)
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, label="Multiply by")
    assert "the 6 listed controls did not change" in reply and "may still have worked" in reply


async def test_a_window_that_changed_completely_is_listed_again(calculator, calc_tools):
    await observed(calculator)
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, label="Equals")
    assert "the window changed; its controls now:" in reply and '0 Button "OK"' in reply


async def test_a_digit_key_can_be_named_by_digit_or_word(calculator, calc_tools):
    await observed(calculator)
    await calc_tools.execute("desktop_click", pid=1, window_id=2, label="7")
    await calc_tools.execute("desktop_click", pid=1, window_id=2, label="eight")
    assert clicks(calculator) == ["b1", "b2"]
    assert calculator._driver.display == "78"


async def test_a_digit_word_reaches_a_key_labeled_with_the_digit(desktop, tools):
    desktop._driver.current_elements = lambda: [
        {"element_index": 3, "element_token": "t3", "enabled": True, "role": "Button", "label": "3"}]
    await observed(desktop)
    await tools.execute("desktop_click", pid=1, window_id=2, label="Three")
    assert clicks(desktop) == ["t3"]


async def test_digit_aliases_do_not_match_other_labels(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Seventeen")
    assert 'No enabled control is labeled "Seventeen"' in error
    assert clicks(calculator) == []


async def test_unknown_label_lists_candidates_and_clicks_nothing(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Twenty")
    assert 'No enabled control is labeled "Twenty"' in error and '"Seven"' in error
    assert clicks(calculator) == []
    # The observation survives a bad label, so the corrected call needs no new desktop_observe.
    await calc_tools.execute("desktop_click", pid=1, window_id=2, label="Seven")
    assert clicks(calculator) == ["b1"]


async def test_a_partial_label_is_not_guessed(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Multiply")
    assert "Similar labels" in error and '"Multiply by"' in error
    assert clicks(calculator) == []


async def test_a_disabled_control_cannot_be_clicked(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Clear entry")
    assert "No enabled control" in error
    assert clicks(calculator) == []


async def test_shared_label_needs_an_index(desktop, tools):
    desktop._driver.current_elements = lambda: [
        {"element_index": i, "element_token": f"t{i}", "enabled": True, "role": "Button",
         "label": "Close", "actions": ["invoke"]} for i in (3, 4)]
    await observed(desktop)
    error = await refused(tools, "desktop_click", pid=1, window_id=2, label="Close")
    assert '2 controls are labeled "Close"' in error and "3 Button" in error and "4 Button" in error
    assert clicks(desktop) == []
    await tools.execute("desktop_click", pid=1, window_id=2, label="Close", index=4)
    assert clicks(desktop) == ["t4"]
    await observed(desktop)
    error = await refused(tools, "desktop_click", pid=1, window_id=2, label="Close", index=9)
    assert "No enabled control" in error


async def test_index_alone_reaches_an_unlabeled_control(desktop, tools):
    desktop._driver.current_elements = lambda: [
        {"element_index": 7, "element_token": "t7", "enabled": True, "role": "Button", "label": None}]
    await observed(desktop)
    assert "clicked 7 Button" in await tools.execute("desktop_click", pid=1, window_id=2, index=7)
    assert clicks(desktop) == ["t7"]


async def test_an_action_must_name_its_control(desktop, tools):
    await observed(desktop)
    error = await refused(tools, "desktop_click", pid=1, window_id=2)
    assert "label" in error and clicks(desktop) == []


async def test_digits_press_a_numbers_keys_in_order(calculator, calc_tools):
    await observed(calculator)
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, digits="788")
    assert clicks(calculator) == ["b1", "b2", "b2"]
    assert calculator._driver.display == "788"
    assert reply.startswith("pressed the keys 7 8 8") and '0 Text "Display is 788"' in reply


async def test_a_number_given_as_a_label_is_an_error_not_key_presses(calculator, calc_tools):
    # A model that reads "21 Button ..." and passes 21 as the label must be told, not obeyed.
    await observed(calculator)
    for label in ("788", "21"):
        error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label=label)
        assert f'No enabled control is labeled "{label}"' in error
    assert clicks(calculator) == []


async def test_digits_that_are_not_digits_are_refused(calculator, calc_tools):
    await observed(calculator)
    for bad in ("7+8", "seven", "1" * 21, "٣"):
        assert "digits must be" in await refused(calc_tools, "desktop_click", pid=1, window_id=2, digits=bad)
    assert clicks(calculator) == []


async def test_digits_cannot_be_combined_with_a_label(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Seven", digits="7")
    assert "either digits or a label" in error and clicks(calculator) == []


async def test_a_number_with_a_missing_key_says_how_far_it_got(calculator, calc_tools):
    await observed(calculator)
    error = await refused(calc_tools, "desktop_click", pid=1, window_id=2, digits="789")
    assert 'Pressed 78 of 789, then: No enabled control is labeled "9"' in error
    assert clicks(calculator) == ["b1", "b2"]


async def test_only_clicks_take_digits(calculator, calc_tools):
    await observed(calculator)
    error = await calc_tools.execute("desktop_type", pid=1, window_id=2, label="Text", text="x", digits="7")
    assert "Unknown parameter" in error and "digits" in error
    assert clicks(calculator) == []


async def test_a_number_stops_when_the_window_cannot_be_re_read(calculator, calc_tools):
    await observed(calculator)
    calculator._observe = AsyncMock(side_effect=RuntimeError("window closed"))
    reply = await calc_tools.execute("desktop_click", pid=1, window_id=2, digits="788")
    assert clicks(calculator) == ["b1"]  # it does not press the rest blind
    assert "outcome is unverified" in reply


async def test_a_keypad_gets_a_hint_where_the_model_decides(desktop):
    desktop._driver.current_elements = lambda: [
        {"element_index": i, "element_token": f"t{i}", "enabled": True, "role": "Button",
         "label": word} for i, word in enumerate(
            ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"])]
    assert 'digits="' in describe_observation(await observed(desktop))
    assert 'digits="' not in describe_observation(
        {"pid": 1, "window_id": 2, "title": "x", "elements": [
            {"element_index": 0, "role": "Button", "label": "Seven", "enabled": True}]})


async def test_a_minimized_window_is_flagged_for_the_model(desktop, tools):
    desktop._driver.minimized = True
    text = await tools.execute("desktop_windows")
    assert "minimized" in text and "ask the user to restore it" in text


async def test_label_actions_need_a_fresh_observation(calculator, calc_tools):
    assert "stale" in await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Seven")
    await observed(calculator)
    calculator._snapshots[(1, 2)] = calculator._snapshots[(1, 2)]._replace(at=0)
    assert "stale" in await refused(calc_tools, "desktop_click", pid=1, window_id=2, label="Seven")
    assert clicks(calculator) == []


async def test_type_and_key_by_label(desktop, tools):
    await observed(desktop)
    reply = await tools.execute("desktop_type", pid=1, window_id=2, label="Text", text="hello")
    assert reply.startswith('typed into 0 edit "Text"')
    await tools.execute("desktop_key", pid=1, window_id=2, index=0, key="return")
    sent = {name: args for name, args in desktop._driver.calls if name in ("type_text", "press_key")}
    assert sent["type_text"]["text"] == "hello" and sent["press_key"]["key"] == "return"
    assert sent["type_text"]["element_token"] == sent["press_key"]["element_token"] == "token"


async def test_unverified_outcome_is_reported_honestly(desktop, tools):
    await observed(desktop)
    desktop._observe = AsyncMock(side_effect=RuntimeError("window closed"))
    reply = await tools.execute("desktop_click", pid=1, window_id=2, label="Text")
    assert "outcome is unverified" in reply and "window closed" in reply


async def test_agent_can_press_the_same_key_twice(calculator, calc_tools):
    def call(name, args):
        return {"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}

    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(side_effect=[
        call("desktop_windows", {}),
        call("desktop_observe", {"pid": 1, "window_id": 2}),
        call("desktop_click", {"pid": 1, "window_id": 2, "label": "Seven"}),
        call("desktop_click", {"pid": 1, "window_id": 2, "label": "Seven"}),
        {"message": {"content": "The display shows 77."}},
    ]))
    agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="test", desktop_tools=calc_tools)
    assert "77" in await agent.run("Use the calculator window to enter 77")
    assert calculator._driver.display == "77"


async def test_agent_allows_repeated_observations(desktop, tools):
    def call(name, args):
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": name, "arguments": args}}]}}

    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(side_effect=[
        call("desktop_windows", {}),
        call("desktop_observe", {"pid": 1, "window_id": 2}),
        call("desktop_observe", {"pid": 1, "window_id": 2}),
        {"message": {"content": "The fixture has an editable text control."}},
    ]))
    agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="test", desktop_tools=tools)
    answer = await agent.run("Inspect the desktop window")
    assert "editable" in answer
    assert [name for name, _ in desktop._driver.calls].count("get_window_state") == 2
    offered = llm.chat.call_args_list[0].kwargs["tools"]
    assert len(offered) == 6
    assert all(t["function"]["name"].startswith("desktop_") for t in offered)


async def test_desktop_requests_get_more_steps_than_other_requests(desktop, tools):
    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(side_effect=[
        {"message": {"content": "", "tool_calls": [{"function": {"name": "desktop_windows", "arguments": {}}}]}}
        for _ in range(MAX_DESKTOP_ITERATIONS + 3)]))
    agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="test", desktop_tools=tools)
    answer = await agent.run("List the windows on my desktop")
    assert "step limit" in answer
    assert llm.chat.await_count == MAX_DESKTOP_ITERATIONS
