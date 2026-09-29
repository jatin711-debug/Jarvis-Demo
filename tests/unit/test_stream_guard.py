"""Streamed text that could be tool-call markup is held back, never spoken."""

from jarvis.agent.stream_guard import StreamGuard


def feed_all(tokens):
    guard = StreamGuard()
    spoken = "".join(guard.feed(t) for t in tokens)
    return guard, spoken


def test_plain_text_passes_straight_through():
    guard, spoken = feed_all(["Hello ", "there, ", "friend."])
    assert spoken == "Hello there, friend."
    assert guard.held == ""
    assert guard.text == "Hello there, friend."


def test_marker_split_across_tokens_is_held_back():
    guard, spoken = feed_all(["Sure. ", "<", "|tool_call_start|>", '[open_application(app_name="x")]'])
    assert spoken == "Sure. "
    assert guard.held.startswith("<|tool_call_start|>")


def test_text_before_a_marker_is_emitted_once():
    guard, spoken = feed_all(["Opening it now ```json", '{"tool": "x"}```'])
    assert spoken == "Opening it now "
    assert guard.held.startswith("```json")


def test_everything_after_a_marker_stays_held():
    guard, spoken = feed_all(['{"tool": "a"}', " and then more text"])
    assert spoken == ""
    assert guard.text == '{"tool": "a"} and then more text'


def test_partial_marker_that_never_completes_is_released_later():
    guard = StreamGuard()
    assert guard.feed("Use the <") == "Use the "
    assert guard.held == "<"
    assert guard.feed(" symbol") == "< symbol"
    assert guard.held == ""
    assert guard.text == "Use the < symbol"
