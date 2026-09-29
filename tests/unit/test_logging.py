"""Console and file log lines are plain, aligned, and tagged with the turn."""

import logging

import pytest

from jarvis.agent.react_agent import _brief_args
from jarvis.utils.logger import (
    ConsoleFormatter,
    FileFormatter,
    _TurnFilter,
    begin_turn,
    current_turn,
    end_turn,
    setup_logging,
)


def make_record(message="hello", level=logging.INFO, name="jarvis.agent.react_agent"):
    record = logging.LogRecord(name, level, __file__, 1, message, None, None)
    _TurnFilter().filter(record)
    return record


@pytest.fixture(autouse=True)
def no_active_turn():
    end_turn()
    yield
    end_turn()


@pytest.fixture(autouse=True)
def restore_root_logger():
    """setup_logging replaces the root handlers; put them back so later tests are unaffected."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_console_line_has_level_component_and_message():
    line = ConsoleFormatter(color=False).format(make_record("route surface=chat"))
    assert "INFO " in line
    assert " agent " in line
    assert line.endswith("route surface=chat")


def test_console_line_is_free_of_ansi_when_color_is_off():
    line = ConsoleFormatter(color=False).format(make_record())
    assert "\x1b" not in line


def test_console_line_colors_only_when_asked():
    line = ConsoleFormatter(color=True).format(make_record(level=logging.WARNING))
    assert "\x1b[33m" in line


def test_turn_tag_appears_on_records_inside_a_turn():
    turn = begin_turn()
    assert current_turn() == turn
    line = ConsoleFormatter(color=False).format(make_record())
    assert f"t{turn}" in line
    assert f"| t{turn} |" in FileFormatter().format(make_record())
    end_turn()
    assert current_turn() is None
    assert "| - |" in FileFormatter().format(make_record())


def test_turn_numbers_increase():
    first = begin_turn()
    assert begin_turn() == first + 1


def test_file_line_keeps_the_full_logger_name():
    line = FileFormatter().format(make_record(name="jarvis.core.orchestrator"))
    assert "| jarvis.core.orchestrator |" in line


def test_exceptions_are_appended_to_the_console_line():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record()
        record.exc_info = sys.exc_info()
    text = ConsoleFormatter(color=False).format(record)
    assert "ValueError: boom" in text


def test_setup_logging_writes_debug_to_file_but_info_to_console(tmp_path, capsys):
    setup_logging(level="INFO", log_dir=str(tmp_path))
    log = logging.getLogger("jarvis.test")
    log.debug("debug-only detail")
    log.info("visible event")
    for handler in logging.getLogger().handlers:
        handler.flush()
    err = capsys.readouterr().err
    assert "visible event" in err
    assert "debug-only detail" not in err
    written = (tmp_path / "jarvis.log").read_text(encoding="utf-8")
    assert "debug-only detail" in written and "visible event" in written


def test_tool_arguments_are_redacted_and_shortened_in_logs():
    shown = _brief_args({"password": "hunter2", "api_key": "abc", "text": "x" * 500, "n": 3})
    assert "hunter2" not in shown and "abc" not in shown
    assert "password=<redacted>" in shown and "api_key=<redacted>" in shown
    assert len(shown) < 200
    assert "n=3" in shown
