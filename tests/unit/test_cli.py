"""Entry point: config errors, exit codes, text mode, and clean shutdown."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import jarvis.__main__ as cli


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"system:\n  data_dir: '{tmp_path.as_posix()}'\n  log_dir: '{(tmp_path / 'logs').as_posix()}'\n",
        encoding="utf-8")
    return path


def install_fake(monkeypatch, **overrides):
    fake = SimpleNamespace(startup=AsyncMock(), run=AsyncMock(), ask=AsyncMock(return_value="Hello!"),
                           shutdown=AsyncMock())
    for name, value in overrides.items():
        setattr(fake, name, value)
    monkeypatch.setattr(cli, "Orchestrator", lambda config: fake)
    return fake


async def test_invalid_config_exits_2_before_anything_starts(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  temperature: not-a-number\n", encoding="utf-8")
    fake = install_fake(monkeypatch)
    assert await cli.main(["--config", str(bad)]) == cli.EXIT_BAD_CONFIG
    assert "cannot load configuration" in capsys.readouterr().err
    fake.startup.assert_not_called()


async def test_missing_config_file_exits_2(tmp_path, capsys):
    assert await cli.main(["--config", str(tmp_path / "missing.yaml")]) == cli.EXIT_BAD_CONFIG
    assert "cannot load configuration" in capsys.readouterr().err


def test_empty_text_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--text", "   "])
    assert exc.value.code == 2


async def test_text_mode_prints_only_the_reply_to_stdout(config_file, monkeypatch, capsys):
    fake = install_fake(monkeypatch)
    assert await cli.main(["--config", str(config_file), "--text", "hi there"]) == cli.EXIT_OK
    out = capsys.readouterr()
    assert out.out == "Hello!\n"
    fake.startup.assert_awaited_once_with(voice_enabled=False)
    fake.ask.assert_awaited_once_with("hi there")
    fake.shutdown.assert_awaited_once()


async def test_voice_mode_runs_the_loop_with_voice_enabled(config_file, monkeypatch):
    fake = install_fake(monkeypatch)
    assert await cli.main(["--config", str(config_file)]) == cli.EXIT_OK
    fake.startup.assert_awaited_once_with(voice_enabled=True)
    fake.run.assert_awaited_once()
    fake.shutdown.assert_awaited_once()


async def test_preflight_failure_exits_1_and_still_shuts_down(config_file, monkeypatch, capsys):
    fake = install_fake(monkeypatch, startup=AsyncMock(side_effect=RuntimeError("Ollama is not running")))
    assert await cli.main(["--config", str(config_file), "--text", "hi"]) == cli.EXIT_STARTUP_FAILED
    assert "startup failed: Ollama is not running" in capsys.readouterr().err
    fake.ask.assert_not_called()
    fake.shutdown.assert_awaited_once()


async def test_ctrl_c_exits_130_after_cleanup(config_file, monkeypatch):
    fake = install_fake(monkeypatch, run=AsyncMock(side_effect=KeyboardInterrupt))
    assert await cli.main(["--config", str(config_file)]) == cli.EXIT_INTERRUPTED
    fake.shutdown.assert_awaited_once()


async def test_verbose_puts_debug_on_the_console(config_file, monkeypatch):
    install_fake(monkeypatch)
    await cli.main(["--config", str(config_file), "--text", "hi", "-v"])
    console = next(h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler)
    assert console.level == logging.DEBUG


async def test_configured_json_format_is_used_for_file_logs(config_file, monkeypatch, tmp_path):
    with config_file.open("a", encoding="utf-8") as config:
        config.write("  log_format: json\n")
    install_fake(monkeypatch)
    assert await cli.main(["--config", str(config_file), "--text", "hi"]) == cli.EXIT_OK
    for handler in logging.getLogger().handlers:
        handler.flush()
    events = [json.loads(line) for line in (tmp_path / "logs" / "jarvis.log").read_text(
        encoding="utf-8").splitlines()]
    assert any(event["message"].startswith("starting jarvis") for event in events)
    assert all(event["run_id"] == events[0]["run_id"] for event in events)


async def test_invalid_log_format_is_a_configuration_error(config_file, monkeypatch):
    with config_file.open("a", encoding="utf-8") as config:
        config.write("  log_format: xml\n")
    fake = install_fake(monkeypatch)
    assert await cli.main(["--config", str(config_file), "--text", "hi"]) == cli.EXIT_BAD_CONFIG
    fake.startup.assert_not_called()


async def test_list_tools_prints_the_inventory_without_starting_services(config_file, monkeypatch, capsys):
    fake = install_fake(monkeypatch)
    assert await cli.main(["--config", str(config_file), "--list-tools"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("# Tool inventory")
    assert "`run_command`" in out and "`desktop_observe`" in out
    fake.startup.assert_not_called()
