"""Unit tests for jarvis.utils.config."""

import pytest

from jarvis.utils.config import JarvisConfig, expand_path, load_config


class TestDefaults:
    def test_default_config_values(self):
        cfg = JarvisConfig()
        assert cfg.system.name == "Jarvis"
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.base_url == "http://localhost:11434"
        assert cfg.voice.wake_word == "hey_jarvis"
        assert cfg.memory.enabled is True
        assert cfg.security.confirm_dangerous_ops is True
        assert cfg.security.command_whitelist is True

    def test_default_allowed_commands_present(self):
        cfg = JarvisConfig()
        assert "echo" in cfg.security.allowed_commands
        assert "shutdown" in cfg.security.blocked_commands


class TestYamlLoading:
    def test_partial_yaml_merges_with_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "system:\n  name: TestJarvis\n"
            "llm:\n  model: test-model:1b\n  temperature: 0.5\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.system.name == "TestJarvis"
        assert cfg.llm.model == "test-model:1b"
        assert cfg.llm.temperature == 0.5
        # Untouched sections keep defaults
        assert cfg.voice.wake_word == "hey_jarvis"

    def test_empty_yaml_gives_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg == JarvisConfig()

    def test_missing_explicit_path_fails_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_shape_rejected(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  temperature: not-a-number\n", encoding="utf-8")
        with pytest.raises(Exception):
            load_config(cfg_file)


class TestExpandPath:
    def test_tilde_expands_to_home(self):
        from pathlib import Path
        assert expand_path("~") == Path.home()

    def test_subpath_under_home(self):
        from pathlib import Path
        assert expand_path("~/.jarvis/logs") == Path.home() / ".jarvis" / "logs"
