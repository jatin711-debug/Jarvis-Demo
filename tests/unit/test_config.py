"""Unit tests for jarvis.utils.config."""

from pathlib import Path

import pytest

from jarvis.utils.config import JarvisConfig, SecurityConfig, expand_path, load_config


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


class TestExtends:
    def test_extends_merges_mappings_and_replaces_lists(self, tmp_path):
        (tmp_path / "base.yaml").write_text(
            "llm:\n  model: base-model\n  temperature: 0.2\n"
            "security:\n  allowed_commands: [git, echo]\n  confirm_dangerous_ops: true\n",
            encoding="utf-8")
        child = tmp_path / "child.yaml"
        child.write_text(
            "extends: base.yaml\nllm:\n  temperature: 0.7\n"
            "security:\n  allowed_commands: [dir]\n",
            encoding="utf-8")
        cfg = load_config(child)
        assert cfg.llm.model == "base-model"          # inherited
        assert cfg.llm.temperature == 0.7              # overridden
        assert cfg.security.allowed_commands == ["dir"]  # lists replace, not append
        assert cfg.security.confirm_dangerous_ops is True

    def test_extends_is_relative_to_the_extending_file(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        (tmp_path / "base.yaml").write_text("system:\n  name: FromBase\n", encoding="utf-8")
        child = nested / "child.yaml"
        child.write_text("extends: ../base.yaml\n", encoding="utf-8")
        assert load_config(child).system.name == "FromBase"

    def test_extends_cycle_is_rejected(self, tmp_path):
        (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cycle"):
            load_config(tmp_path / "a.yaml")

    def test_missing_base_fails_clearly(self, tmp_path):
        child = tmp_path / "child.yaml"
        child.write_text("extends: nope.yaml\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_config(child)

    def test_top_level_must_be_a_mapping(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(bad)


class TestShippedProfiles:
    """The committed configs must load, and the default must be the safe profile."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_default_config_is_the_standard_security_profile(self):
        cfg = load_config(self.ROOT / "config.yaml")
        assert cfg.security.confirm_dangerous_ops is True
        assert cfg.security.command_whitelist is True
        assert cfg.security.allowed_directories == ["~"]
        assert cfg.security.blocked_commands  # non-empty blocklist
        assert cfg.system.log_level == "INFO"

    def test_code_defaults_match_the_shipped_security_profile(self):
        """A config that omits a security key must not be wider than config.yaml."""
        assert SecurityConfig() == load_config(self.ROOT / "config.yaml").security

    def test_bonsai_profile_loads(self):
        cfg = load_config(self.ROOT / "config.bonsai.yaml")
        assert cfg.llm.provider == "llamacpp"
        assert cfg.llm.model == "ternary-bonsai-2-4b"
        assert cfg.llm.server.mmproj is not None

    def test_bonsai_profile_inherits_the_safe_defaults_and_no_machine_settings(self):
        base = load_config(self.ROOT / "config.yaml")
        bonsai = load_config(self.ROOT / "config.bonsai.yaml")
        assert bonsai.security == base.security
        assert bonsai.voice == base.voice
        assert bonsai.voice.input_device is None, "device indexes belong in config.local.yaml"
        assert bonsai.memory.enabled is False
        assert bonsai.llm.model != base.llm.model
