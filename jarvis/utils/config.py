"""
JARVIS Configuration Loader
============================
Loads and validates config.yaml, merging with defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Pydantic config models (type-safe + validated)
# ──────────────────────────────────────────────


class SystemConfig(BaseModel):
    name: str = "Jarvis"
    log_level: str = "INFO"
    log_dir: str = "~/.jarvis/logs"
    data_dir: str = "~/.jarvis"
    latency_report: bool = True  # Log per-turn timing breakdown (record/STT/LLM/TTS)


class STTConfig(BaseModel):
    model: str = "small"
    language: str = "en"
    beam_size: int = 5
    compute_type: str = "int8"


class TTSConfig(BaseModel):
    engine: Literal["kokoro", "piper"] = "kokoro"
    voice: str = "af_heart"
    speed: float = Field(default=1.0, gt=0, le=3)
    sample_rate: int = 24000
    piper_model: str = "en_US-lessac-medium.onnx"


class VoiceConfig(BaseModel):
    wake_word: str = "hey_jarvis"
    sensitivity: float = Field(default=0.5, ge=0, le=1)
    input_device: int | None = None
    output_device: int | None = None
    barge_in: bool = True  # Let the user interrupt TTS playback by speaking
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class LLMConfig(BaseModel):
    provider: Literal["ollama", "huggingface"] = "ollama"
    model: str = "qwen3:1.7b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.6, ge=0, le=2)
    top_p: float = 0.95
    top_k: int = 50               # For LFM (HuggingFace)
    repetition_penalty: float = 1.05  # For LFM (HuggingFace)
    context_window: int = Field(default=8192, ge=1024)
    request_timeout: float = Field(default=120.0, gt=0)
    first_response_timeout: float = Field(default=30.0, gt=0)
    max_tool_result_chars: int = Field(default=4000, ge=512)
    max_history_chars: int = Field(default=8000, ge=0)
    system_prompt: str = (
        "You are Jarvis, a helpful local AI assistant. "
        "Be concise, helpful, and proactive."
    )


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    collection_name: str = "jarvis_memories"
    path: str = "~/.jarvis/vectors"


class MemoryConfig(BaseModel):
    enabled: bool = True
    auto_save: bool = True
    retrieval_timeout: float = Field(default=2.0, gt=0)
    max_conversation_turns: int = Field(default=15, ge=0, le=100)
    pending_writes: int = Field(default=16, ge=1, le=1000)
    shutdown_timeout: float = Field(default=10.0, gt=0)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


class DesktopConfig(BaseModel):
    """Cua is opt-in and imported only when selected."""

    backend: Literal["legacy", "cua"] = "legacy"
    timeout: float = Field(default=30.0, gt=0, le=120)
    max_elements: int = Field(default=80, ge=1, le=200)
    max_depth: int = Field(default=10, ge=1, le=20)
    snapshot_ttl: float = Field(default=30.0, gt=0, le=120)
    delivery_mode: Literal["background", "foreground"] = "background"
    telemetry: bool = False
    update_check: bool = False


class SecurityConfig(BaseModel):
    """Defaults are the standard profile in config.yaml (a test keeps them identical),
    so a config that omits a key never ends up wider than the shipped one."""

    confirm_dangerous_ops: bool = True
    command_whitelist: bool = True
    allowed_commands: list[str] = Field(default_factory=lambda: [
        "git", "echo", "dir", "type", "ping", "ipconfig", "systeminfo", "tasklist",
        "notepad", "calc", "explorer", "code",
    ])
    blocked_commands: list[str] = Field(default_factory=lambda: [
        "rm -rf", "format", "del /s", "shutdown", "restart",
    ])
    allowed_directories: list[str] = Field(default_factory=lambda: ["~"])
    # None preserves legacy unrestricted targets; [] denies every target.
    allowed_applications: list[str] | None = None
    allowed_hosts: list[str] | None = None


class JarvisConfig(BaseModel):
    """Root configuration object for JARVIS."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

_CONFIG_SEARCH_PATHS = [
    Path("config.local.yaml"),  # untracked, per-machine overrides
    Path("config.yaml"),
    Path("config.yml"),
    Path.home() / ".jarvis" / "config.yaml",
]


def _find_config_file() -> Path | None:
    """Search for a config file in standard locations."""
    # Check env var first
    env_path = os.environ.get("JARVIS_CONFIG")
    if env_path:
        return expand_path(env_path)

    for path in _CONFIG_SEARCH_PATHS:
        if path.exists():
            return path
    return None


def resolve_config_path(config_path: str | Path | None = None) -> Path | None:
    """The file `load_config` would read, or None when defaults apply."""
    if config_path:
        return expand_path(str(config_path))
    return _find_config_file()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; any other value (lists included) is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_layers(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read ``path``, first applying the file named by its top-level ``extends`` key."""
    path = path.resolve()
    if path in chain:
        raise ValueError(f"Config 'extends' cycle: {' -> '.join(str(p) for p in (*chain, path))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError(f"'extends' in {path} must be a file path")
    base = _read_layers(path.parent / os.path.expanduser(os.path.expandvars(parent)), (*chain, path))
    return _deep_merge(base, raw)


def load_config(config_path: str | Path | None = None) -> JarvisConfig:
    """
    Load JARVIS configuration.

    Priority:
    1. Explicit path argument
    2. JARVIS_CONFIG environment variable
    3. ./config.local.yaml, ./config.yaml, ./config.yml
    4. ~/.jarvis/config.yaml
    5. Pure defaults

    A file may start from another with ``extends: path/to/base.yaml``
    (relative to the file); its own keys are merged over the base.
    """
    path = resolve_config_path(config_path)
    if path is None:
        return JarvisConfig()
    return JarvisConfig.model_validate(_read_layers(path))


def expand_path(path_str: str) -> Path:
    """Expand ~ and env vars in a path string, then resolve."""
    return Path(os.path.expanduser(os.path.expandvars(path_str))).resolve()
