"""
JARVIS Configuration Loader
============================
Loads and validates config.yaml, merging with defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Pydantic config models (type-safe + validated)
# ──────────────────────────────────────────────


class SystemConfig(BaseModel):
    name: str = "Jarvis"
    debug: bool = False
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
    engine: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    piper_model: str = "en_US-lessac-medium.onnx"


class VoiceConfig(BaseModel):
    wake_word: str = "hey_jarvis"
    sensitivity: float = 0.5
    input_device: int | None = None
    output_device: int | None = None
    barge_in: bool = True  # Let the user interrupt TTS playback by speaking
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class LLMConfig(BaseModel):
    provider: str = "ollama"      # "ollama" or "huggingface"
    model: str = "qwen3:1.7b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50               # For LFM (HuggingFace)
    repetition_penalty: float = 1.05  # For LFM (HuggingFace)
    context_window: int = 8192
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
    max_conversation_turns: int = 15
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


class SecurityConfig(BaseModel):
    confirm_dangerous_ops: bool = True
    command_whitelist: bool = True
    allowed_commands: list[str] = Field(default_factory=lambda: [
        "code", "notepad", "calc", "explorer", "chrome", "firefox",
        "msedge", "powershell", "cmd", "ping", "ipconfig",
        "systeminfo", "tasklist", "dir", "type", "echo",
    ])
    blocked_commands: list[str] = Field(default_factory=lambda: [
        "rm -rf", "format", "del /s", "shutdown", "restart",
    ])
    allowed_directories: list[str] = Field(default_factory=lambda: [
        "~", "C:\\Users", "D:\\",
    ])


class JarvisConfig(BaseModel):
    """Root configuration object for JARVIS."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

_CONFIG_SEARCH_PATHS = [
    Path("config.yaml"),
    Path("config.yml"),
    Path.home() / ".jarvis" / "config.yaml",
]


def _find_config_file() -> Path | None:
    """Search for config.yaml in standard locations."""
    # Check env var first
    env_path = os.environ.get("JARVIS_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    for path in _CONFIG_SEARCH_PATHS:
        if path.exists():
            return path
    return None


def load_config(config_path: str | Path | None = None) -> JarvisConfig:
    """
    Load JARVIS configuration.

    Priority:
    1. Explicit path argument
    2. JARVIS_CONFIG environment variable
    3. ./config.yaml
    4. ~/.jarvis/config.yaml
    5. Pure defaults
    """
    path: Path | None = None

    if config_path:
        path = Path(config_path)
    else:
        path = _find_config_file()

    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return JarvisConfig.model_validate(raw)

    # No config file found — use defaults
    return JarvisConfig()


def expand_path(path_str: str) -> Path:
    """Expand ~ and env vars in a path string, then resolve."""
    return Path(os.path.expanduser(os.path.expandvars(path_str))).resolve()
