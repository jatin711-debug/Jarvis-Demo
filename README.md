# JARVIS - Local AI Voice Assistant

A fully local, voice-enabled AI assistant with intelligent memory, natural speech interaction, and system control.

## Features

- Wake word detection (OpenWakeWord) with echo suppression
- Streaming responses — sentences are spoken as the LLM generates them
- **Barge-in** — interrupt Jarvis mid-response just by talking
- ReAct agent with native Ollama tool calling + automatic text-parsing fallback
- ~40 tools: shell, files, apps, windows, clipboard, browser (Playwright), vision, scheduler, web search
- Long-term memory (Mem0 + ChromaDB), 100% local embeddings
- Per-turn latency report (record / STT / memory / LLM / TTS timings)
- Whisper hallucination filtering and garbage-transcript rejection

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- A microphone and speakers

### Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Install the project (dependencies are declared in pyproject.toml)
pip install -r requirements.txt

# 3. Pull required Ollama models
ollama pull qwen3.5:2b
ollama pull nomic-embed-text

# 4. Install the Playwright browser (for web automation tools)
playwright install chromium

# 5. Run Jarvis
jarvis            # console entry point
python -m jarvis  # equivalent
```

The Piper TTS voice model is auto-downloaded to `~/.jarvis/piper_voices/` on first run
(only when `voice.tts.engine: "piper"`).

### Configuration
Edit `config.yaml` to customize wake word, voice settings, model selection, and permissions.
Notable options: `voice.barge_in`, `system.latency_report`, `voice.tts.engine` (`kokoro` or `piper`).

## Architecture

```
jarvis/
├── core/           # Orchestrator (voice event loop)
├── voice/          # Wake word, STT, TTS (engine interfaces + implementations)
├── agent/          # LLM client, ReAct loop, tool registry
├── memory/         # Mem0 integration
├── tools/          # System, browser, and vision tools
└── utils/          # Config loader, logging

tests/
├── unit/           # Automated pytest suite (no hardware/services needed)
└── manual/         # Interactive scripts requiring mic/Ollama (run directly)
```

## Tech Stack
- **LLM**: Qwen 3.5 (via Ollama, native tool calling; HuggingFace provider experimental)
- **STT**: faster-whisper (distil-small.en, int8)
- **TTS**: Kokoro-82M (default) or Piper
- **Wake Word**: OpenWakeWord
- **Memory**: Mem0.ai + ChromaDB (telemetry disabled)

## Development

```bash
pip install -e .[dev]   # pytest, pytest-asyncio, ruff

pytest                  # run the unit suite (tests/unit)
ruff check .            # lint
```

The unit suite mocks the LLM and needs no microphone, Ollama, or models.
The scripts in `tests/manual/` are interactive and require live services.

CI runs pytest + ruff on every push (see `.github/workflows/ci.yml`).

### Experimental / unreached capabilities
`ReActAgent.research()` (autonomous web research) and `ReActAgent.computer_use()`
(see-think-act desktop automation) exist but are not yet wired into the voice loop.
The HuggingFace LLM provider works but has no true streaming.

## License
MIT
