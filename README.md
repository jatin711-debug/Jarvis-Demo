# JARVIS - Local AI Voice Assistant

A fully local, voice-enabled AI assistant with intelligent memory, natural speech interaction, and system control.

## Features

- Wake word detection (OpenWakeWord) with echo suppression
- Streaming responses — sentences are spoken as the LLM generates them (markdown markup and raw URLs are not read aloud)
- **Barge-in** — interrupt Jarvis mid-response just by talking
- ReAct agent with native tool calling (Ollama or llama.cpp) + automatic text-parsing fallback
- Optional Cua desktop backend and resumable fixture/Calculator acceptance workflows
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

# 5. Check the setup (Ollama, models, audio devices, packages), then run Jarvis
jarvis --doctor   # prints what is wrong and how to fix it
jarvis            # console entry point
python -m jarvis  # equivalent
python -m jarvis --text "How are you doing?"  # one turn without voice hardware
```

The Piper TTS voice model is auto-downloaded to `~/.jarvis/piper_voices/` on first run
(only when `voice.tts.engine: "piper"`).

**Smaller, faster on a consumer GPU:** [llama.cpp](docs/llamacpp.md) runs the same Qwen model as a
text-only GGUF that fits a 4 GB card entirely (Ollama's bundled vision encoder pushes it onto the
CPU) and Jarvis can start and stop `llama-server` for you.

### Configuration
`config.yaml` holds the safe defaults. Put machine-specific changes (audio device,
wider permissions) in an untracked `config.local.yaml` that starts with
`extends: config.yaml`; it is picked up automatically. Pick another file with
`--config PATH` or `JARVIS_CONFIG`. See [security profiles](docs/security.md#profiles).
Compare models or catch regressions with `python -m jarvis.bench` ([scenario benchmark](docs/bench.md)).
Notable options: `voice.barge_in`, `system.latency_report`, `voice.tts.engine` (`kokoro` or `piper`).
`voice.input_device` / `voice.output_device` take an index or a device name such as `"JBL"`;
a name keeps working when a headset reconnects and indexes shift, and falls back to the system
default (with a warning) when that device is not connected. `jarvis --doctor` lists the devices.

### Logs
Console output is one readable line per event, tagged with the turn number
(`jarvis -v` adds DEBUG detail). The full record, always at DEBUG, is in
`~/.jarvis/logs/jarvis.log`. See [logging](docs/logging.md) for how to read it.

## Architecture

```
jarvis/
├── core/           # Orchestrator (service startup, voice event loop)
├── voice/          # Wake word, STT, TTS, audio cues, transcript filters
├── agent/          # LLM client, ReAct loop, capability router, permission gateway, tool registry
├── desktop/        # Cua window-scoped desktop backend
├── tasks/          # Durable task store and resumable workflows (text CLI)
├── memory/         # Mem0 integration
├── tools/          # System, browser, and vision tools
└── utils/          # Config loader, logging, deadlines

tests/
├── unit/           # Automated pytest suite (no hardware/services needed)
└── manual/         # Interactive scripts requiring mic/Ollama (run directly)
```

## Tech Stack
- **LLM**: Qwen 3.5 (via Ollama or llama.cpp, native tool calling; HuggingFace provider experimental)
- **STT**: faster-whisper (distil-small.en, int8)
- **TTS**: Kokoro-82M (default) or Piper
- **Wake Word**: OpenWakeWord
- **Memory**: Mem0.ai + ChromaDB (telemetry disabled)

## Development

```bash
pip install -e .[dev]   # pytest, pytest-asyncio, ruff, coverage

pytest                  # run the unit suite (tests/unit)
ruff check .            # lint
coverage run -m pytest && coverage report   # coverage, with the floor CI enforces
```

The unit suite mocks the LLM and needs no microphone, Ollama, or models.
The scripts in `tests/manual/` exercise specific live services. The Cua smoke
needs an interactive Windows desktop but no model or voice devices. See
[durable task commands](docs/tasks.md), [current permission behavior](docs/security.md),
[how to read the logs](docs/logging.md), the [scenario benchmark](docs/bench.md), the [tool inventory](docs/tools.md)
(`jarvis --list-tools`), and the [roadmap](docs/roadmap.md).

CI runs pytest + ruff on every push (see `.github/workflows/ci.yml`).

### Experimental / unreached capabilities
`ReActAgent.research()` (autonomous web research) exists but is not wired into the
voice loop. The HuggingFace provider has decoded-text streaming; complex tool
reliability still needs end-to-end evaluation. Desktop automation goes through the
Cua backend and the permission gateway; the earlier screenshot-driven
`computer_use()` loop was removed because its global typing tools are refused by
the gateway.

## License
MIT
