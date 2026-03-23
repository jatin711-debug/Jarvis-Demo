# JARVIS - Local AI Voice Assistant

A fully local, voice-enabled AI assistant with intelligent memory, natural speech interaction, and system control.

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

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull required Ollama models
ollama pull hf.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M
ollama pull nomic-embed-text

# 4. Run Jarvis
python -m jarvis
```

### Configuration
Edit `config.yaml` to customize wake word, voice settings, model selection, and permissions.

## Architecture

```
jarvis/
├── core/           # Orchestrator, event bus, state management
├── voice/          # Wake word, STT, TTS
├── agent/          # LLM interaction, ReAct loop, tool registry
├── memory/         # Mem0 integration, context management
├── tools/          # System control tools (shell, files, apps, web)
└── utils/          # Config loader, logging, helpers
```

## Tech Stack
- **LLM**: NVIDIA Nemotron-3-Nano-4B (via Ollama)
- **STT**: Whisper (via faster-whisper)
- **TTS**: Kokoro-82M
- **Wake Word**: OpenWakeWord
- **Memory**: Mem0.ai + ChromaDB
- **Backend**: FastAPI + asyncio

## License
MIT
