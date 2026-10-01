# LFM profile (local HuggingFace model)

`config.lfm.yaml` is `config.yaml` with the local `LiquidAI/LFM2.5-350M` model
instead of Ollama/Qwen (`extends: config.yaml`, so security, voice and desktop
settings are inherited). Memory is off in this profile: Mem0 needs an Ollama
extraction model. It does not need Ollama for chat. The model is text-only; the
agent uses Cua accessibility observations, not screenshots.

## Which Python

Use an interpreter whose PyTorch has CUDA and whose Transformers is 5.16.1 or
later in the 5.x series (the checkpoint's tokenizer needs it). On this machine
that is the global Python 3.12, not `.venv` (CPU-only torch, older Transformers):

```powershell
$py = 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pip install -e ".[lfm,desktop]"   # only if packages are missing
& $py -m jarvis --doctor                   # checks devices and packages
```

## Making it your default

Per-machine, in the untracked `config.local.yaml`: `extends: config.lfm.yaml`.
Then plain `python -m jarvis` uses LFM. To go back to Qwen, change it to
`extends: config.yaml` and start Ollama. The committed `config.yaml` stays on Qwen.

## Run it

```powershell
# 1. Model only, no voice hardware: a greeting and one harmless tool call.
& $py tests\manual\lfm_smoke.py

# 2. The whole agent on text: routing, tools, permissions. No mic or speakers.
& $py -m jarvis --text "What time is it?"

# 3. The full voice pipeline: wake word -> record -> STT -> LFM -> TTS.
& $py -m jarvis -v
```

The first run downloads the checkpoint to the Hugging Face cache; later runs are
local. Read the console as one line per event; the full DEBUG record is in
`~/.jarvis/logs/jarvis.log` ([logging](logging.md)). The `latency` line splits
record, STT, first token and TTS time.

## Behavior to know

- The backend streams decoded text and keeps Liquid's tool-call markers for the
  existing parser. Cancelling a turn stops generation at the next model step; a
  native inference step cannot be forcibly interrupted. Model loading is outside
  the response deadline.
- The model runs on whatever your PyTorch build supports; a CPU build cannot use
  CUDA, and CPU Transformers speed is not comparable to GGUF runtimes.
- The 350M model is for deterministic, structured tasks; its card does not
  recommend it for knowledge-heavy work. Larger LFM2.5 checkpoints exist, but a
  2.6B model in 16-bit is about 5 GB and does not fit this machine's 4 GB GPU.
  Compare models on the same scenarios before changing the default.
- No automatic model fallback or routing is introduced.
