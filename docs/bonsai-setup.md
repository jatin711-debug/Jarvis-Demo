# Bonsai 2 4B local setup

On the configured Windows machine, `config.local.yaml` extends `config.bonsai.yaml`,
so running `python -m jarvis` from the project root starts Bonsai 2 4B by default.
Jarvis launches and stops the local Prism CUDA `llama-server` automatically on
`127.0.0.1:18083`, separate from the usual `8080` server port.

The project-local files are:

```text
config.bonsai.yaml
runtime/prism/llama-server.exe           # Prism b10777-e3de715 and its DLLs
models/bonsai-4b/Ternary-Bonsai-2-4B-PQ2_0.gguf
models/bonsai-4b/Ternary-Bonsai-2-4B-mmproj-Q8_0.gguf
```

The `runtime/` and `models/` directories are ignored by Git because they contain
large platform-specific binaries. To set up another Windows/NVIDIA machine,
download both the CUDA 12.4 binary and CUDA runtime archives from
[Prism b10777-e3de715](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715),
and extract both into `runtime/prism/`. Download the two named GGUF files from the
[Bonsai 4B model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/tree/8d6557168168f91ea6697a4a49ed2ef622b53adb)
into `models/bonsai-4b/`. The setup on this machine uses that pinned model revision.
If needed, make a `config.local.yaml` containing `extends: config.bonsai.yaml`,
and add machine-specific voice or security settings there.

```powershell
.\.venv\Scripts\python.exe -m jarvis --doctor
.\.venv\Scripts\python.exe -m jarvis --text "What time is it?"
.\.venv\Scripts\python.exe -m jarvis
```

The profile uses the plain PQ2 model, thinking off, and 8192 tokens of context.
The Q8 projector enables screenshot/image requests. Cua desktop actions still use
the accessibility tree; this profile does not add screenshot-based Cua targeting.
The earlier model-driven Calculator probe was unreliable for this model, so verify
desktop task outcomes instead of relying on the assistant's completion claim.

`config.yaml` retains the portable Ollama setup; choosing the Bonsai profile
locally does not require changing another machine's configuration.
