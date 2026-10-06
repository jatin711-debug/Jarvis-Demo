# llama.cpp backend

`llm.provider: llamacpp` talks to a [`llama-server`](https://github.com/ggml-org/llama.cpp)
(OpenAI-compatible API, native tool calling) running a GGUF model. Jarvis can start and
stop the server itself.

## Why

On the reference machine (RTX 3050 Laptop, 4 GB VRAM, 15 GB RAM) the same Qwen3.5 model is much
lighter and faster as a text-only GGUF than through Ollama, which bundles the vision encoder:

| Setup | VRAM | Where it runs | Generation |
|---|---|---|---|
| Ollama `qwen3.5:4b`, `num_ctx` 4096 | 3.7 GB | 50% CPU / 50% GPU | ~14 tok/s |
| llama.cpp, Qwen3.5-4B Q4_K_M (2.7 GB file), CUDA | 2.9 GB in use | all on GPU | ~31 tok/s, prompt ~780 tok/s |
| llama.cpp, same model, Vulkan build (winget `ggml.llamacpp`, or the newer release) | 2.9 GB | all on GPU | ~27-28 tok/s |
| llama.cpp, Qwen3.5-2B Q4_K_M (1.3 GB file), CUDA | 1.4 GB | all on GPU | ~60 tok/s, prompt ~1900 tok/s |

`llama-bench` on an otherwise idle machine, one run of each, so treat the figures as approximate;
the Ollama figure is one short cold prompt. Benchmark with nothing else running: the same Vulkan
build measured 7 tok/s while two downloads ran and 28 tok/s once they finished. Through the whole agent, four text prompts (greeting, system
usage, list a folder, read a config) were answered correctly by both the 4B and 2B models in 2.4-5.2 s
each including Python start-up. That is too small a sample to rank them; per Qwen's model cards the
4B scores far higher on agent benchmarks (τ²-Bench 79.9 vs 48.8, BFCL-V4 50.3 vs 43.6).

## Setup

1. **llama.cpp.** `winget install ggml.llamacpp` gives a Vulkan build that works on any GPU and
   is enough. For an NVIDIA card the CUDA build is about 10% faster: take
   `llama-<build>-bin-win-cuda-<ver>-x64.zip` and the matching `cudart-...zip` from the
   [releases page](https://github.com/ggml-org/llama.cpp/releases), and unpack both into one folder.
2. **A model.** Q4_K_M is the usual size/quality balance. Text-only files keep VRAM free:
   ```powershell
   python -c "from huggingface_hub import hf_hub_download as d; print(d('unsloth/Qwen3.5-4B-GGUF', 'Qwen3.5-4B-Q4_K_M.gguf', local_dir='D:/llm-models'))"
   ```
3. **Config.** In your untracked `config.local.yaml`:
   ```yaml
   extends: config.llamacpp.yaml
   llm:
     server:
       command: "D:/llm-models/llama-cuda/llama-server.exe"
       model: "D:/llm-models/Qwen3.5-4B-Q4_K_M.gguf"
       args: ["-ngl", "99", "-fa", "on", "-lm", "none"]
   ```
4. **Check and run.**
   ```powershell
   python -m jarvis --doctor
   python -m jarvis --text "What time is it?"
   python -m jarvis -v
   ```

## How the server is managed

- If `llm.server.model` is set and nothing answers at `llm.base_url`, Jarvis starts `llama-server`
  with your `args` plus `-m <model> --host --port -c <context_window> -np 1 --alias <llm.model>`
  (Jarvis's flags come last, so they win), waits until the model is loaded, and stops it on exit.
- A server that is already running is used as it is and never stopped by Jarvis.
- If a server Jarvis started dies (killed for memory, a driver reset), it is restarted before the
  next request. A request that loses the server before any of its reply was produced is sent
  again once; a reply cut off part-way is reported as an error rather than replayed, and the
  following request restarts the server. A restart that fails (or never finishes loading) is
  reported with the reason and stops the half-started server. A server you started yourself is
  never restarted by Jarvis.
- If Jarvis is killed without shutting down, its server keeps running (holding the VRAM). The next
  start reuses it. Stop it with `taskkill /IM llama-server.exe`.
- Server output goes to `~/.jarvis/logs/llama-server.log`. If the server exits while loading, the
  error shows its last lines (for example an unknown flag).
- Without `llm.server.model`, start the server yourself and pass an explicit `--port`: newer
  builds announce a different default port.

## Flags that matter

| Flag | Why |
|---|---|
| `-ngl 99` | every layer on the GPU |
| `-fa on` | flash attention: less memory, faster |
| `-lm none` (older builds: `--no-mmap`) | the model is not kept in RAM after it is copied to the GPU: the server's resident memory dropped from 3.1 GB to about 1 GB. Flag names change between builds; `llama-server --help` is the authority |
| `-c` | set from `llm.context_window` by Jarvis. A model that fits comfortably at 4096 leaves room to raise it |

## Behavior to know

- Thinking is switched off per request for Qwen3.5 models, as in the Ollama path.
- Images work only when the server was started with a multimodal projector: set
  `llm.server.mmproj` (about 0.7 GB more VRAM for the 4B). Without it Jarvis reports that the
  provider cannot read images and carries on with text.
- `memory.enabled` is off in this profile: Mem0 still extracts and embeds through Ollama.
- The committed `config.yaml` stays on Ollama so a fresh checkout needs nothing but Ollama.
