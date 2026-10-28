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

## Faster replies with MTP

Qwen3.5 GGUFs from Unsloth's [`Qwen3.5-4B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF)
carry a small multi-token-prediction head. The server uses it to draft a few tokens ahead and
check them in one pass (speculative decoding), so replies come out faster with the same model:

```yaml
llm:
  server:
    model: "D:/llm-models/Qwen3.5-4B-MTP-Q4_K_M.gguf"   # the repo's Qwen3.5-4B-Q4_K_M.gguf, 2.83 GB
    args: ["-ngl", "99", "-fa", "on", "-lm", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
```

Needs a llama.cpp build that has `--spec-type draft-mtp` (merged in May 2026; check
`llama-server --help`). Measured on the reference machine (RTX 3050 Laptop), same Jarvis sampling
(thinking off, temperature 0.1), median of 3 runs:

| Draft tokens (`--spec-draft-n-max`) | Explanation | Facts | Code | Tool call |
|---|---|---|---|---|
| off (the plain Q4_K_M) | 29.6 tok/s | 29.3 | 29.8 | 29.6 |
| **2** | **35.8** | **35.6** | **45.4** | **45.2** |
| 3 | 26.9 | 32.3 | 44.7 | 48.8 |
| 4 | 24.0 | 30.6 | 45.7 | 48.1 |

- Keep it at **2**: it is the only setting never slower than no MTP. At 3 or 4 prose replies get
  slower than the baseline (fewer drafts are accepted) while tool calls and code gain a little more.
- The [benchmark](bench.md), 13 scenarios x 3 runs, passed 39/39 both ways. Median reply time went
  from 2.3 s to 1.8 s and p95 from 5.7 s to 4.3 s (a second baseline run gave 2.3 s and 5.2 s, so
  it is not drift). The first words arrive at the same time; only the generation after them speeds up.
- Cost: about 270 MiB more VRAM (2.9 to 3.2 GB in use) and 110 MB more RAM.
- **`--mmproj` does not work with MTP**, and neither does `-np` above 1 (Jarvis uses 1). Without
  a projector the model cannot read images: desktop control through Cua is not affected (it reads
  accessibility trees, not screenshots), but screenshot questions and the PDF/image tools are. To
  use them, switch back to the plain Q4_K_M, drop the two `--spec-*` flags, and set `mmproj`.

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
| `-lm none` (older builds: `--no-mmap`) | the model is not kept in RAM after it is copied to the GPU: an idle server's resident memory dropped from 3.1 GB to about 1 GB. Flag names change between builds; `llama-server --help` is the authority |
| `-cram N` | set from `llm.server.cache_ram` (default 0) by Jarvis. The server keeps old prompts in host RAM for reuse, up to 8 GB by default: it grew from 1 GB to 2.9 GB over the 13 [benchmark](bench.md) scenarios and would keep going. With 0 it stayed at 1.4 GB, and the median reply time did not change (2.3 s). Put `-cram` in `args` to choose your own, or set `cache_ram: null` for a build that rejects the flag |
| `-c` | set from `llm.context_window` by Jarvis (8192 in `config.llamacpp.yaml`). Controlling an app adds about 400 tokens per step and 4096 ran out after three observations; Qwen3.5 keeps a cache for only 8 of its 32 layers, so 8192 costs about 150 MiB more VRAM than 4096 for the 4B (3.2 to 3.4 GB in use with MTP) |

## Behavior to know

- Thinking is switched off per request for Qwen3.5 models, as in the Ollama path.
- Images work only when the server was started with a multimodal projector: set
  `llm.server.mmproj` (about 0.7 GB more VRAM for the 4B). Without it Jarvis reports that the
  provider cannot read images and carries on with text.
- `memory.enabled` is off in this profile: Mem0 still extracts and embeds through Ollama.
- The committed `config.yaml` stays on Ollama so a fresh checkout needs nothing but Ollama.
