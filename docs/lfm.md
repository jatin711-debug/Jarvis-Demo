# Optional LFM2.5-350M profile

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[lfm,desktop]"
.\.venv\Scripts\python.exe -m jarvis --config config.lfm.yaml
```

For the CUDA-enabled global Python 3.12 on this machine, use:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' -m jarvis --config config.lfm.yaml
```

That interpreter must also have the voice and desktop dependencies installed.
They are not inherited from `.venv`. Install the project extras with that exact
interpreter (`-m pip install -e ".[lfm,desktop]"`) before full voice startup.
The voice-free model test can run without installing Jarvis as a package:

```powershell
python -c "import runpy; runpy.run_path('tests/manual/lfm_smoke.py', run_name='__main__')"
```

The current checkpoint's tokenizer requires newer Transformers; the LFM extra
requires 5.16.1 or later in the 5.x series. No remote tokenizer code is enabled.

The first startup downloads `LiquidAI/LFM2.5-350M` from Hugging Face to its normal
model cache. Later inference is local. This is a separate profile, not an overlay
on config.yaml: unspecified settings use application defaults. Your Qwen profile
is unchanged. Stop the existing Jarvis process before starting this profile.

This profile disables memory (Mem0 currently assumes an Ollama extraction model),
uses microphone device 2 (JBL), distil-small.en recognition, and Kokoro am_michael
to match the original voice profile. It enables Cua and dangerous-operation voice
confirmations. It does not need Ollama for chat. It is text-only: use Cua's
accessibility observations, not screenshot interpretation.

The Hugging Face backend uses real decoded-text streaming and preserves Liquid
tool-call markers for the existing parser. Cancellation requests generation stop
at the next model step and waits for the worker before releasing the model lock.
A native inference step cannot be forcibly interrupted; this isn't a hard
process-level deadline. Model loading itself is outside the response deadline.

Run the opt-in, voice-free test with:

```powershell
.\.venv\Scripts\python.exe tests\manual\lfm_smoke.py
```

It tests a greeting and a harmless deterministic tool, not desktop reliability.
The model runs on the devices supported by your installed PyTorch build; a CPU
build cannot use CUDA. CPU Transformers speed is not comparable to advertised
optimized GGUF runtime throughput. Keep Qwen as the default until task accuracy
and latency meet your needs. No automatic model fallback/routing is introduced.
