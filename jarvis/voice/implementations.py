"""
JARVIS Voice Pipeline — Real Implementations
==============================================
Concrete STT (faster-whisper), TTS (Kokoro), and WakeWord (OpenWakeWord)
engines implementing the interfaces defined in engines.py.

Key design decision: Mic is always opened at its NATIVE sample rate
(typically 44100 Hz on Windows), then audio is resampled to 16000 Hz
in software using scipy. This avoids silent failures from drivers
that don't support hardware resampling.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import numpy as np

from jarvis.utils.logger import get_logger
from jarvis.voice.engines import STTEngine, TTSEngine, WakeWordEngine

if TYPE_CHECKING:
    from jarvis.utils.config import VoiceConfig

logger = get_logger(__name__)

# Target sample rate for STT and wake word models
_TARGET_SR = 16000

# Clock and waits are indirected so tests can drive time without sleeping.
_clock = time.monotonic
_nap = time.sleep  # blocking wait, only ever called from worker threads
_sleep = asyncio.sleep  # event-loop wait


def _describe(error: BaseException) -> str:
    """One-line ``Type: message`` text for a log field."""
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _resample(audio_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample int16 audio from orig_sr to target_sr using soxr (fast) or scipy (fallback)."""
    if orig_sr == target_sr:
        return audio_int16
    # Try soxr first — 5-10x faster than scipy, same quality
    try:
        import soxr
        resampled = soxr.resample(audio_int16.astype(np.float32), orig_sr, target_sr)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(orig_sr, target_sr)
        resampled = resample_poly(audio_int16.astype(np.float64), target_sr // g, orig_sr // g)
    # Filter ringing can overshoot full scale; clip so int16 does not wrap around.
    return np.clip(resampled, -32768, 32767).astype(np.int16)


# Native input rate per device index (None = system default), queried once: every
# query opens a PyAudio instance, which is slow and blocks.
_MIC_RATE_CACHE: dict[int | None, int] = {}


def _get_mic_native_rate(device_index: int | None = None, *, refresh: bool = False) -> int:
    """Native sample rate of an input device (blocking on the first call, then cached)."""
    if not refresh and device_index in _MIC_RATE_CACHE:
        return _MIC_RATE_CACHE[device_index]
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        if device_index is not None:
            info = pa.get_device_info_by_index(device_index)
        else:
            info = pa.get_default_input_device_info()
        logger.debug("input device index=%d name=%s", info["index"], info["name"])
        rate = int(info.get("defaultSampleRate", 44100))
    finally:
        pa.terminate()
    _MIC_RATE_CACHE[device_index] = rate
    return rate


async def _native_rate(device_index: int | None) -> int:
    """``_get_mic_native_rate`` without blocking the event loop."""
    cached = _MIC_RATE_CACHE.get(device_index)
    if cached is not None:
        return cached
    return await asyncio.to_thread(_get_mic_native_rate, device_index)


def _chunk_frames(native_sr: int, sample_rate: int, frames: int = 1024) -> int:
    """Native-rate frames that equal ``frames`` at ``sample_rate``."""
    return int(frames * native_sr / sample_rate) if native_sr != sample_rate else frames


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def _validate_mic(device_index: int | None, sample_rate: int, duration: float = 0.5) -> bool:
    """Quick mic capture test — returns True if audio level is above noise floor."""
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024,
            input_device_index=device_index,
        )
        peak_rms = 0.0
        for _ in range(int(sample_rate * duration / 1024)):
            raw = stream.read(1024, exception_on_overflow=False)
            peak_rms = max(peak_rms, _rms(np.frombuffer(raw, dtype=np.int16)))
        return peak_rms > 5.0
    except Exception as e:
        logger.warning("mic validation failed device=%s error=%s", device_index, e)
        return False
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.debug("mic stream close failed error=%s", e)
        pa.terminate()


def _list_input_devices() -> None:
    """Log the input devices PyAudio can see (blocking)."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        logger.info("available input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                logger.info(
                    "  input device index=%d name=%s channels=%d rate=%d",
                    i, info["name"], info["maxInputChannels"], int(info["defaultSampleRate"]),
                )
    finally:
        pa.terminate()


# ──────────────────────────────────────────────
# Microphone stream
# ──────────────────────────────────────────────

# Longest wait for an in-flight blocking mic call before closing the stream anyway.
_SETTLE_SECONDS = 3.0
# A mic that cannot be opened or read is retried after this long, doubling up to the cap.
_REOPEN_MIN_SECONDS = 1.0
_REOPEN_MAX_SECONDS = 10.0


def _retrieve(future: asyncio.Future) -> None:
    """Mark a future's exception as seen, so an abandoned call does not log noise."""
    if not future.cancelled():
        future.exception()


class _MicInput:
    """
    One open PyAudio input stream whose blocking calls all run in worker threads.

    open() never leaves a PyAudio instance behind when the device fails to open,
    and close() waits (bounded) for a read still in flight before it closes the
    stream, so the stream is never closed under a running read().
    """

    def __init__(self, device_index: int | None, native_rate: int, frames: int) -> None:
        self.device_index = device_index
        self.native_rate = native_rate
        self.frames = frames
        self._pa = None
        self._stream = None
        self._pending: asyncio.Future | None = None

    def _open(self) -> None:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.native_rate,
                input=True,
                frames_per_buffer=self.frames,
                input_device_index=self.device_index,
            )
        except BaseException:
            pa.terminate()
            raise
        self._pa, self._stream = pa, stream

    def _read(self) -> bytes:
        return self._stream.read(self.frames, exception_on_overflow=False)

    def _release(self) -> None:
        stream, pa = self._stream, self._pa
        self._stream = self._pa = None
        if stream is not None:
            for step in (stream.stop_stream, stream.close):
                try:
                    step()
                except Exception as e:
                    logger.debug("mic stream close failed error=%s", e)
        if pa is not None:
            try:
                pa.terminate()
            except Exception as e:
                logger.debug("mic terminate failed error=%s", e)

    async def _run(self, call: Callable[[], object]):
        # Shielded: if the caller is cancelled the call keeps running in its thread,
        # and close() can still wait for it through ``_pending``.
        future = asyncio.get_running_loop().run_in_executor(None, call)
        future.add_done_callback(_retrieve)
        self._pending = future
        return await asyncio.shield(future)

    async def open(self) -> None:
        await self._run(self._open)

    async def read(self) -> np.ndarray:
        """Next block of native-rate int16 samples."""
        raw = await self._run(self._read)
        return np.frombuffer(raw, dtype=np.int16)

    async def close(self) -> None:
        """Release the stream and PyAudio; safe to call twice or after a failed open."""
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done():
            _, still_running = await asyncio.wait({pending}, timeout=_SETTLE_SECONDS)
            if still_running:
                logger.warning("mic call still blocked after %.0fs; closing anyway", _SETTLE_SECONDS)
        if self._stream is not None or self._pa is not None:
            await asyncio.to_thread(self._release)


# ──────────────────────────────────────────────
# Gapless audio playback
# ──────────────────────────────────────────────

# TTS chunks are written in slices of this length so abort() cuts playback short.
_SLICE_SECONDS = 0.08
# drain() never waits longer than this for queued audio to play out.
_DRAIN_MAX_SECONDS = 10.0


def _audio_errors() -> tuple[type[BaseException], ...]:
    """Exceptions a failing audio device raises (sounddevice imported lazily)."""
    try:
        import sounddevice as sd
    except ImportError:
        return (OSError,)
    return (getattr(sd, "PortAudioError", OSError), OSError)


class _GaplessPlayer:
    """
    Persistent sounddevice output stream for TTS playback.

    The old per-chunk sd.play()/sd.wait() pattern closed and reopened the
    audio stream for every chunk and sentence — the source of audible gaps
    and clicks, especially over Bluetooth. This keeps ONE stream open with
    a high-latency buffer and writes chunks into it continuously.

    Call reset_abort() before each new utterance: while the abort flag is set,
    play() writes nothing. abort() is safe from any thread.
    """

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._stream = None
        self._sr: int | None = None
        self._ends_at = 0.0  # monotonic time at which everything queued has been played
        self._aborted = False
        self._broken = False  # stream was aborted or failed: reopen before the next write
        self._lock = threading.RLock()  # serialises writes with close(); abort() never waits on it

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _stream_for(self, sample_rate: int):
        """The open stream for ``sample_rate``, (re)opening it when needed. Holds the lock."""
        stream = self._stream
        if (stream is not None and not self._broken and not stream.closed
                and sample_rate == self._sr):
            return stream
        self.close()
        import sounddevice as sd
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            latency="high",  # larger buffer — survives Bluetooth jitter
            device=self._device,
        )
        try:
            stream.start()
        except BaseException:
            stream.close()
            raise
        self._stream, self._sr, self._broken, self._ends_at = stream, sample_rate, False, 0.0
        return stream

    def _write(self, piece: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            if self._aborted:
                return
            self._stream_for(sample_rate).write(piece)
            self._ends_at = max(self._ends_at, _clock()) + len(piece) / sample_rate

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Queue audio for playback, a slice at a time so abort() takes effect within
        ~``_SLICE_SECONDS``. The stream is (re)opened when the rate changes or after
        an abort; a failed write reopens it once and retries before giving up.
        """
        data = np.ascontiguousarray(audio, dtype=np.float32)
        step = max(int(sample_rate * _SLICE_SECONDS), 1)
        reopened = False
        start = 0
        while start < len(data):
            if self._aborted:
                return
            try:
                self._write(data[start:start + step], sample_rate)
            except _audio_errors() as e:
                if self._aborted:
                    return  # abort() interrupted the write; the stream is dropped, not faulty
                self._broken = True
                if reopened:
                    raise
                reopened = True
                logger.warning("audio output failed; reopening stream error=%s", e)
                continue
            start += step

    def drain(self, timeout: float = _DRAIN_MAX_SECONDS) -> None:
        """Block until everything queued so far should have finished playing (bounded)."""
        if self._stream is None:
            return
        deadline = _clock() + min(timeout, max(self._ends_at - _clock(), 0.0))
        while not self._aborted:
            left = deadline - _clock()
            if left <= 0:
                break
            _nap(min(left, 0.02))

    def reset_abort(self) -> None:
        """Clear the aborted flag before starting a new utterance."""
        self._aborted = False

    def abort(self) -> None:
        """
        Stop playback immediately (barge-in). Buffered audio is dropped and the
        stream is marked for reopening: a stopped PortAudio stream rejects writes.
        """
        self._aborted = True
        stream = self._stream
        if stream is not None and not stream.closed:
            self._broken = True
            try:
                stream.abort()
            except Exception as e:
                logger.debug("audio stream abort failed error=%s", e)

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            self._ends_at = 0.0
            if stream is not None:
                try:
                    stream.close()
                except Exception as e:
                    logger.debug("audio stream close failed error=%s", e)


# ──────────────────────────────────────────────
# Kokoro TTS
# ──────────────────────────────────────────────


class KokoroTTS(TTSEngine):
    """Text-to-Speech using Kokoro-82M (CPU-only, ~200 MB)."""

    def __init__(self, config: VoiceConfig) -> None:
        self.voice = config.tts.voice
        self.speed = config.tts.speed
        self.sample_rate = config.tts.sample_rate
        self._pipeline = None
        self._player = _GaplessPlayer(device=config.output_device)

    async def startup(self) -> None:
        loop = asyncio.get_event_loop()

        def _init_kokoro():
            # Suppress warnings inside this thread (thread pools don't inherit filters)
            import warnings
            warnings.simplefilter("ignore")

            from kokoro import KPipeline
            return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

        self._pipeline = await loop.run_in_executor(None, _init_kokoro)
        logger.info("tts ready engine=kokoro voice=%s speed=%.1f", self.voice, self.speed)

    async def shutdown(self) -> None:
        self._pipeline = None
        self._player.abort()  # unblocks a write still in flight so close() cannot wait on it
        await asyncio.to_thread(self._player.close)

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes."""
        import soundfile as sf

        loop = asyncio.get_event_loop()
        audio_chunks: list[np.ndarray] = []

        def _synth():
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
            ):
                audio_chunks.append(audio)

        await loop.run_in_executor(None, _synth)

        if not audio_chunks:
            return b""

        full_audio = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, full_audio, self.sample_rate, format="WAV")
        return buf.getvalue()

    async def speak(self, text: str) -> None:
        """Synthesize and play through speakers (gapless continuous stream)."""
        loop = asyncio.get_event_loop()
        self._player.reset_abort()

        def _speak():
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
            ):
                if self._player.aborted:
                    break
                self._player.play(np.asarray(audio, dtype=np.float32), self.sample_rate)
            self._player.drain()

        await loop.run_in_executor(None, _speak)

    def abort(self) -> None:
        """Barge-in: stop playback immediately."""
        self._player.abort()


# ──────────────────────────────────────────────
# Piper TTS (ultra-low-latency, ONNX)
# ──────────────────────────────────────────────


class PiperTTS(TTSEngine):
    """Text-to-Speech using Piper (ONNX, ~150ms synthesis)."""

    # Default voice model path under ~/.jarvis/piper_voices/
    _DEFAULT_MODEL = "en_US-lessac-medium.onnx"

    def __init__(self, config: VoiceConfig) -> None:
        self.speed = config.tts.speed
        model_name = getattr(config.tts, "piper_model", None) or self._DEFAULT_MODEL
        self._model_path = os.path.join(
            os.path.expanduser("~"), ".jarvis", "piper_voices", model_name,
        )
        self._voice = None
        self._synth_kwargs: dict = {}  # per-call synthesis options (speed), set at startup
        self.sample_rate: int = 22050  # updated from model config at startup
        self._player = _GaplessPlayer(device=config.output_device)

    def _speed_options(self) -> dict:
        """``synthesize`` keyword arguments that apply ``speed`` (Piper's length scale)."""
        if abs(self.speed - 1.0) < 1e-3:
            return {}
        try:
            from piper.config import SynthesisConfig
        except ImportError:
            logger.warning("tts speed ignored engine=piper speed=%.2f reason=piper_too_old", self.speed)
            return {}
        # length_scale stretches phoneme duration: below 1 is faster, above 1 is slower.
        return {"syn_config": SynthesisConfig(length_scale=1.0 / self.speed)}

    async def startup(self) -> None:
        from piper import PiperVoice

        if not os.path.exists(self._model_path):
            await self._download_voice_model()

        loop = asyncio.get_event_loop()

        def _load():
            return PiperVoice.load(self._model_path)

        self._voice = await loop.run_in_executor(None, _load)
        self.sample_rate = self._voice.config.sample_rate
        self._synth_kwargs = self._speed_options()
        logger.info(
            "tts ready engine=piper model=%s sample_rate=%d speed=%.1f",
            os.path.basename(self._model_path), self.sample_rate, self.speed,
        )

    async def _download_voice_model(self) -> None:
        """
        Auto-download the Piper voice (.onnx + .onnx.json) from the
        rhasspy/piper-voices HuggingFace repo when missing locally.
        Model name format: <lang>_<region>-<voice>-<quality>.onnx
        """
        import httpx

        filename = os.path.basename(self._model_path)
        stem = filename[:-5] if filename.endswith(".onnx") else filename
        try:
            lang_region, voice, quality = stem.split("-")
            lang = lang_region.split("_")[0]
        except ValueError:
            raise RuntimeError(
                f"Piper voice model not found at {self._model_path} and the "
                f"name '{filename}' doesn't follow the '<lang>_<region>-<voice>-"
                "<quality>.onnx' convention — download it manually from "
                "https://huggingface.co/rhasspy/piper-voices"
            )

        base = (
            f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            f"{lang}/{lang_region}/{voice}/{quality}/{stem}"
        )
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)

        logger.info("downloading piper voice file=%s", filename)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=httpx.Timeout(300.0, connect=15.0),
            ) as client:
                # (remote suffix, local path) — JSON config is "<model>.onnx.json"
                for remote_suffix, dest in (
                    (".onnx", self._model_path),
                    (".onnx.json", self._model_path + ".json"),
                ):
                    resp = await client.get(base + remote_suffix)
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        f.write(resp.content)
                    logger.info("piper voice saved path=%s bytes=%d", dest, len(resp.content))
        except Exception as e:
            # Clean up partial downloads so a retry isn't confused
            for partial in (self._model_path, self._model_path + ".json"):
                if os.path.exists(partial):
                    try:
                        os.remove(partial)
                    except OSError:
                        pass
            raise RuntimeError(
                f"Piper voice model missing and auto-download failed: {e}. "
                f"Download '{filename}' and '{filename}.json' manually from "
                "https://huggingface.co/rhasspy/piper-voices into "
                f"{os.path.dirname(self._model_path)}"
            )

    async def shutdown(self) -> None:
        self._voice = None
        self._player.abort()  # unblocks a write still in flight so close() cannot wait on it
        await asyncio.to_thread(self._player.close)

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes."""
        import soundfile as sf

        loop = asyncio.get_event_loop()

        def _synth():
            chunks = list(self._voice.synthesize(text, **self._synth_kwargs))
            if not chunks:
                return b""
            audio = np.concatenate([
                np.frombuffer(c.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                for c in chunks
            ])
            buf = io.BytesIO()
            sf.write(buf, audio, self.sample_rate, format="WAV")
            return buf.getvalue()

        return await loop.run_in_executor(None, _synth)

    async def speak(self, text: str) -> None:
        """Synthesize and stream to speakers (gapless continuous playback)."""
        loop = asyncio.get_event_loop()
        self._player.reset_abort()

        def _speak():
            for chunk in self._voice.synthesize(text, **self._synth_kwargs):
                if self._player.aborted:
                    break
                audio = (
                    np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
                self._player.play(audio, self.sample_rate)
            self._player.drain()

        await loop.run_in_executor(None, _speak)

    def abort(self) -> None:
        """Barge-in: stop playback immediately."""
        self._player.abort()


# ──────────────────────────────────────────────
# Faster-Whisper STT
# ──────────────────────────────────────────────


class WhisperSTT(STTEngine):
    """Speech-to-Text using faster-whisper (CTranslate2 backend)."""

    def __init__(self, config: VoiceConfig) -> None:
        self.model_size = config.stt.model
        self.compute_type = config.stt.compute_type
        self.beam_size = config.stt.beam_size
        self.language = config.stt.language
        self._model = None

    async def startup(self) -> None:
        from faster_whisper import WhisperModel

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.model_size, device="cpu", compute_type=self.compute_type
            ),
        )
        logger.info("stt ready engine=whisper model=%s compute=%s", self.model_size, self.compute_type)

    async def shutdown(self) -> None:
        self._model = None

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes (WAV) → text. Passes numpy array directly to faster-whisper (no temp file)."""
        import io

        import soundfile as sf

        loop = asyncio.get_event_loop()

        # Decode WAV bytes to float32 numpy array in one shot
        audio_np, sr = sf.read(io.BytesIO(audio_data), dtype="float32")

        # Ensure mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Resample to 16 kHz if needed (Whisper expects 16kHz)
        if sr != _TARGET_SR:
            # Scale to int16 range, resample, then normalize back to float32
            audio_int16 = (audio_np * 32767.0).astype(np.int16)
            audio_np = _resample(audio_int16, sr, _TARGET_SR).astype(np.float32) / 32768.0

        def _transcribe():
            segments, _info = self._model.transcribe(
                audio_np, beam_size=self.beam_size, language=self.language
            )
            return " ".join(seg.text.strip() for seg in segments)

        text = await loop.run_in_executor(None, _transcribe)
        return text

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """Accumulate audio stream and transcribe when stream ends."""
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        combined = b"".join(chunks)
        text = await self.transcribe(combined)
        yield text


# ──────────────────────────────────────────────
# OpenWakeWord
# ──────────────────────────────────────────────


class OpenWakeWordEngine(WakeWordEngine):
    """Wake word detection using OpenWakeWord (ONNX inference).

    Records at the mic's NATIVE sample rate (e.g. 44100 Hz) and
    resamples to 16000 Hz in software before feeding to the model.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self.wake_word = config.wake_word
        self.sensitivity = config.sensitivity
        self.input_device = config.input_device
        self._model = None
        self._mic: _MicInput | None = None
        self._running = False
        self._native_sr: int = 44100
        self._frame_count = 0  # for periodic debug logging
        self._suppressed = False  # True while TTS is playing
        self._cooldown_until: float = 0.0  # timestamp until which detections are ignored
        self._mic_error: str | None = None  # last mic failure, so repeats are logged quietly

    async def startup(self) -> None:
        from openwakeword.model import Model

        # Detect mic native sample rate (blocking PyAudio calls run in worker threads)
        self._native_sr = await _native_rate(self.input_device)
        logger.debug("mic native sample rate=%dHz", self._native_sr)

        # Quick mic health check at startup
        if not await asyncio.to_thread(_validate_mic, self.input_device, self._native_sr):
            logger.warning(
                "mic appears silent device=%s; check Windows microphone privacy settings, "
                "the mute switch, or set voice.input_device in config.yaml",
                self.input_device if self.input_device is not None else "default",
            )
            # List available input devices to help the user
            await asyncio.to_thread(_list_input_devices)

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: Model(
                wakeword_models=[self.wake_word],
                inference_framework="onnx",
            ),
        )
        logger.info("wake word ready engine=openwakeword model=%s", self.wake_word)

    async def shutdown(self) -> None:
        self._running = False
        mic, self._mic = self._mic, None
        if mic is not None:
            await mic.close()

    async def _open_mic(self, *, refresh_rate: bool) -> None:
        """Open the wake word stream; ``refresh_rate`` re-queries a device that may have changed."""
        if refresh_rate:
            self._native_sr = await asyncio.to_thread(
                _get_mic_native_rate, self.input_device, refresh=True)
        # Native-rate samples that equal 1280 @ 16kHz (80ms)
        mic = _MicInput(self.input_device, self._native_sr, int(1280 * self._native_sr / _TARGET_SR))
        try:
            await mic.open()
        except BaseException:
            await mic.close()
            raise
        self._mic = mic

    async def _mic_failed(self, stage: str, error: OSError, delay: float) -> float:
        """Log a mic failure (first of a kind at WARNING), drop the stream, wait; return next delay."""
        text = _describe(error)
        if text == self._mic_error:
            logger.debug("mic %s failed again error=%s", stage, text)
        else:
            self._mic_error = text
            logger.warning("mic %s failed error=%s; retrying in %.0fs", stage, text, delay)
        mic, self._mic = self._mic, None
        if mic is not None:
            await mic.close()
        await _sleep(delay)
        return min(delay * 2, _REOPEN_MAX_SECONDS)

    async def _next_block(self, delay: float) -> tuple[np.ndarray | None, float]:
        """
        Next block of native-rate mic samples and the retry delay to carry forward.
        On an OSError the failure is logged, the stream dropped and the delay waited
        out; the result is then (None, longer delay) and the next call reopens the mic.
        """
        stage = "open"
        try:
            if self._mic is None:
                # After a failure the device may have changed (another headset, new rate)
                await self._open_mic(refresh_rate=self._mic_error is not None)
            stage = "read"
            audio = await self._mic.read()
        except OSError as e:
            return None, await self._mic_failed(stage, e, delay)
        if self._mic_error is not None:
            logger.info("mic recovered rate=%dHz", self._native_sr)
            self._mic_error = None
        return audio, _REOPEN_MIN_SECONDS

    async def listen(self) -> AsyncIterator[bool]:
        """
        Yield True whenever the wake word is detected.

        Records at native mic rate, resamples to 16 kHz, feeds
        1280-sample chunks to the model. A mic that cannot be opened or read
        (headset unplugged, driver reset) is closed and reopened with a growing
        delay instead of ending the loop; each failed attempt yields False.
        """
        self._running = True
        logger.debug("wake word loop started model=%s", self.wake_word)

        delay = _REOPEN_MIN_SECONDS
        try:
            while self._running:
                # Read audio at native sample rate (always consume to keep buffer clear)
                native_audio, delay = await self._next_block(delay)
                if native_audio is None:
                    yield False
                    continue

                # While suppressed (TTS playing), just drain the buffer
                if self._suppressed:
                    yield False
                    continue

                # Cooldown after TTS ends — ignore detections for a short period
                if _clock() < self._cooldown_until:
                    yield False
                    continue

                # Resample to 16 kHz
                audio_16k = _resample(native_audio, self._native_sr, _TARGET_SR)

                # Calculate RMS to prove mic is alive
                rms = np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2))

                # Run prediction
                scores = self._model.predict(audio_16k)
                score = scores.get(self.wake_word, 0)

                # Debug: log score every ~5 seconds (62 frames at 80ms each)
                self._frame_count += 1
                if self._frame_count % 62 == 0:
                    logger.debug("mic alive level=%.0f wake_score=%.4f", rms, score)

                if score >= self.sensitivity:
                    logger.debug("wake word score=%.3f", score)
                    self._frame_count = 0
                    yield True
                else:
                    yield False
        finally:
            await self.shutdown()

    def suppress(self) -> None:
        """Suppress wake word detection (call before TTS starts)."""
        self._suppressed = True
        # Reset the model state to prevent TTS audio from building up scores
        if self._model:
            self._model.reset()

    def unsuppress(self, cooldown: float = 1.0) -> None:
        """Resume wake word detection with a cooldown period after TTS."""
        self._suppressed = False
        self._cooldown_until = _clock() + cooldown
        # Reset model again so leftover TTS audio doesn't trigger false positives
        if self._model:
            self._model.reset()


# ──────────────────────────────────────────────
# Live Microphone Recording Helper
# ──────────────────────────────────────────────

# Auto-calibrated end-of-speech threshold = _CALIBRATION_FACTOR x the quiet end of the
# calibration frames, kept within these bounds (RMS of int16 samples). The floor stops
# very quiet mics from treating hiss as speech; the ceiling stops a calibration that
# already contains speech from setting a threshold the speaker never reaches.
_CALIBRATION_FACTOR = 3.0
_CALIBRATION_PERCENTILE = 25
_THRESHOLD_FLOOR = 30.0
_THRESHOLD_CEILING = 500.0


def _calibrated_threshold(levels: list[float]) -> tuple[float, float]:
    """(threshold, ambient) from per-frame RMS levels that may already contain speech."""
    ambient = max(float(np.percentile(levels, _CALIBRATION_PERCENTILE)), 1.0)
    threshold = min(max(ambient * _CALIBRATION_FACTOR, _THRESHOLD_FLOOR), _THRESHOLD_CEILING)
    return threshold, ambient


async def record_until_silence(
    sample_rate: int = 16000,
    silence_threshold: float | None = None,
    silence_duration: float = 1.5,
    max_duration: float = 15.0,
    min_speech_duration: float = 0.5,
    input_device: int | None = None,
) -> bytes:
    """
    Record microphone audio until silence is detected.
    Records at native mic rate, resamples to target, returns WAV bytes.

    If silence_threshold is None, it is calibrated from the first ~0.3s: 3x the quiet
    end (25th percentile) of those frames, within [30, 500]. Those frames are part of
    the recording, so a user who is already talking when recording starts keeps their
    first words. The stream is always released, including when opening or reading fails.
    """
    import soundfile as sf

    native_sr = await _native_rate(input_device)
    mic = _MicInput(input_device, native_sr, _chunk_frames(native_sr, sample_rate))

    frames_16k: list[np.ndarray] = []
    silent_chunks = 0
    speech_detected = False
    threshold = silence_threshold
    max_silent = int(silence_duration * sample_rate / 1024)
    max_chunks = max(int(max_duration * sample_rate / 1024), 1)
    min_speech_chunks = int(min_speech_duration * sample_rate / 1024)

    def judge(level: float) -> bool:
        """Count one frame as silence or speech; True once the utterance is over."""
        nonlocal silent_chunks, speech_detected
        if level < threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
            speech_detected = True
        # Only stop for silence AFTER we've heard actual speech
        # and recorded at least min_speech_duration
        return (speech_detected
                and silent_chunks >= max_silent
                and len(frames_16k) > min_speech_chunks)

    try:
        try:
            await mic.open()
        except OSError:
            _MIC_RATE_CACHE.pop(input_device, None)  # the device may have changed; re-query
            raise

        calibration_levels: list[float] = []
        if threshold is None:
            for _ in range(min(max(int(0.3 * sample_rate / 1024), 3), max_chunks)):
                audio = _resample(await mic.read(), native_sr, sample_rate)
                frames_16k.append(audio)
                calibration_levels.append(_rms(audio))
            threshold, ambient = _calibrated_threshold(calibration_levels)
            logger.debug("silence threshold calibrated threshold=%.0f ambient=%.0f",
                         threshold, ambient)

        logger.debug("recording native_rate=%dHz target_rate=%dHz silence_threshold=%.0f",
                     native_sr, sample_rate, threshold)

        # Judge the calibration frames now that the threshold is known.
        finished = False
        for level in calibration_levels:
            if judge(level):
                finished = True
                break
        while not finished and len(frames_16k) < max_chunks:
            audio = _resample(await mic.read(), native_sr, sample_rate)
            frames_16k.append(audio)
            finished = judge(_rms(audio))
    finally:
        await mic.close()

    # Convert to WAV bytes
    all_audio = np.concatenate(frames_16k)
    buf = io.BytesIO()
    sf.write(buf, all_audio, sample_rate, format="WAV")
    return buf.getvalue()


# ──────────────────────────────────────────────
# Barge-in speech monitor
# ──────────────────────────────────────────────


async def wait_for_speech(
    input_device: int | None = None,
    sample_rate: int = 16000,
    cancel_event: asyncio.Event | None = None,
    threshold_factor: float = 3.0,
    sustain_ms: int = 350,
    max_duration: float = 180.0,
) -> bool:
    """
    Monitor the microphone while TTS is playing; return True as soon as
    sustained user speech is detected (barge-in), or False when
    cancel_event is set (playback finished) or max_duration elapses.

    The baseline is calibrated from the first ~0.5s of monitoring, which
    already includes speaker bleed from the TTS playback — so the trigger
    threshold sits relative to whatever the mic hears during speech output,
    not to a silent room.
    """
    native_sr = await _native_rate(input_device)
    mic = _MicInput(input_device, native_sr, _chunk_frames(native_sr, sample_rate))
    chunk_ms = 1024 / sample_rate * 1000

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    async def _level() -> float:
        return _rms(_resample(await mic.read(), native_sr, sample_rate))

    try:
        try:
            await mic.open()
        except OSError:
            _MIC_RATE_CACHE.pop(input_device, None)  # the device may have changed; re-query
            raise

        # --- Calibrate baseline (includes TTS speaker bleed) ---
        calib_chunks = max(int(0.5 * sample_rate / 1024), 3)
        levels: list[float] = []
        for _ in range(calib_chunks):
            if _cancelled():
                return False
            levels.append(await _level())
        baseline = max(float(np.mean(levels)), 1.0)
        threshold = max(baseline * threshold_factor, 100.0)
        logger.debug("barge-in monitor baseline=%.0f threshold=%.0f", baseline, threshold)

        sustain_chunks = max(int(sustain_ms / chunk_ms), 1)
        max_chunks = int(max_duration * sample_rate / 1024)
        loud_chunks = 0

        for _ in range(max_chunks):
            if _cancelled():
                return False
            rms = await _level()

            loud_chunks = loud_chunks + 1 if rms > threshold else 0
            if loud_chunks >= sustain_chunks:
                logger.debug("barge-in speech detected rms=%.0f", rms)
                return True
        return False
    finally:
        await mic.close()
