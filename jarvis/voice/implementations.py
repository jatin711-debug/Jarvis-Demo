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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np

from jarvis.utils.logger import get_logger
from jarvis.voice.engines import STTEngine, TTSEngine, WakeWordEngine

if TYPE_CHECKING:
    from jarvis.utils.config import VoiceConfig

logger = get_logger(__name__)

# Target sample rate for STT and wake word models
_TARGET_SR = 16000


def _resample(audio_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample int16 audio from orig_sr to target_sr using soxr (fast) or scipy (fallback)."""
    if orig_sr == target_sr:
        return audio_int16
    # Try soxr first — 5-10x faster than scipy, same quality
    try:
        import soxr
        resampled = soxr.resample(audio_int16.astype(np.float32), orig_sr, target_sr)
        return resampled.astype(np.int16)
    except ImportError:
        pass
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(audio_int16.astype(np.float64), up, down)
    return resampled.astype(np.int16)


def _get_mic_native_rate(device_index: int | None = None) -> int:
    """Query the native sample rate of the default input device."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        if device_index is not None:
            info = pa.get_device_info_by_index(device_index)
        else:
            info = pa.get_default_input_device_info()
        logger.debug("input device index=%d name=%s", info["index"], info["name"])
        return int(info.get("defaultSampleRate", 44100))
    finally:
        pa.terminate()


def _validate_mic(device_index: int | None, sample_rate: int, duration: float = 0.5) -> bool:
    """Quick mic capture test — returns True if audio level is above noise floor."""
    import pyaudio
    pa = pyaudio.PyAudio()
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
            audio = np.frombuffer(raw, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            peak_rms = max(peak_rms, rms)
        stream.stop_stream()
        stream.close()
        return peak_rms > 5.0
    except Exception as e:
        logger.warning("mic validation failed device=%s error=%s", device_index, e)
        return False
    finally:
        pa.terminate()


# ──────────────────────────────────────────────
# Gapless audio playback
# ──────────────────────────────────────────────


class _GaplessPlayer:
    """
    Persistent sounddevice output stream for TTS playback.

    The old per-chunk sd.play()/sd.wait() pattern closed and reopened the
    audio stream for every chunk and sentence — the source of audible gaps
    and clicks, especially over Bluetooth. This keeps ONE stream open with
    a high-latency buffer and writes chunks into it continuously.
    """

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._stream = None
        self._sr: int | None = None
        self._written = 0.0  # seconds of audio queued since stream start
        self._aborted = False

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Queue audio for playback (recreates the stream if rate changed)."""
        import sounddevice as sd
        if self._stream is None or self._stream.closed or sample_rate != self._sr:
            self.close()
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                latency="high",  # larger buffer — survives Bluetooth jitter
                device=self._device,
            )
            self._stream.start()
            self._sr = sample_rate
            self._written = 0.0
        self._stream.write(np.ascontiguousarray(audio, dtype=np.float32))
        self._written += len(audio) / sample_rate

    def drain(self, timeout: float = 60.0) -> None:
        """Block until everything queued so far has actually been played."""
        if self._stream is None or self._stream.closed:
            return
        import time
        deadline = time.monotonic() + timeout
        while (not self._aborted
               and self._stream.time < self._written
               and time.monotonic() < deadline):
            time.sleep(0.02)

    def reset_abort(self) -> None:
        """Clear the aborted flag before starting a new utterance."""
        self._aborted = False

    def abort(self) -> None:
        """Stop playback immediately (barge-in). Buffered audio is dropped."""
        self._aborted = True
        if self._stream is not None and not self._stream.closed:
            try:
                self._stream.abort()
            except Exception as e:
                logger.debug("audio stream abort failed error=%s", e)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as e:
                logger.debug("audio stream close failed error=%s", e)
            self._stream = None
            self._written = 0.0


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
        self._player.close()

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
                if self._player._aborted:
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
        self.sample_rate: int = 22050  # updated from model config at startup
        self._player = _GaplessPlayer(device=config.output_device)

    async def startup(self) -> None:
        from piper import PiperVoice

        if not os.path.exists(self._model_path):
            await self._download_voice_model()

        loop = asyncio.get_event_loop()

        def _load():
            return PiperVoice.load(self._model_path)

        self._voice = await loop.run_in_executor(None, _load)
        self.sample_rate = self._voice.config.sample_rate
        logger.info(
            "tts ready engine=piper model=%s sample_rate=%d",
            os.path.basename(self._model_path), self.sample_rate,
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
        self._player.close()

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes."""
        import soundfile as sf

        loop = asyncio.get_event_loop()

        def _synth():
            chunks = list(self._voice.synthesize(text))
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
            for chunk in self._voice.synthesize(text):
                if self._player._aborted:
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
        self._pyaudio = None
        self._stream = None
        self._running = False
        self._native_sr: int = 44100
        self._frame_count = 0  # for periodic debug logging
        self._suppressed = False  # True while TTS is playing
        self._cooldown_until: float = 0.0  # timestamp until which detections are ignored

    async def startup(self) -> None:
        from openwakeword.model import Model

        # Detect mic native sample rate
        self._native_sr = _get_mic_native_rate(self.input_device)
        logger.debug("mic native sample rate=%dHz", self._native_sr)

        # Quick mic health check at startup
        if not _validate_mic(self.input_device, self._native_sr):
            logger.warning(
                "mic appears silent device=%s; check Windows microphone privacy settings, "
                "the mute switch, or set voice.input_device in config.yaml",
                self.input_device if self.input_device is not None else "default",
            )
            # List available input devices to help the user
            import pyaudio
            pa = pyaudio.PyAudio()
            try:
                logger.info("available input devices:")
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        logger.info(
                            "  input device index=%d name=%s channels=%d rate=%d",
                            i, info["name"], info["maxInputChannels"],
                            int(info["defaultSampleRate"]),
                        )
            finally:
                pa.terminate()

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
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug("mic stream close failed error=%s", e)
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

    async def listen(self) -> AsyncIterator[bool]:
        """
        Yield True whenever the wake word is detected.

        Records at native mic rate, resamples to 16 kHz, feeds
        1280-sample chunks to the model.
        """
        import pyaudio

        self._pyaudio = pyaudio.PyAudio()

        # Calculate how many native-rate samples equal 1280 @ 16kHz (80ms)
        native_chunk = int(1280 * self._native_sr / _TARGET_SR)

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._native_sr,
            input=True,
            frames_per_buffer=native_chunk,
            input_device_index=self.input_device,
        )
        self._running = True

        loop = asyncio.get_event_loop()
        logger.debug("wake word loop started model=%s", self.wake_word)

        try:
            while self._running:
                # Read audio at native sample rate (always consume to keep buffer clear)
                raw = await loop.run_in_executor(
                    None, lambda: self._stream.read(native_chunk, exception_on_overflow=False)
                )

                # While suppressed (TTS playing), just drain the buffer
                if self._suppressed:
                    yield False
                    continue

                # Cooldown after TTS ends — ignore detections for a short period
                import time as _time
                if _time.monotonic() < self._cooldown_until:
                    yield False
                    continue

                native_audio = np.frombuffer(raw, dtype=np.int16)

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
        import time as _time
        self._suppressed = False
        self._cooldown_until = _time.monotonic() + cooldown
        # Reset model again so leftover TTS audio doesn't trigger false positives
        if self._model:
            self._model.reset()


# ──────────────────────────────────────────────
# Live Microphone Recording Helper
# ──────────────────────────────────────────────


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

    If silence_threshold is None, calibrates automatically from the first
    0.3s of ambient noise (sets threshold to 3x ambient RMS).
    """
    import pyaudio
    import soundfile as sf

    native_sr = _get_mic_native_rate(input_device)
    chunk_native = int(1024 * native_sr / sample_rate) if native_sr != sample_rate else 1024

    loop = asyncio.get_event_loop()
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=native_sr,
        input=True,
        frames_per_buffer=chunk_native,
        input_device_index=input_device,
    )

    frames_16k: list[np.ndarray] = []
    silent_chunks = 0
    speech_detected = False
    max_silent = int(silence_duration * sample_rate / 1024)
    max_chunks = int(max_duration * sample_rate / 1024)
    min_speech_chunks = int(min_speech_duration * sample_rate / 1024)

    # --- Auto-calibrate silence threshold from ambient noise ---
    if silence_threshold is None:
        calibration_chunks = int(0.3 * sample_rate / 1024)  # 0.3s
        ambient_levels = []
        for _ in range(max(calibration_chunks, 3)):
            raw = await loop.run_in_executor(
                None, lambda: stream.read(chunk_native, exception_on_overflow=False)
            )
            audio = _resample(np.frombuffer(raw, dtype=np.int16), native_sr, sample_rate)
            ambient_levels.append(float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))))
        ambient_rms = max(np.mean(ambient_levels), 1.0)
        silence_threshold = ambient_rms * 3.0
        # Ensure a reasonable minimum so we don't get stuck on very quiet mics
        silence_threshold = max(silence_threshold, 30.0)
        logger.debug("silence threshold calibrated threshold=%.0f ambient=%.0f", silence_threshold, ambient_rms)

    logger.debug("recording native_rate=%dHz target_rate=%dHz silence_threshold=%.0f", native_sr, sample_rate, silence_threshold)

    try:
        for _ in range(max_chunks):
            raw = await loop.run_in_executor(
                None, lambda: stream.read(chunk_native, exception_on_overflow=False)
            )
            native_audio = np.frombuffer(raw, dtype=np.int16)
            audio_16k = _resample(native_audio, native_sr, sample_rate)
            frames_16k.append(audio_16k)

            rms = np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2))

            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
                speech_detected = True

            # Only stop for silence AFTER we've heard actual speech
            # and recorded at least min_speech_duration
            if (speech_detected
                    and silent_chunks >= max_silent
                    and len(frames_16k) > min_speech_chunks):
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

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
    import pyaudio

    native_sr = _get_mic_native_rate(input_device)
    chunk_native = int(1024 * native_sr / sample_rate) if native_sr != sample_rate else 1024
    chunk_ms = 1024 / sample_rate * 1000

    loop = asyncio.get_event_loop()
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=native_sr,
        input=True,
        frames_per_buffer=chunk_native,
        input_device_index=input_device,
    )

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    try:
        # --- Calibrate baseline (includes TTS speaker bleed) ---
        calib_chunks = max(int(0.5 * sample_rate / 1024), 3)
        levels: list[float] = []
        for _ in range(calib_chunks):
            if _cancelled():
                return False
            raw = await loop.run_in_executor(
                None, lambda: stream.read(chunk_native, exception_on_overflow=False)
            )
            audio = _resample(np.frombuffer(raw, dtype=np.int16), native_sr, sample_rate)
            levels.append(float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))))
        baseline = max(float(np.mean(levels)), 1.0)
        threshold = max(baseline * threshold_factor, 100.0)
        logger.debug("barge-in monitor baseline=%.0f threshold=%.0f", baseline, threshold)

        sustain_chunks = max(int(sustain_ms / chunk_ms), 1)
        max_chunks = int(max_duration * sample_rate / 1024)
        loud_chunks = 0

        for _ in range(max_chunks):
            if _cancelled():
                return False
            raw = await loop.run_in_executor(
                None, lambda: stream.read(chunk_native, exception_on_overflow=False)
            )
            audio = _resample(np.frombuffer(raw, dtype=np.int16), native_sr, sample_rate)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

            loud_chunks = loud_chunks + 1 if rms > threshold else 0
            if loud_chunks >= sustain_chunks:
                logger.debug("barge-in speech detected rms=%.0f", rms)
                return True
        return False
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
