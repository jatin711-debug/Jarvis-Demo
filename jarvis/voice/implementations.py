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
import numpy as np
from typing import AsyncIterator, TYPE_CHECKING

from jarvis.voice.engines import STTEngine, TTSEngine, WakeWordEngine
from jarvis.utils.logger import get_logger

if TYPE_CHECKING:
    from jarvis.utils.config import VoiceConfig

logger = get_logger(__name__)

# Target sample rate for STT and wake word models
_TARGET_SR = 16000


def _resample(audio_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample int16 audio from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return audio_int16
    from scipy.signal import resample_poly
    from math import gcd
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
        logger.debug("🎤 Using input device [%d] %s", info["index"], info["name"])
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
        logger.warning("🎤 Mic validation failed for device %s: %s", device_index, e)
        return False
    finally:
        pa.terminate()


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

    async def startup(self) -> None:
        loop = asyncio.get_event_loop()

        def _init_kokoro():
            # Suppress warnings inside this thread (thread pools don't inherit filters)
            import warnings
            warnings.simplefilter("ignore")

            from kokoro import KPipeline
            return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

        self._pipeline = await loop.run_in_executor(None, _init_kokoro)
        logger.info("🔊 Kokoro TTS ready (voice=%s, speed=%.1f)", self.voice, self.speed)

    async def shutdown(self) -> None:
        self._pipeline = None

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
        """Synthesize and play through speakers."""
        import sounddevice as sd

        loop = asyncio.get_event_loop()

        def _speak():
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
            ):
                sd.play(audio, self.sample_rate)
                sd.wait()

        await loop.run_in_executor(None, _speak)


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

    async def startup(self) -> None:
        from piper import PiperVoice

        loop = asyncio.get_event_loop()

        def _load():
            return PiperVoice.load(self._model_path)

        self._voice = await loop.run_in_executor(None, _load)
        self.sample_rate = self._voice.config.sample_rate
        logger.info(
            "🔊 Piper TTS ready (model=%s, sr=%d)",
            os.path.basename(self._model_path), self.sample_rate,
        )

    async def shutdown(self) -> None:
        self._voice = None

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
        """Synthesize and stream each sentence to speakers immediately."""
        import sounddevice as sd

        loop = asyncio.get_event_loop()

        def _speak():
            for chunk in self._voice.synthesize(text):
                audio = (
                    np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
                sd.play(audio, self.sample_rate)
                sd.wait()

        await loop.run_in_executor(None, _speak)


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
        logger.info("📝 Whisper STT ready (model=%s, compute=%s)", self.model_size, self.compute_type)

    async def shutdown(self) -> None:
        self._model = None

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes (WAV) → text."""
        import tempfile

        loop = asyncio.get_event_loop()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        def _transcribe():
            segments, _info = self._model.transcribe(
                tmp_path, beam_size=self.beam_size, language=self.language
            )
            return " ".join(seg.text.strip() for seg in segments)

        try:
            text = await loop.run_in_executor(None, _transcribe)
        finally:
            os.unlink(tmp_path)

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
        logger.info("🎤 Mic native sample rate: %d Hz", self._native_sr)

        # Quick mic health check at startup
        if not _validate_mic(self.input_device, self._native_sr):
            logger.warning(
                "⚠️  Mic device %s appears SILENT (Level ≈ 0). "
                "Check: Windows mic privacy settings, mute switch, "
                "or set 'input_device' in config.yaml to a working device index.",
                self.input_device if self.input_device is not None else "(default)",
            )
            # List available input devices to help the user
            import pyaudio
            pa = pyaudio.PyAudio()
            try:
                logger.info("🎤 Available input devices:")
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        logger.info(
                            "    [%d] %s (ch=%d, rate=%d)",
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
        logger.info("👂 OpenWakeWord ready (model=%s, ONNX)", self.wake_word)

    async def shutdown(self) -> None:
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
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
        logger.info("🎙️ Listening for wake word '%s'...", self.wake_word)

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
                    logger.info("🎙️ Mic Alive | Level: %4.0f | Wake Score: %.4f", rms, score)

                if score >= self.sensitivity:
                    logger.info("🔔 Wake word detected! (score=%.3f)", score)
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
        logger.debug("Auto-calibrated silence threshold: %.0f (ambient=%.0f)", silence_threshold, ambient_rms)

    logger.debug("Recording at %d Hz, resampling to %d Hz, silence_thresh=%.0f", native_sr, sample_rate, silence_threshold)

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
