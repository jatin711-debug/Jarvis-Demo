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
        return int(info.get("defaultSampleRate", 44100))
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

    async def startup(self) -> None:
        from openwakeword.model import Model

        # Detect mic native sample rate
        self._native_sr = _get_mic_native_rate(self.input_device)
        logger.info("🎤 Mic native sample rate: %d Hz", self._native_sr)

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
                # Read audio at native sample rate
                raw = await loop.run_in_executor(
                    None, lambda: self._stream.read(native_chunk, exception_on_overflow=False)
                )
                native_audio = np.frombuffer(raw, dtype=np.int16)

                # Resample to 16 kHz
                audio_16k = _resample(native_audio, self._native_sr, _TARGET_SR)

                # Run prediction
                scores = self._model.predict(audio_16k)
                score = scores.get(self.wake_word, 0)

                # Debug: log score every ~5 seconds (62 frames at 80ms each)
                self._frame_count += 1
                if self._frame_count % 62 == 0:
                    logger.debug("Wake word score: %.4f (threshold: %.2f)", score, self.sensitivity)

                if score >= self.sensitivity:
                    logger.info("🔔 Wake word detected! (score=%.3f)", score)
                    self._frame_count = 0
                    yield True
                else:
                    yield False
        finally:
            await self.shutdown()


# ──────────────────────────────────────────────
# Live Microphone Recording Helper
# ──────────────────────────────────────────────


async def record_until_silence(
    sample_rate: int = 16000,
    silence_threshold: float = 300.0,
    silence_duration: float = 1.5,
    max_duration: float = 15.0,
    input_device: int | None = None,
) -> bytes:
    """
    Record microphone audio until silence is detected.
    Records at native mic rate, resamples to target, returns WAV bytes.
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
    max_silent = int(silence_duration * sample_rate / 1024)
    max_chunks = int(max_duration * sample_rate / 1024)

    logger.debug("Recording at %d Hz, resampling to %d Hz", native_sr, sample_rate)

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

            if silent_chunks >= max_silent and len(frames_16k) > max_silent:
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
