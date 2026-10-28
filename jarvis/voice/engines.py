"""
JARVIS Voice Pipeline Stubs
=============================
Placeholder interfaces for STT, TTS, and Wake Word.
Full implementations will be added in Phase 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Abstract interfaces
# ──────────────────────────────────────────────


class STTEngine(ABC):
    """Speech-to-Text engine interface."""

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """Stream transcription results."""
        ...


class TTSEngine(ABC):
    """Text-to-Speech engine interface."""

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV)."""
        ...

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize and play audio through speakers."""
        ...

    def abort(self) -> None:
        """Stop any ongoing playback immediately (barge-in). Optional."""
        return


class WakeWordEngine(ABC):
    """Wake word detection engine interface."""

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def listen(self) -> AsyncIterator[bool]:
        """Yield True when wake word is detected."""
        ...


# ──────────────────────────────────────────────
# Placeholder implementations (for Phase 0 testing)
# ──────────────────────────────────────────────


class DummySTT(STTEngine):
    """Placeholder STT — just returns keyboard input."""

    async def startup(self) -> None:
        logger.info("stt ready engine=dummy")

    async def shutdown(self) -> None:
        pass

    async def transcribe(self, audio_data: bytes) -> str:
        return "(dummy transcription)"

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        yield "(dummy transcription)"


class DummyTTS(TTSEngine):
    """Placeholder TTS — just prints to console."""

    async def startup(self) -> None:
        logger.info("tts ready engine=dummy")

    async def shutdown(self) -> None:
        pass

    async def synthesize(self, text: str) -> bytes:
        logger.info("tts say chars=%d", len(text))
        return b""

    async def speak(self, text: str) -> None:
        logger.info("tts say chars=%d", len(text))


class DummyWakeWord(WakeWordEngine):
    """Placeholder wake word — always active."""

    async def startup(self) -> None:
        logger.info("wake word ready engine=dummy")

    async def shutdown(self) -> None:
        pass

    async def listen(self) -> AsyncIterator[bool]:
        yield True
