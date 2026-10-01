"""Short audio cues that tell the user what Jarvis is doing."""

from __future__ import annotations

import asyncio

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 24000
_FADE_SECONDS = 0.01


async def _play_tone(frequency: float, duration: float, amplitude: float,
                     device: int | None = None) -> None:
    """Play a sine tone on ``device`` (None = system default). Cues are cosmetic, so any
    audio failure is ignored."""
    try:
        import numpy as np
        import sounddevice as sd

        t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), False)
        tone = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
        fade = int(_SAMPLE_RATE * _FADE_SECONDS)  # avoid clicks
        tone[:fade] *= np.linspace(0, 1, fade).astype(np.float32)
        tone[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: (sd.play(tone, _SAMPLE_RATE, device=device), sd.wait()))
    except Exception as e:
        logger.debug("audio cue failed device=%s error=%s", device, e)


async def play_wake_cue(device: int | None = None) -> None:
    """Wake word detected."""
    await _play_tone(880, 0.2, 0.3, device)


async def play_listening_cue(device: int | None = None) -> None:
    """Ready for a follow-up command."""
    await _play_tone(660, 0.1, 0.15, device)
