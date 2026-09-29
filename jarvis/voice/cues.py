"""Short audio cues that tell the user what Jarvis is doing."""

from __future__ import annotations

import asyncio

_SAMPLE_RATE = 24000
_FADE_SECONDS = 0.01


async def _play_tone(frequency: float, duration: float, amplitude: float) -> None:
    """Play a sine tone. Cues are cosmetic, so any audio failure is ignored."""
    try:
        import numpy as np
        import sounddevice as sd

        t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), False)
        tone = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
        fade = int(_SAMPLE_RATE * _FADE_SECONDS)  # avoid clicks
        tone[:fade] *= np.linspace(0, 1, fade).astype(np.float32)
        tone[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: (sd.play(tone, _SAMPLE_RATE), sd.wait()))
    except Exception:
        pass


async def play_wake_cue() -> None:
    """Wake word detected."""
    await _play_tone(880, 0.2, 0.3)


async def play_listening_cue() -> None:
    """Ready for a follow-up command."""
    await _play_tone(660, 0.1, 0.15)
