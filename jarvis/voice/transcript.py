"""Pure text/audio heuristics for rejecting bad transcripts and reading approvals."""

from __future__ import annotations

import io
import re

from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

# Known Whisper hallucination phrases on near-silent audio
_HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "you", "bye", "the end", "see you next time", "subscribe",
    "please subscribe", "like and subscribe", "goodbye",
    "thank you so much", "thanks for listening",
}

# Repeated words / stuttering patterns that indicate noise/gibberish
_GIBBERISH_RE = re.compile(r"(\b\w+\b)(\W+\1\b){2,}")  # same word 3+ times

# RMS (int16 scale) below which a stock hallucination phrase is treated as silence.
_SILENCE_RMS = 100


def is_hallucination(text: str, audio_bytes: bytes) -> bool:
    """Detect likely Whisper hallucinations on very short / quiet audio."""
    import numpy as np
    import soundfile as sf

    cleaned = text.strip().lower().rstrip(".!?,")
    if cleaned not in _HALLUCINATION_PHRASES:
        return False
    # Verify the audio is actually very quiet (confirms hallucination)
    try:
        audio_data, _ = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
    except Exception:
        return False
    if rms < _SILENCE_RMS:
        logger.debug("filtered hallucination text=%r rms=%.0f", text, rms)
        return True
    return False


def is_garbage(text: str) -> bool:
    """Detect nonsensical / stuttered transcripts from ambient noise."""
    stripped = text.strip()
    if not stripped:
        return True

    # Heavy repetition / stuttering (e.g. "oh. oh. oh.")
    if _GIBBERISH_RE.search(stripped):
        logger.debug("filtered garbage text=%r reason=repetition", text)
        return True

    # No length filtering: short commands like "open notepad" are legitimate.
    # Whisper hallucinations are handled separately by is_hallucination().
    return False


def is_explicit_yes(text: str) -> bool:
    """Voice approval accepts only the requested single-word answer."""
    return text.strip().casefold().rstrip(".!?").strip() == "yes"
