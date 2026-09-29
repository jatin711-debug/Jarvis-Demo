"""Unit tests for Whisper transcript filtering in jarvis.voice.transcript."""

import io

import numpy as np
import soundfile as sf

from jarvis.voice.transcript import is_garbage, is_hallucination


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def _quiet_wav() -> bytes:
    return _wav_bytes(np.zeros(16000, dtype=np.int16))


def _loud_wav() -> bytes:
    t = np.linspace(0, 1, 16000)
    samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    return _wav_bytes(samples)


class TestIsGarbage:
    def test_empty_string(self):
        assert is_garbage("") is True

    def test_whitespace_only(self):
        assert is_garbage("   \n  ") is True

    def test_repeated_word_with_punctuation(self):
        assert is_garbage("oh. oh. oh.") is True

    def test_repeated_word_plain(self):
        assert is_garbage("yeah yeah yeah") is True

    def test_normal_command_not_garbage(self):
        assert is_garbage("open notepad") is False

    def test_short_command_not_garbage(self):
        assert is_garbage("take screenshot") is False

    def test_two_repeats_not_garbage(self):
        assert is_garbage("thank you thank you") is False


class TestIsHallucination:
    def test_known_phrase_with_quiet_audio(self):
        assert is_hallucination("thank you", _quiet_wav()) is True

    def test_known_phrase_with_loud_audio_kept(self):
        # Real speech saying "thank you" must not be filtered.
        assert is_hallucination("thank you", _loud_wav()) is False

    def test_non_hallucination_phrase_kept(self):
        assert is_hallucination("open notepad", _quiet_wav()) is False

    def test_punctuation_and_case_normalized(self):
        assert is_hallucination("Thank You!", _quiet_wav()) is True

    def test_invalid_audio_does_not_crash(self):
        assert is_hallucination("thank you", b"not audio") is False
