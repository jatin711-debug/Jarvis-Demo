"""Hold back streamed text that might be an inline (fallback) tool call."""

from __future__ import annotations

# Markers that may begin an inline tool call in streamed text.
STREAM_MARKERS = ("<|", "```json", '{"tool"')


class StreamGuard:
    """
    Feed streamed tokens in, get back the text that is safe to speak.

    Text that could be the start of a tool call is held until it is clear
    whether it is one, so call markup is never spoken. Once a marker is seen,
    everything after it is held. ``held`` is what remains at the end of the
    stream; the caller decides whether it was a tool call or ordinary text.
    """

    def __init__(self) -> None:
        self._emitted: list[str] = []
        self._held = ""
        self._suspicious = False

    def feed(self, token: str) -> str:
        """Add a token; return any text that is now safe to emit."""
        self._held += token
        if self._suspicious:
            return ""

        cut = min((i for i in (self._held.find(m) for m in STREAM_MARKERS) if i != -1), default=-1)
        if cut != -1:
            safe, self._held = self._held[:cut], self._held[cut:]
            self._suspicious = True
        else:
            # Retain any suffix that could become a marker in the next chunk,
            # not merely the last two characters.
            hold = max((n for marker in STREAM_MARKERS for n in range(1, len(marker))
                        if self._held.endswith(marker[:n])), default=0)
            safe = self._held[:-hold] if hold else self._held
            self._held = self._held[-hold:] if hold else ""

        if safe:
            self._emitted.append(safe)
        return safe

    @property
    def held(self) -> str:
        """Text withheld so far."""
        return self._held

    @property
    def text(self) -> str:
        """Everything received: emitted text plus what is still held."""
        return "".join(self._emitted) + self._held
