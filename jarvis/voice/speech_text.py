"""Turn a model reply into text worth speaking: no markdown markup, no raw URLs."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# "---", "***", "* * *" on a line of their own.
_RULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
# Heading, quote and bullet markers at the start of a line.
_LINE_MARKER = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+•][ \t]+)", re.MULTILINE)
# [text](url) and ![alt](url): keep the words.
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
# A web address, not counting punctuation that ends the sentence.
_URL = re.compile(r"https?://[^\s)>\]]*[^\s)>\].,;:!?]")
# **bold**, __bold__, *italic* (not the * in "2 * 3" or "a*b").
_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_EMPHASIS = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
# What is left of a marker whose other half was in another sentence.
_STRAY = re.compile(r"\*\*|__|~~|`")


def _host(match: re.Match[str]) -> str:
    """A long address is not worth hearing; its site is."""
    return urlparse(match.group()).netloc.removeprefix("www.") or "a link"


def speakable(text: str) -> str:
    """``text`` as it should be said aloud: markup removed, words kept, one line.

    Works on one sentence at a time (a reply is spoken as it streams), so a marker
    whose partner sits in another sentence is simply dropped.
    """
    text = _RULE.sub("", text)
    text = _LINE_MARKER.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(_host, text)
    text = _STRONG.sub(r"\2", text)
    text = _EMPHASIS.sub(r"\1", text)
    text = _STRAY.sub("", text)
    return " ".join(text.split())
