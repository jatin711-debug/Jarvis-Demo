"""Request and response policy for the direct browser-search path.

The browser owns retrieval; this module only decides when that path applies and
formats the evidence returned by it. It never claims to have read result pages.
"""

from __future__ import annotations

import re

from jarvis.agent.routing import is_web_research

_AFFIRMATIVE = re.compile(r"\s*(?:yes|yeah|yep|sure|please do)[.!?\s]*", re.I)
_OFFER = re.compile(r"\b(?:would you like|shall i|want me to)\b[^?]*\?", re.I)
_SEARCH_OFFER = re.compile(r"\b(?:search|research|look|do that|do it)\b", re.I)


def research_query(message: str, history: list[dict[str, str]] | None) -> str | None:
    """Find an explicit research request, including acceptance of a recent offer."""
    if is_web_research(message):
        return message
    if not _AFFIRMATIVE.fullmatch(message) or not history or len(history) < 2:
        return None

    previous_user, previous_reply = history[-2:]
    offer = previous_reply.get("content", "")
    if (previous_user.get("role") == "user" and previous_reply.get("role") == "assistant"
            and is_web_research(previous_user.get("content", ""))
            and _OFFER.search(offer) and _SEARCH_OFFER.search(offer)):
        return previous_user["content"]
    return None


def format_search_results(result: str | None) -> str:
    """Return up to three snippets and links without inventing article contents."""
    blocks = (result or "").split("\n\n")[1:4]
    if not blocks:
        return "I couldn't retrieve search results. Please try again."
    return "I found these search results. These are snippets, not full articles:\n\n" + "\n\n".join(blocks)
