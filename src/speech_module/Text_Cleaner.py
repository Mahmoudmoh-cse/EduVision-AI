# -*- coding: utf-8 -*-

import os
import re
from typing import Optional


_SYSTEM_PROMPT = """\
You are a speech input handler for an AI educational assistant.
Your task is to clean raw transcribed text from Google Speech-to-Text API.
1. Remove filler words like um, uh, like, you know, يعني, إيه, آه
2. Fix obvious speech-to-text errors and broken words
3. Preserve the student's original intent exactly
4. Return ONLY the cleaned question in one line, properly punctuated
5. No explanation, no labels, no preamble\
"""


_FILLER_WORDS = [
    "um",
    "uh",
    "like",
    "you know",
    "i mean",
    "يعني",
    "إيه",
    "آه",
    "اه",
    "hmm",
    "mm",
    "mmh",
    "right",
    "so",
]


def _local_clean(raw_text: str, lang: str = "en-US") -> str:
    text = (raw_text or "").strip()

    # Remove common filler words (case-insensitive)
    for filler in _FILLER_WORDS:
        pattern = r"(?i)\b" + re.escape(filler) + r"\b"
        text = re.sub(pattern, " ", text)

    # Collapse repeated whitespace and tidy punctuation spacing
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!،؟])", r"\1", text)

    # Capitalize English sentence start when appropriate
    if text and "a" <= text[0] <= "z":
        text = text[0].upper() + text[1:]

    # Ensure sentence ends with punctuation. Try to guess question vs statement.
    if text and text[-1] not in ".!?؟":
        lower = text.lower()
        qwords_en = ("what", "how", "why", "when", "where", "who", "which", "is", "are", "do", "does", "did", "can", "could")
        qwords_ar = ("ما", "متى", "لماذا", "كيف", "أين", "هل", "ماذا")
        if any(lower.startswith(q + " ") for q in qwords_en) or any(text.startswith(q) for q in qwords_ar):
            text = text + "?"
        else:
            text = text + "."

    return text


def _try_anthropic_clean(raw_text: str, lang: str = "en-US") -> Optional[str]:
    """Attempt to clean using the Anthropic Python client if available.

    This function is intentionally defensive: if the installed Anthropics
    client API differs from expectations or the API key is missing, it
    returns None so the caller can fall back to the local cleaner.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except Exception:
        return None

    # Try a few possible client constructors to remain compatible
    Client = None
    for candidate in ("Client", "Anthropic", "AnthropicClient"):
        if hasattr(anthropic, candidate):
            Client = getattr(anthropic, candidate)
            break

    client = None
    try:
        if Client is not None:
            client = Client(api_key=api_key)
        else:
            # Some older/newer clients expose top-level helpers
            client = anthropic
    except Exception:
        try:
            client = anthropic.Client(api_key=api_key)
        except Exception:
            client = None

    if client is None:
        return None

    # Try common completion interfaces. Be tolerant to different SDK shapes.
    try:
        # Completions-style API
        if hasattr(client, "completions") and hasattr(client.completions, "create"):
            prompt = _SYSTEM_PROMPT + "\n\n" + f"Language: {lang}\nRaw text: {raw_text}"
            resp = client.completions.create(model="claude-sonnet-4-20250514", prompt=prompt, max_tokens_to_sample=200)
            cleaned = getattr(resp, "completion", None) or (resp.get("completion") if isinstance(resp, dict) else None)
            if cleaned:
                return cleaned.strip()

        # Chat/messages-style API
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Language: {lang}\nRaw text: {raw_text}"}],
            )

            if isinstance(resp, dict):
                content = resp.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"].strip()

                cleaned = resp.get("completion") or resp.get("result") or resp.get("output")
                if cleaned:
                    return cleaned.strip()

            cleaned = getattr(resp, "completion", None)
            if cleaned:
                return cleaned.strip()

    except Exception:
        return None

    return None


def clean_text(raw_text: str, lang: str = "en-US") -> str:
    """Public cleaning entrypoint.

    Prefer using Anthropic when configured; otherwise fall back to a
    lightweight local cleaner so the module works offline and in tests.
    """
    cleaned = _try_anthropic_clean(raw_text, lang)
    if cleaned:
        return cleaned
    return _local_clean(raw_text, lang)
