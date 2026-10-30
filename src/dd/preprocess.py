"""Text cleaning for social media posts (Polish + English)."""
from __future__ import annotations

import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)

_PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_PL_STOP = {"się", "nie", "że", "jest", "to", "na", "w", "z", "do", "jak", "ale", "mnie", "mi", "ja"}
_EN_STOP = {"the", "and", "is", "to", "of", "in", "that", "it", "i", "my", "me", "was", "but"}


def clean_text(text: str) -> str:
    """Remove HTML, URLs and user mentions (privacy); normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (unicode-aware, keeps Polish diacritics)."""
    return _TOKEN.findall(text.lower())


def detect_lang(text: str) -> str:
    """Very rough PL/EN heuristic. Replace with fastText/langdetect for real work."""
    toks = tokenize(text)
    if not toks:
        return "unk"
    pl = sum(t in _PL_STOP for t in toks) + 2 * sum(any(c in _PL_CHARS for c in t) for t in toks)
    en = sum(t in _EN_STOP for t in toks)
    if pl == en:
        return "unk"
    return "pl" if pl > en else "en"
