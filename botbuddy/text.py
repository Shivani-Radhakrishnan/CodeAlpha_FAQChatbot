"""Text normalisation helpers shared by every matching method.

Everything here is deterministic and needs no downloaded NLTK corpora
(the Porter stemmer is pure Python), so it runs the same locally and in CI.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Order matters: specific forms before the generic suffix rules.
_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"\bain't\b", "is not"),
    (r"\blet's\b", "let us"),
    (r"\bi'm\b", "i am"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ll\b", " will"),
    (r"'ve\b", " have"),
    (r"'d\b", " would"),
    (r"\b(what|where|who|how|when|why|it|that|there|here|he|she)'s\b", r"\1 is"),
    (r"'s\b", ""),  # possessive: "today's" -> "today"
)

#: Small closed-class stop-word list. Deliberately excludes words that carry
#: intent in this domain ("help", "hi", "bye", "human", "new", ...), which
#: generic lists such as scikit-learn's sometimes contain.
STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if so of to in on at by for from with about as into
    is am are was were be been being do does did doing have has had having
    i me my mine we us our you your yours it its this that these those there
    what which who whom how when where why can could would should will shall
    may might must please just any some get got""".split()
)


def normalize(text: str) -> str:
    """Lower-case, fold unicode quotes/accents and expand English contractions."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    for pattern, repl in _CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    return " ".join(_TOKEN_RE.findall(text))


def tokenize(text: str) -> list[str]:
    """Split *text* into normalised word tokens (stop words kept)."""
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    """Tokens with stop words removed; falls back to all tokens if none remain."""
    tokens = tokenize(text)
    content = [t for t in tokens if t not in STOPWORDS]
    return content or tokens


@lru_cache(maxsize=4096)
def stem(token: str) -> str:
    """Porter-stem a single token (cached; stemming is the hot path)."""
    return _STEMMER.stem(token)


def stemmed_text(text: str) -> str:
    """Normalised text with every token stemmed, ready for a vectorizer."""
    return " ".join(stem(t) for t in tokenize(text))
