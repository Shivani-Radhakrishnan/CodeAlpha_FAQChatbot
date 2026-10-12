"""FAQ retrieval: score a user question against every known FAQ.

The module is UI-free so it can be unit-tested and evaluated offline.

Methods (all return one similarity score in ``[0, 1]`` per FAQ):

``keyword``
    Baseline. Jaccard overlap of content words (no stemming, no fuzziness).
``tfidf``
    Word-level TF-IDF cosine blended with token-set similarity - the approach
    the project originally described.
``hybrid``
    Character n-gram TF-IDF on the question (robust to typos) plus stemmed
    word TF-IDF over question *and* answer text. Lightweight default.
``embedding``
    Sentence-embedding similarity (``sentence-transformers``, CPU). Best
    quality; used automatically when the optional dependency is installed.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .text import content_tokens, normalize, stem, stemmed_text, tokenize

logger = logging.getLogger(__name__)

Status = Literal["answer", "suggest", "fallback"]


class FAQDataError(ValueError):
    """Raised when the FAQ file is missing or malformed."""


@dataclass(frozen=True)
class FAQ:
    """One question/answer pair from the knowledge base."""

    question: str
    answer: str


@dataclass(frozen=True)
class Candidate:
    """An FAQ together with its similarity score for a given query."""

    faq: FAQ
    score: float


@dataclass(frozen=True)
class MatchResult:
    """What the bot should reply to one user message.

    ``status`` is ``"answer"`` when confident, ``"suggest"`` when it is unsure
    and offers "Did you mean ...?" options, and ``"fallback"`` otherwise.
    """

    status: Status
    answer: str
    score: float
    faq: FAQ | None = None
    suggestions: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence for display in the UI."""
        return {"answer": "high", "suggest": "low", "fallback": "none"}[self.status]


def load_faqs(path: str | Path = config.FAQ_PATH) -> list[FAQ]:
    """Load FAQs from a CSV with ``Question`` and ``Answer`` columns.

    Blank rows are skipped and duplicate questions keep their first answer.

    Raises:
        FAQDataError: if the file is missing, lacks the required columns or
            contains no usable rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FAQDataError(f"FAQ file not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"Question", "Answer"} <= set(reader.fieldnames):
            raise FAQDataError(f"{path} must have 'Question' and 'Answer' columns")
        faqs: list[FAQ] = []
        seen: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                if question or answer:
                    logger.warning("Skipping incomplete FAQ row %d in %s", line_no, path)
                continue
            key = normalize(question)
            if key in seen:
                logger.warning("Skipping duplicate FAQ %r (row %d)", question, line_no)
                continue
            seen.add(key)
            faqs.append(FAQ(question, answer))
    if not faqs:
        raise FAQDataError(f"No FAQs found in {path}")
    logger.info("Loaded %d FAQs from %s", len(faqs), path)
    return faqs


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------


class Scorer(Protocol):
    """Interface every matching method implements."""

    def fit(self, faqs: Sequence[FAQ]) -> "Scorer":
        """Index the knowledge base."""

    def score(self, query: str) -> np.ndarray:
        """Return one similarity in ``[0, 1]`` per indexed FAQ."""


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class KeywordScorer:
    """Baseline: Jaccard overlap between the content words of query and FAQ."""

    def fit(self, faqs: Sequence[FAQ]) -> "KeywordScorer":
        self._docs = [set(content_tokens(f.question)) for f in faqs]
        return self

    def score(self, query: str) -> np.ndarray:
        q = set(content_tokens(query))
        return np.array([_jaccard(q, d) for d in self._docs])


class TfidfTokenScorer:
    """Word TF-IDF cosine blended with token-set (Jaccard) similarity."""

    def __init__(self, tfidf_weight: float = 0.7) -> None:
        self.tfidf_weight = tfidf_weight

    def fit(self, faqs: Sequence[FAQ]) -> "TfidfTokenScorer":
        questions = [normalize(f.question) for f in faqs]
        self._vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self._matrix = self._vec.fit_transform(questions)
        self._tokens = [set(tokenize(f.question)) for f in faqs]
        return self

    def score(self, query: str) -> np.ndarray:
        cos = cosine_similarity(self._vec.transform([normalize(query)]), self._matrix)[0]
        q = set(tokenize(query))
        tok = np.array([_jaccard(q, d) for d in self._tokens])
        return self.tfidf_weight * cos + (1 - self.tfidf_weight) * tok


class HybridScorer:
    """Character n-gram TF-IDF on the question + stemmed word TF-IDF on Q&A.

    * Character 2-4-grams (within word boundaries) tolerate typos and
      inflections: "trak my packge" still shares most n-grams with "track".
    * Stemmed content words of question *and* answer let a query match on
      vocabulary that only the answer uses ("COD", "PayPal", "recyclable").

    The 50/50 weighting was selected on the dev split of the evaluation set.
    """

    def __init__(self, char_weight: float = 0.5, ngram_range: tuple[int, int] = (2, 4)) -> None:
        if not 0.0 <= char_weight <= 1.0:
            raise ValueError("char_weight must be in [0, 1]")
        self.char_weight = char_weight
        self.ngram_range = ngram_range

    @staticmethod
    def _content_stems(text: str) -> str:
        return " ".join(stem(t) for t in content_tokens(text))

    def fit(self, faqs: Sequence[FAQ]) -> "HybridScorer":
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True
        )
        self._char_m = self._char.fit_transform([normalize(f.question) for f in faqs])
        self._word = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)
        self._word_m = self._word.fit_transform(
            [self._content_stems(f"{f.question} {f.answer}") for f in faqs]
        )
        return self

    def score(self, query: str) -> np.ndarray:
        char = cosine_similarity(self._char.transform([normalize(query)]), self._char_m)[0]
        word = cosine_similarity(self._word.transform([self._content_stems(query)]), self._word_m)[0]
        return self.char_weight * char + (1 - self.char_weight) * word


class EmbeddingScorer:
    """Sentence-embedding similarity (optional dependency).

    A query is compared with both the FAQ question and its answer; the score
    is ``max(cos(query, question), answer_weight * cos(query, answer))``, so a
    question phrased like the answer ("can I pay when the package arrives")
    still matches, while the question itself stays the primary signal.
    """

    def __init__(self, answer_weight: float = 0.8) -> None:
        self.answer_weight = answer_weight
        try:
            from botbuddy.encoder import get_encoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'embedding' method needs: pip install -r requirements-embedding.txt"
            ) from exc
        self._model = get_encoder()

    def fit(self, faqs: Sequence[FAQ]) -> "EmbeddingScorer":
        self._q_emb = self._encode([f.question for f in faqs])
        self._a_emb = self._encode([f.answer for f in faqs])
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts)

    def score(self, query: str) -> np.ndarray:
        q = self._encode([query])[0]
        sim = np.maximum(self._q_emb @ q, self.answer_weight * (self._a_emb @ q))
        return np.clip(sim, 0.0, 1.0)


SCORERS: dict[str, type] = {
    "keyword": KeywordScorer,
    "tfidf": TfidfTokenScorer,
    "hybrid": HybridScorer,
    "embedding": EmbeddingScorer,
}


def make_scorer(method: str) -> tuple[str, Scorer]:
    """Instantiate the scorer for *method*; returns ``(resolved_method, scorer)``.

    ``"auto"`` tries each entry of :data:`config.AUTO_METHODS` in order and
    falls back to the next one if it cannot be loaded (e.g. the optional
    ``sentence-transformers`` package is missing or the model download fails).
    """
    if method == "auto":
        for candidate in config.AUTO_METHODS:
            try:
                return make_scorer(candidate)
            except Exception as exc:  # noqa: BLE001 - any load failure -> next method
                logger.warning("Method %r unavailable (%s); trying the next one", candidate, exc)
        raise RuntimeError(f"None of {config.AUTO_METHODS} could be loaded")
    try:
        cls = SCORERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown method {method!r}; choose from {sorted(SCORERS)} or 'auto'"
        ) from None
    return method, cls()


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------


class FAQMatcher:
    """Answer user questions from a list of FAQs.

    Args:
        faqs: the knowledge base.
        method: one of :data:`SCORERS`, or ``"auto"`` (best available).
        threshold: minimum score to answer directly (defaults to the tuned
            value in :data:`config.ANSWER_THRESHOLDS`).
        suggest_ratio: scores in ``[threshold * suggest_ratio, threshold)``
            produce "Did you mean ...?" suggestions.
        scorer: pre-built scorer (mainly for tests / evaluation reuse).
    """

    def __init__(
        self,
        faqs: Sequence[FAQ],
        method: str = config.DEFAULT_METHOD,
        threshold: float | None = None,
        suggest_ratio: float = config.SUGGEST_RATIO,
        num_suggestions: int = config.NUM_SUGGESTIONS,
        scorer: Scorer | None = None,
    ) -> None:
        if not faqs:
            raise FAQDataError("FAQMatcher needs at least one FAQ")
        self.faqs: list[FAQ] = list(faqs)
        if scorer is None:
            method, scorer = make_scorer(method)
        self.method = method
        self.suggest_ratio = suggest_ratio
        self.set_threshold(config.ANSWER_THRESHOLDS[method] if threshold is None else threshold)
        self.num_suggestions = num_suggestions
        self._scorer = scorer.fit(self.faqs)

    def set_threshold(self, threshold: float) -> None:
        """Change the answer threshold (the suggestion band scales with it)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.suggest_threshold = threshold * self.suggest_ratio

    @classmethod
    def from_csv(cls, path: str | Path = config.FAQ_PATH, **kwargs) -> "FAQMatcher":
        """Build a matcher straight from a FAQ CSV file."""
        return cls(load_faqs(path), **kwargs)

    def scores(self, query: str) -> np.ndarray:
        """Similarity of *query* to every FAQ (zeros for an empty query)."""
        if not tokenize(query):
            return np.zeros(len(self.faqs))
        return np.asarray(self._scorer.score(query), dtype=float)

    def rank(self, query: str, k: int = 3) -> list[Candidate]:
        """Top-*k* FAQs for *query*, best first (ties keep file order)."""
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [Candidate(self.faqs[i], float(s[i])) for i in order]

    def respond(self, query: str) -> MatchResult:
        """Decide how to reply to *query*: answer, suggest, or fall back."""
        ranked = self.rank(query, k=max(self.num_suggestions * 2, 1))
        best = ranked[0]
        if best.score >= self.threshold:
            logger.debug("answer %.3f %r -> %r", best.score, query, best.faq.question)
            return MatchResult("answer", best.faq.answer, best.score, best.faq)

        suggestions: list[Candidate] = []
        seen_answers: set[str] = set()
        for cand in ranked:
            if cand.score < self.suggest_threshold or len(suggestions) >= self.num_suggestions:
                break
            if cand.faq.answer not in seen_answers:
                seen_answers.add(cand.faq.answer)
                suggestions.append(cand)
        if suggestions:
            logger.debug("suggest %.3f %r", best.score, query)
            return MatchResult("suggest", config.SUGGEST_ANSWER, best.score, None, tuple(suggestions))
        logger.debug("fallback %.3f %r", best.score, query)
        return MatchResult("fallback", config.FALLBACK_ANSWER, best.score)
