"""Central configuration for BotBuddy (paths, defaults, thresholds)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FAQ_PATH: Path = PROJECT_ROOT / "faq_data.csv"
EVAL_PATH: Path = PROJECT_ROOT / "data" / "eval_questions.csv"

#: Matching method used by the app. ``"auto"`` picks the first method in
#: ``AUTO_METHODS`` that loads. The order comes from the held-out evaluation
#: (``python -m botbuddy.evaluate``; see the README results table):
#: sentence embeddings score best, the TF-IDF hybrid is the lightweight
#: fallback when ``sentence-transformers`` is not installed.
DEFAULT_METHOD: str = "auto"
AUTO_METHODS: tuple[str, ...] = ("embedding", "hybrid")

#: Minimum score needed to answer directly, per method. Each value is tuned
#: on the *dev* split of ``data/eval_questions.csv`` by the evaluation script
#: (it prints the tuned value); results are then reported on the *test* split.
ANSWER_THRESHOLDS: dict[str, float] = {
    "keyword": 0.08,
    "tfidf": 0.37,
    "hybrid": 0.18,
    "embedding": 0.40,
}

#: Below the answer threshold but at least this fraction of it, the bot asks
#: "Did you mean ...?" with its top suggestions instead of a flat fallback.
SUGGEST_RATIO: float = 0.6

#: Number of "Did you mean" suggestions shown.
NUM_SUGGESTIONS: int = 3

FALLBACK_ANSWER: str = (
    "Sorry, I don't have an answer for that yet. I can help with orders, "
    "shipping, returns, payments and your account - try rephrasing, or type "
    "'human' to reach our support team."
)

SUGGEST_ANSWER: str = "I'm not completely sure what you mean. Did you mean one of these?"

#: Sentence-embedding model for the optional ``embedding`` method.
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Document answers (botbuddy.rag): below this question-to-section similarity the bot
#: declines without calling the language model; answer sentences must be at least this
#: similar to a sentence of the section they cite. Both chosen on the dev split of
#: data/rag_questions.csv (python -m botbuddy.rag_eval --tune).
RAG_MIN_RELEVANCE: float = 0.25
RAG_MIN_SUPPORT: float = 0.5
