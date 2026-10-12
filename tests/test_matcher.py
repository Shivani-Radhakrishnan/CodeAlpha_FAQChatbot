from pathlib import Path

import numpy as np
import pytest

from botbuddy import config, matcher as matcher_mod
from botbuddy.matcher import FAQ, FAQDataError, FAQMatcher, load_faqs, make_scorer

LIGHT_METHODS = ["keyword", "tfidf", "hybrid"]


# ---------------------------------------------------------------- loading


def test_bundled_faq_file_loads(faqs):
    assert len(faqs) >= 60
    assert all(f.question and f.answer for f in faqs)


def test_load_faqs_skips_blank_and_duplicate_rows(tmp_path: Path):
    p = tmp_path / "faq.csv"
    p.write_text(
        'Question,Answer\nHi,"Hello!"\n,\nhi!,"Duplicate"\nOnly question,\nBye,"Goodbye"\n',
        encoding="utf-8",
    )
    assert load_faqs(p) == [FAQ("Hi", "Hello!"), FAQ("Bye", "Goodbye")]


def test_load_faqs_missing_file(tmp_path: Path):
    with pytest.raises(FAQDataError, match="not found"):
        load_faqs(tmp_path / "nope.csv")


def test_load_faqs_wrong_columns(tmp_path: Path):
    p = tmp_path / "faq.csv"
    p.write_text("Q,A\nHi,Hello\n", encoding="utf-8")
    with pytest.raises(FAQDataError, match="columns"):
        load_faqs(p)


def test_load_faqs_empty(tmp_path: Path):
    p = tmp_path / "faq.csv"
    p.write_text("Question,Answer\n", encoding="utf-8")
    with pytest.raises(FAQDataError, match="No FAQs"):
        load_faqs(p)


# ---------------------------------------------------------------- matching


@pytest.mark.parametrize("method", LIGHT_METHODS)
def test_every_known_question_maps_to_its_own_answer(faqs, method):
    m = FAQMatcher(faqs, method=method)
    for faq in faqs:
        result = m.respond(faq.question)
        assert result.status == "answer", faq.question
        assert result.faq == faq
        assert result.answer == faq.answer


@pytest.mark.parametrize("method", LIGHT_METHODS)
def test_scores_are_bounded(faqs, method):
    m = FAQMatcher(faqs, method=method)
    s = m.scores("Where is my parcel?")
    assert s.shape == (len(faqs),)
    assert np.all(s >= 0) and np.all(s <= 1 + 1e-9)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("how do i rest my pasword", "How do I reset my password?"),
        ("trak my ordr", "Can I track my order?"),
        ("refnd time", "What's your refund time?"),
        ("do u ship internationaly", "Do you ship internationally?"),
        ("studnt discount?", "Do you offer student discounts?"),
    ],
)
def test_hybrid_tolerates_typos(hybrid, query, expected):
    result = hybrid.respond(query)
    assert result.status == "answer"
    assert result.faq.question == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("do you accept paypal", "What payment methods are accepted?"),  # word only in the answer
        ("is the packaging recyclable", "Are the products eco-friendly?"),
        ("Do you ship abroad?", "Do you ship internationally?"),
    ],
)
def test_hybrid_uses_answer_vocabulary(hybrid, query, expected):
    assert hybrid.rank(query, k=1)[0].faq.question == expected


@pytest.mark.parametrize(
    "query",
    ["what is the capital of france", "who won the world cup", "how tall is mount everest", "asdfgh qwerty"],
)
def test_out_of_scope_is_not_answered(hybrid, query):
    result = hybrid.respond(query)
    assert result.status in {"fallback", "suggest"}
    assert result.answer in {config.FALLBACK_ANSWER, config.SUGGEST_ANSWER}
    assert result.faq is None


@pytest.mark.parametrize("query", ["", "   ", "?!?"])
def test_empty_query_falls_back(hybrid, query):
    result = hybrid.respond(query)
    assert result.status == "fallback"
    assert result.score == 0.0
    assert result.suggestions == ()


def test_quick_action_questions_answer(hybrid):
    for q in ["Track my order", "Start a return", "Contact support", "hi", "thanks!"]:
        assert hybrid.respond(q).status == "answer", q


def test_rank_is_sorted_and_limited(hybrid):
    ranked = hybrid.rank("track my order", k=5)
    assert len(ranked) == 5
    assert [c.score for c in ranked] == sorted((c.score for c in ranked), reverse=True)


# ---------------------------------------------------------------- suggestions


SMALL = [
    FAQ("How do I reset my password?", "Use the reset link."),
    FAQ("How do I change my password?", "Use the reset link."),  # same answer
    FAQ("How do I change my email?", "Go to settings."),
    FAQ("Do you ship abroad?", "Yes."),
]


def test_low_confidence_gives_deduplicated_suggestions():
    m = FAQMatcher(SMALL, method="keyword", threshold=0.9, suggest_ratio=0.1)
    result = m.respond("password")
    assert result.status == "suggest"
    assert result.answer == config.SUGGEST_ANSWER
    answers = [c.faq.answer for c in result.suggestions]
    assert len(answers) == len(set(answers))  # one suggestion per distinct answer
    assert result.suggestions[0].faq.question in {SMALL[0].question, SMALL[1].question}
    assert result.confidence_label == "low"


def test_suggestions_respect_limit():
    m = FAQMatcher(SMALL, method="keyword", threshold=1.0, suggest_ratio=0.01, num_suggestions=1)
    assert len(m.respond("how do i change my password or email").suggestions) == 1


def test_no_suggestions_when_nothing_is_close():
    m = FAQMatcher(SMALL, method="keyword", threshold=0.5)
    result = m.respond("weather tomorrow")
    assert result.status == "fallback"
    assert result.answer == config.FALLBACK_ANSWER
    assert result.confidence_label == "none"


# ---------------------------------------------------------------- configuration


def test_unknown_method_raises(faqs):
    with pytest.raises(ValueError, match="Unknown method"):
        FAQMatcher(faqs, method="magic")


def test_invalid_threshold_raises(faqs):
    with pytest.raises(ValueError):
        FAQMatcher(faqs, method="keyword", threshold=1.5)


def test_empty_faq_list_raises():
    with pytest.raises(FAQDataError):
        FAQMatcher([], method="keyword")


def test_every_method_has_a_threshold():
    assert set(matcher_mod.SCORERS) <= set(config.ANSWER_THRESHOLDS)


def test_auto_falls_back_when_embeddings_unavailable(monkeypatch, faqs):
    class Broken:
        def __init__(self):
            raise ImportError("sentence-transformers not installed")

    monkeypatch.setitem(matcher_mod.SCORERS, "embedding", Broken)
    method, _ = make_scorer("auto")
    assert method == "hybrid"
    m = FAQMatcher(faqs, method="auto")
    assert m.method == "hybrid"
    assert m.threshold == config.ANSWER_THRESHOLDS["hybrid"]
