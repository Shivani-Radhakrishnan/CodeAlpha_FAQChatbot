import numpy as np
import pytest

from botbuddy import config
from botbuddy.evaluate import EvalItem, _metrics, evaluate, format_table, load_eval, tune_threshold
from botbuddy.matcher import FAQ, FAQMatcher
from botbuddy.text import normalize


@pytest.fixture(scope="module")
def items():
    return load_eval(config.EVAL_PATH)


def test_eval_set_is_held_out_and_covers_every_faq(items, faqs):
    faq_questions = {f.question for f in faqs}
    normalized_faqs = {normalize(q) for q in faq_questions}
    assert all(normalize(it.question) not in normalized_faqs for it in items)
    covered = set().union(*(it.expected for it in items))
    assert covered == faq_questions
    assert all(it.expected <= faq_questions for it in items)
    assert {it.split for it in items} == {"dev", "test"}
    assert sum(1 for it in items if not it.expected) >= 20  # out-of-scope questions


def test_metrics_on_handmade_example():
    faqs = [FAQ("a", "A"), FAQ("b", "B")]
    items = [
        EvalItem("q1", frozenset({"a"}), "test"),  # ranked right, answered
        EvalItem("q2", frozenset({"b"}), "test"),  # ranked wrong (top-3 hit)
        EvalItem("q3", frozenset(), "test"),  # out of scope, low score -> fallback
        EvalItem("q4", frozenset(), "test"),  # out of scope, high score -> wrongly answered
    ]
    scores = np.array([[0.9, 0.1], [0.6, 0.5], [0.2, 0.1], [0.8, 0.0]])
    m = _metrics("x", scores, items, faqs, threshold=0.5, ms=0.0)
    assert m.top1 == 0.5
    assert m.top3 == 1.0
    assert m.end_to_end == 0.5  # q1 and q3
    assert m.fallback_precision == 1.0  # the one fallback (q3) was out of scope
    assert m.fallback_recall == 0.5  # q4 was missed
    assert tune_threshold(scores, items, faqs) > 0.8  # only then is q4 rejected


@pytest.mark.parametrize("method", ["keyword", "tfidf", "hybrid"])
def test_config_thresholds_match_dev_tuning(method):
    """The thresholds shipped in config.py must be the ones the eval script tunes."""
    (result,) = evaluate([method])
    assert result.threshold == pytest.approx(config.ANSWER_THRESHOLDS[method])


def test_improved_method_beats_baselines():
    keyword, tfidf, hybrid = evaluate(["keyword", "tfidf", "hybrid"])
    assert hybrid.top1 > max(keyword.top1, tfidf.top1)
    assert hybrid.end_to_end > max(keyword.end_to_end, tfidf.end_to_end)
    assert "| `hybrid` |" in format_table([hybrid])
