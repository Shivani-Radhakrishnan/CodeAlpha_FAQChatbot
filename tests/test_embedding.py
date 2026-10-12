"""Embedding-method checks; skipped unless the optional ONNX embedding dependencies are installed."""

import pytest

pytest.importorskip("onnxruntime")

from botbuddy.matcher import FAQMatcher  # noqa: E402


@pytest.fixture(scope="module")
def emb(faqs):
    return FAQMatcher(faqs, method="embedding")


def test_paraphrases(emb):
    assert emb.respond("I forgot my password, what now").faq.question == "How do I reset my password?"
    assert emb.respond("can I pay when the package arrives").faq.question == "Is cash on delivery available?"


def test_out_of_scope(emb):
    assert emb.respond("what is the capital of france").status != "answer"


def test_known_questions(emb, faqs):
    for faq in faqs:
        assert emb.respond(faq.question).faq == faq
