"""Document answers: chunking, sentence splitting, retrieval, grounding and the replay cache.

The language model is replaced by a fake, so these run without llama.cpp or the model file.
The embedding model is needed (skipped if the optional ONNX dependencies are missing).
"""
import json

import pytest

from botbuddy import rag
from botbuddy.llm import CacheMiss, ReplayCache

pytest.importorskip("onnxruntime")


class FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, 0

    def chat(self, messages, max_tokens=200):
        self.calls += 1
        return self.reply


@pytest.fixture(scope="module")
def retriever():
    from botbuddy.encoder import get_encoder

    return rag.Retriever(rag.load_chunks(), get_encoder())


def answerer(retriever, reply):
    model = FakeModel(reply)
    return rag.DocumentAnswerer(retriever, model, min_relevance=0.25, min_support=0.5), model


def test_chunks_are_one_per_section():
    chunks = rag.load_chunks()
    assert len(chunks) == 39 and len({c.id for c in chunks}) == 39
    window = next(c for c in chunks if c.id == "returns#return-window")
    assert window.heading == "Return window" and "30 days" in window.text


@pytest.mark.parametrize("text,expected", [
    ("No, there is no late fee [1]. Missed payments are retried after 3 days [1].",
     ["No, there is no late fee [1].", "Missed payments are retried after 3 days [1]."]),
    ("[1] If an order is late we email you. [2] Express fees are refunded.",
     ["[1] If an order is late we email you.", "[2] Express fees are refunded."]),
    ("Yes. You can exchange it within 30 days [1].", ["Yes. You can exchange it within 30 days [1]."]),
])
def test_sentences_keep_citations_with_their_sentence(text, expected):
    assert rag.sentences(text) == expected


def test_retrieval_finds_the_section(retriever):
    hits = retriever.search("is there a fee if one of my installments doesn't go through")
    assert "orders_payments#paying-in-installments" in [h.chunk.id for h in hits]


def test_off_topic_is_declined_without_calling_the_model(retriever):
    a, model = answerer(retriever, "should not be used")
    out = a.answer("what is the capital of france")
    assert out.refused and out.reason == "low relevance" and model.calls == 0


def test_supported_answer_is_kept_and_cited_to_the_right_section(retriever):
    a, _ = answerer(retriever, "No, there is no late fee; a missed installment is retried after 3 days [2].")
    out = a.answer("is there a late fee if an installment fails")
    assert not out.refused and "late fee" in out.text and "[" not in out.text
    assert out.citations == ["orders_payments#paying-in-installments"]   # attributed by support, not by "[2]"


def test_unsupported_sentences_are_dropped_and_all_unsupported_declines(retriever):
    a, _ = answerer(retriever, "You can return items within 30 days [1]. We also give every customer a free puppy [1].")
    out = a.answer("how long do I have to return something")
    assert out.reason == "partly grounded" and "puppy" not in out.text and len(out.dropped) == 1
    a, _ = answerer(retriever, "Our CEO loves jazz and lives on a boat [1].")
    assert a.answer("how long do I have to return something").reason == "unsupported answer"


def test_the_model_declining_is_respected(retriever):
    for reply in ("NOT_IN_SOURCES", "The sources do not provide information about that."):
        a, _ = answerer(retriever, reply)
        assert a.answer("how long is the warranty").refused


def test_replay_cache(tmp_path):
    path = tmp_path / "cache.json"
    rec = ReplayCache(path, FakeModel("hi"))
    msgs = [{"role": "user", "content": "x"}]
    assert rec.chat(msgs) == "hi" and rec.dirty
    rec.save()
    replay = ReplayCache(path)
    assert replay.chat(msgs) == "hi"
    with pytest.raises(CacheMiss):
        replay.chat([{"role": "user", "content": "never recorded"}])
    assert len(json.loads(path.read_text())) == 1


def test_published_rag_results_reproduce_from_the_recorded_outputs():
    """Replays data/rag_llm_cache.json; fails on any prompt that was not recorded."""
    from botbuddy import rag_eval

    published = json.loads(rag_eval.RESULTS.read_text())
    assert rag_eval.main([]) == 0
    now = json.loads(rag_eval.RESULTS.read_text())
    assert now["systems"] == published["systems"]
