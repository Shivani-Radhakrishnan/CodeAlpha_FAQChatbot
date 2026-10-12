"""Grounded answers from the store's policy documents, with citations.

    question -> retrieve sections -> (refuse if nothing relevant) -> local LLM writes an
    answer from numbered sources -> grounding check: every sentence must be supported by
    a retrieved section, which becomes its citation -> answer + citations

Refusal comes first and does not use the LLM: if the best section's embedding
similarity is below ``min_relevance`` the bot says it doesn't know (this also turns
away off-topic questions). The grounding check drops sentences that cite nothing or
aren't close to the sentences of the source they cite; if nothing survives, the answer
falls back to quoting the best-matching source sentence. Thresholds are tuned on the
dev questions only (``python -m botbuddy.rag_eval``).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from botbuddy import config

DOCS_DIR = config.PROJECT_ROOT / "data" / "policies"
REFUSAL = ("I couldn't find that in our policies, so I don't want to guess. "
           "Type 'human' to ask our support team.")

SYSTEM_PROMPT = (
    "You are BotBuddy, the customer-support assistant of an online store. Answer the customer's "
    "question in your own words using only the numbered sources. Answer the question directly: for a "
    "yes/no question, start with Yes or No. Use one or two short sentences and put the source number "
    "at the end of each sentence, like this [1]. If the sources do not answer the question, reply "
    "exactly: NOT_IN_SOURCES"
)

# One worked example (on a made-up source, not from the policy pages) shows the small model
# the expected format: a direct answer, then the citation at the end of the sentence.
EXAMPLE = [
    {"role": "user", "content": "Sources:\n[1] Parking: Customers can park for free for up to 2 hours. "
                                "Longer stays cost $3 per hour.\n[2] Opening hours: The shop is open 9am to 6pm.\n\n"
                                "Question: is parking free if I stay three hours?"},
    {"role": "assistant", "content": "No, parking is free only for the first 2 hours; after that it costs $3 per hour [1]."},
]


@dataclass(frozen=True)
class Chunk:
    id: str          # "returns#return-window"
    document: str    # "Returns, exchanges and refunds"
    heading: str     # "Return window"
    text: str

    @property
    def title(self) -> str:
        return f"{self.document}: {self.heading}"


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def load_chunks(directory: Path = DOCS_DIR) -> list[Chunk]:
    """One chunk per "## " section of each Markdown policy page."""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        for m in re.finditer(r"^## (.+?)\n(.+?)(?=^## |\Z)", text, re.M | re.S):
            chunks.append(Chunk(f"{path.stem}#{slug(m.group(1))}", doc, m.group(1).strip(), " ".join(m.group(2).split())))
    return chunks


def _split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping each citation marker with the sentence it belongs to.

    The model either puts markers after a sentence ("... days [1]." / "... days. [1]") or in
    front of it ("[1] If you ..."); it is consistent within one answer, so the style is read
    from whether the answer starts with a marker.
    """
    text = text.strip()
    pieces = re.split(r"((?:\s*\[\d+\])+)", text)
    leading = bool(re.match(r"\[\d+\]", text))
    out: list[str] = []
    pending = ""
    for piece in pieces:
        if not piece.strip():
            continue
        if re.fullmatch(r"(?:\s*\[\d+\])+", piece):
            marks = " ".join(re.findall(r"\[\d+\]", piece))
            if leading:
                pending = marks
            elif out:
                out[-1] = f"{out[-1]} {marks}"
            continue
        sents = _split(piece)
        if leading and pending and sents:
            sents[0] = f"{pending} {sents[0]}"
            pending = ""
        out += sents
    # a full stop right after a marker belongs to the sentence: "... days [1]." -> one sentence
    merged: list[str] = []
    for sent in out:
        if merged and re.fullmatch(r"[.!?]", sent):
            merged[-1] += sent
        else:
            merged.append(sent)
    # a bare "Yes." / "No, sorry." lead-in belongs to the sentence that follows it
    joined: list[str] = []
    for sent in merged:
        if joined and len(joined[-1].split()) <= 3 and not re.search(r"\[\d+\]", joined[-1]):
            joined[-1] = f"{joined[-1]} {sent}"
        else:
            joined.append(sent)
    return joined


# ── retrieval ─────────────────────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], float)
        self.avg = float(self.len.mean())
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float        # hybrid ranking score
    relevance: float    # cosine similarity of question and section (absolute, used for refusal)


class Retriever:
    def __init__(self, chunks: list[Chunk], encoder, weight: float = 0.7) -> None:
        self.chunks = chunks
        self.encoder = encoder
        self.weight = weight
        self.vectors = encoder.encode([f"{c.title}. {c.text}" for c in chunks])
        self.bm25 = _BM25([_words(f"{c.title} {c.text}") for c in chunks])
        # each section is also scored by its best-matching sentence, so a short, specific
        # question ("is there a late fee?") can match one sentence of a long section
        self._owner, sents = [], []
        for i, c in enumerate(chunks):
            for s in sentences(c.text):
                self._owner.append(i)
                sents.append(f"{c.heading}. {s}")
        self._owner = np.array(self._owner)
        self.sentence_vectors = encoder.encode(sents)

    def search(self, question: str, k: int = 3) -> list[Hit]:
        q = self.encoder.encode([question])[0]
        cos = self.vectors @ q
        sent = self.sentence_vectors @ q
        best_sentence = np.full(len(self.chunks), -1.0)
        np.maximum.at(best_sentence, self._owner, sent)
        cos = np.maximum(cos, best_sentence)
        bm = self.bm25.scores(_words(question))
        bm = bm / bm.max() if bm.max() > 0 else bm
        score = self.weight * cos + (1 - self.weight) * bm
        order = np.argsort(-score, kind="stable")[:k]
        return [Hit(self.chunks[i], float(score[i]), float(cos[i])) for i in order]


# ── answering ─────────────────────────────────────────────────────────────────

@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)     # chunk ids, in citation order
    refused: bool = False
    reason: str = ""            # low relevance | model declined | grounded | partly grounded | unsupported answer
    sources: list[Hit] = field(default_factory=list)
    raw: str = ""               # what the model wrote, before the grounding check
    dropped: list[str] = field(default_factory=list)       # sentences removed by the grounding check


_CITE = re.compile(r"\[(\d+)\]")


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    sources = "\n".join(f"[{i + 1}] {h.chunk.heading}: {h.chunk.text}" for i, h in enumerate(hits))
    return [{"role": "system", "content": SYSTEM_PROMPT}, *EXAMPLE,
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]


class DocumentAnswerer:
    def __init__(self, retriever: Retriever, model, k: int = 3, min_relevance: float = 0.35,
                 min_support: float = 0.55, check_grounding: bool = True) -> None:
        self.retriever, self.model = retriever, model
        self.k, self.min_relevance, self.min_support = k, min_relevance, min_support
        self.check_grounding = check_grounding
        enc = retriever.encoder
        self._sent = {c.id: sentences(c.text) for c in retriever.chunks}
        self._sent_vec = {cid: enc.encode(ss) for cid, ss in self._sent.items()}

    def answer(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.k)
        if not hits or hits[0].relevance < self.min_relevance:
            return Answer(REFUSAL, refused=True, reason="low relevance", sources=hits)
        raw = self.model.chat(build_messages(question, hits))
        if "NOT_IN_SOURCES" in raw or not raw.strip():
            return Answer(REFUSAL, refused=True, reason="model declined", sources=hits, raw=raw)
        if not self.check_grounding:
            cited = self._cited_ids(raw, hits)
            return Answer(_strip_markers(raw), cited, reason="ungrounded", sources=hits, raw=raw)
        return self._grounded(question, raw, hits)

    @staticmethod
    def _cited_ids(text: str, hits: list[Hit]) -> list[str]:
        ids = []
        for n in _CITE.findall(text):
            i = int(n) - 1
            if 0 <= i < len(hits) and hits[i].chunk.id not in ids:
                ids.append(hits[i].chunk.id)
        return ids

    def _support(self, sentence: str, chunk_ids: list[str]) -> tuple[float, Optional[str]]:
        """Best similarity between the sentence and any sentence of the given sections."""
        clean = _strip_markers(sentence)
        if not clean or not chunk_ids:
            return 0.0, None
        v = self.retriever.encoder.encode([clean])[0]
        scored = [(float((self._sent_vec[c] @ v).max()), c) for c in chunk_ids]
        return max(scored)

    def _grounded(self, question: str, raw: str, hits: list[Hit]) -> Answer:
        """Keep the sentences a retrieved section supports, and cite that section.

        Citations are attributed after generation: each sentence is credited to the retrieved
        section that supports it best. A small model's own [n] markers are often missing or
        point to the wrong source, so they are not trusted.
        """
        if _DECLINE.search(raw):
            return Answer(REFUSAL, refused=True, reason="model declined", sources=hits, raw=raw)
        ids = [h.chunk.id for h in hits]
        kept, dropped, cited = [], [], []
        for sent in sentences(raw):
            sim, best = self._support(sent, ids)
            if sim >= self.min_support:
                kept.append(_strip_markers(sent))
                if best not in cited:
                    cited.append(best)
            else:
                dropped.append(sent)
        if kept:
            return Answer(" ".join(kept), cited, reason="grounded" if not dropped else "partly grounded",
                          sources=hits, raw=raw, dropped=dropped)
        # Nothing the model wrote is supported by the retrieved sections: decline rather than guess.
        return Answer(REFUSAL, refused=True, reason="unsupported answer", sources=hits, raw=raw, dropped=dropped)


# The model saying, in its own words, that the sources don't answer the question.
_DECLINE = re.compile(r"\bsources?\b[^.]*\b(?:do not|don't|does not|doesn't|did not)\b[^.]*\b(?:provide|mention|contain|say|cover|include)",
                      re.I)


def _strip_markers(text: str) -> str:
    t = _CITE.sub("", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", t)).strip()


class ClosedBookAnswerer:
    """Baseline: the same model answering without any documents."""

    PROMPT = ("You are BotBuddy, the customer-support assistant of an online store. Answer the customer's "
              "question in one to three short sentences. If you don't know, say you don't know.")

    def __init__(self, model) -> None:
        self.model = model

    def answer(self, question: str) -> Answer:
        raw = self.model.chat([{"role": "system", "content": self.PROMPT}, {"role": "user", "content": question}])
        declined = bool(re.search(r"\b(?:don't|do not) (?:know|have (?:that|this|the) information)\b", raw.lower()))
        return Answer(raw, refused=declined, reason="closed book", raw=raw)
