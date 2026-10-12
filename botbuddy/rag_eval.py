"""Evaluate answers from the policy documents (data/rag_questions.csv).

    python -m botbuddy.rag_eval --tune --record   # dev split: choose thresholds (needs the model)
    python -m botbuddy.rag_eval --record          # test split with the chosen thresholds
    python -m botbuddy.rag_eval                   # replay from data/rag_llm_cache.json (no model)

143 hand-written questions: answerable ones list the section(s) holding the answer and
the facts a correct answer must contain ("30 days", "5.95"; alternatives with "|", all
groups separated by ";"); unanswerable ones are on-topic but not covered by the
documents; off-topic ones have nothing to do with the store. A good assistant answers
the first kind correctly and declines the other two.

Systems: the existing FAQ matcher (embedding scorer, its dev-tuned threshold), the same
LLM answering without documents ("closed book"), RAG without the grounding check, and
RAG with it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

from botbuddy import config
from botbuddy.encoder import get_encoder
from botbuddy.llm import ReplayCache, get_model
from botbuddy.rag import Answer, ClosedBookAnswerer, DocumentAnswerer, Retriever, load_chunks

QUESTIONS = config.PROJECT_ROOT / "data" / "rag_questions.csv"
CACHE = config.PROJECT_ROOT / "data" / "rag_llm_cache.json"
RESULTS = config.PROJECT_ROOT / "data" / "rag_results.json"
RESULTS_MD = config.PROJECT_ROOT / "data" / "rag_results.md"
RELEVANCE_GRID = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
SUPPORT_GRID = [0.4, 0.5, 0.6, 0.7]


def load(split: str) -> list[dict]:
    with QUESTIONS.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def facts_ok(answer: str, facts: str) -> bool:
    a = answer.lower().replace("–", "-").replace("—", "-")
    return all(any(alt.strip().lower() in a for alt in group.split("|")) for group in facts.split(";") if group)


def score(questions: list[dict], answers: list[Answer], cites: bool) -> dict:
    by = defaultdict(list)
    cite_ok, retrieved = [], []
    for q, a in zip(questions, answers):
        if q["kind"] == "answerable":
            gold = set(q["gold"].split(";"))
            by["answerable correct"].append((not a.refused) and facts_ok(a.text, q["facts"]))
            by["answerable refused"].append(a.refused)
            if cites and not a.refused:
                cite_ok.append(bool(gold & set(a.citations)))
            if a.sources:
                retrieved.append(bool(gold & {h.chunk.id for h in a.sources}))
        else:
            by[f"{q['kind']} declined"].append(a.refused)
    total = by["answerable correct"] + by["unanswerable declined"] + by["offtopic declined"]
    out = {k: sum(v) / len(v) for k, v in by.items() if v} | {k + " n": len(v) for k, v in by.items() if v}
    out["overall"] = sum(total) / len(total)
    if cite_ok:
        out["citation correct"] = sum(cite_ok) / len(cite_ok)
    if retrieved:
        out["retrieval recall@3"] = sum(retrieved) / len(retrieved)
    return out


class FaqBaseline:
    """The existing FAQ matcher: answers with the best FAQ answer above its threshold."""

    def __init__(self) -> None:
        from botbuddy.matcher import FAQMatcher, load_faqs

        self.matcher = FAQMatcher(load_faqs(), method="embedding")

    def answer(self, question: str) -> Answer:
        r = self.matcher.respond(question)
        if r.status == "answer":
            return Answer(r.answer)
        return Answer(r.answer, refused=True)   # "did you mean" or fallback: no answer given


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="grid-search thresholds on the dev split")
    ap.add_argument("--record", action="store_true", help="call the model for prompts not in the cache")
    args = ap.parse_args(argv)

    llm = ReplayCache(CACHE, get_model() if args.record else None)
    enc = get_encoder()
    retriever = Retriever(load_chunks(), enc)
    try:
        if args.tune:
            dev = load("dev")
            grid = {}
            for rel, sup in product(RELEVANCE_GRID, SUPPORT_GRID):
                rag = DocumentAnswerer(retriever, llm, min_relevance=rel, min_support=sup)
                grid[(rel, sup)] = score(dev, [rag.answer(q["question"]) for q in dev], cites=True)["overall"]
            # ties: prefer the middle of the grid (less extreme settings)
            best = max(grid, key=lambda k: (grid[k], -abs(k[0] - 0.3), -abs(k[1] - 0.55)))
            print("dev overall accuracy by (min_relevance, min_support):")
            for k, v in sorted(grid.items()):
                print(f"  {k}: {v:.3f}{'   <- chosen' if k == best else ''}")
            print(f"set RAG_MIN_RELEVANCE = {best[0]}, RAG_MIN_SUPPORT = {best[1]} in botbuddy/config.py")
            return 0

        test = load("test")
        rel, sup = config.RAG_MIN_RELEVANCE, config.RAG_MIN_SUPPORT
        systems = {
            "FAQ matcher (existing)": (FaqBaseline(), False),
            "LLM, no documents": (ClosedBookAnswerer(llm), False),
            "RAG, no grounding check": (DocumentAnswerer(retriever, llm, min_relevance=rel, min_support=sup,
                                                         check_grounding=False), True),
            "RAG with grounding check": (DocumentAnswerer(retriever, llm, min_relevance=rel, min_support=sup), True),
        }
        report = {"n": len(test), "min_relevance": rel, "min_support": sup, "systems": {}, "examples": []}
        final_answers = None
        for name, (system, cites) in systems.items():
            answers = [system.answer(q["question"]) for q in test]
            report["systems"][name] = score(test, answers, cites)
            final_answers = answers
        reasons = defaultdict(int)
        for a in final_answers:
            reasons[a.reason] += 1
        report["final_reasons"] = dict(reasons)
        report["examples"] = [
            {"question": q["question"], "kind": q["kind"], "answer": a.text, "citations": a.citations,
             "refused": a.refused, "reason": a.reason, "correct": (q["kind"] != "answerable" and a.refused) or
             (q["kind"] == "answerable" and not a.refused and facts_ok(a.text, q["facts"]))}
            for q, a in zip(test, final_answers)]
    finally:
        if args.record and llm.dirty:
            llm.save()
    RESULTS.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render(report)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


def render(rep: dict) -> str:
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    cols = ["overall", "answerable correct", "answerable refused", "unanswerable declined", "offtopic declined",
            "citation correct", "retrieval recall@3"]
    first = next(iter(rep["systems"].values()))
    out = [f"### Answers from the policy documents ({rep['n']} test questions: "
           f"{first['answerable correct n']} answerable, {first['unanswerable declined n']} not covered, "
           f"{first['offtopic declined n']} off-topic)", "",
           f"Thresholds chosen on the dev split: min_relevance {rep['min_relevance']}, min_support "
           f"{rep['min_support']}.", "",
           "| System | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in rep["systems"].items():
        out.append(f"| {name} | " + " | ".join(pct(s.get(c)) for c in cols) + " |")
    out += ["", "How the final system's answers came about: " +
            ", ".join(f"{k} {v}" for k, v in sorted(rep["final_reasons"].items())), "",
            "Final system, every wrong test answer:", "", "| Question | Kind | Answer | Cited |", "|---|---|---|---|"]
    for e in rep["examples"]:
        if not e["correct"]:
            out.append(f"| {e['question']} | {e['kind']} | {e['answer'][:140]} | {', '.join(e['citations'])} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
