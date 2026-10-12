"""Offline evaluation of the matching methods on held-out paraphrases.

Usage::

    python -m botbuddy.evaluate                     # keyword, tfidf, hybrid
    python -m botbuddy.evaluate --methods keyword tfidf hybrid embedding

Protocol:

* ``data/eval_questions.csv`` holds hand-written paraphrases that never appear
  in ``faq_data.csv`` plus out-of-scope questions (empty ``expected``).
  ``expected`` lists every FAQ whose answer is correct, separated by ``|``
  (the knowledge base contains near-duplicates such as "Hi"/"Hello").
* For each method the answer threshold is tuned on the ``dev`` split
  (maximising end-to-end accuracy) and all metrics are reported on the
  ``test`` split, which is never used for tuning.

Metrics (test split):

* **top-1 / top-3** - in-scope questions whose correct FAQ is ranked first /
  in the first three (threshold-independent ranking quality).
* **end-to-end** - share of *all* questions handled correctly: in-scope
  answered with a correct FAQ, out-of-scope not answered.
* **fallback P / R** - precision and recall of "did not answer" as a detector
  of out-of-scope questions.
* **answer or suggestion** - in-scope questions where the bot either answers
  correctly or lists a correct FAQ among its "Did you mean ...?" options.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from .matcher import FAQ, FAQMatcher, load_faqs


@dataclass(frozen=True)
class EvalItem:
    question: str
    expected: frozenset[str]  # empty -> out of scope
    split: str


@dataclass(frozen=True)
class Metrics:
    method: str
    threshold: float
    top1: float
    top3: float
    end_to_end: float
    fallback_precision: float
    fallback_recall: float
    answer_or_suggestion: float
    n_in_scope: int
    n_out_of_scope: int
    ms_per_query: float


def load_eval(path: str | Path = config.EVAL_PATH) -> list[EvalItem]:
    """Read the evaluation CSV (columns: question, expected, split)."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [
            EvalItem(
                row["question"],
                frozenset(e for e in row["expected"].split("|") if e),
                row["split"],
            )
            for row in csv.DictReader(fh)
        ]


def _validate(items: Sequence[EvalItem], faqs: Sequence[FAQ]) -> None:
    known = {f.question for f in faqs}
    for it in items:
        missing = it.expected - known
        if missing:
            raise ValueError(f"Eval question {it.question!r} expects unknown FAQ(s) {missing}")
        if it.question in known:
            raise ValueError(f"Eval question {it.question!r} is copied verbatim from the FAQ file")


def _score_matrix(matcher: FAQMatcher, items: Sequence[EvalItem]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    s = np.vstack([matcher.scores(it.question) for it in items])
    return s, (time.perf_counter() - start) * 1000 / max(len(items), 1)


def _metrics(
    method: str,
    scores: np.ndarray,
    items: Sequence[EvalItem],
    faqs: Sequence[FAQ],
    threshold: float,
    ms: float,
    suggestion_hits: int = 0,
) -> Metrics:
    order = np.argsort(-scores, axis=1, kind="stable")
    best = scores[np.arange(len(items)), order[:, 0]]
    answered = best >= threshold
    top1 = top3 = correct = correct_in = 0
    tp = fp = fn = 0  # for "fallback" as the positive class
    n_in = sum(1 for it in items if it.expected)
    for i, it in enumerate(items):
        ranked = [faqs[j].question for j in order[i, :3]]
        if it.expected:
            hit1 = ranked[0] in it.expected
            top1 += hit1
            top3 += any(q in it.expected for q in ranked)
            correct += bool(answered[i] and hit1)
            correct_in += bool(answered[i] and hit1)
            fp += not answered[i]
        else:
            correct += not answered[i]
            tp += not answered[i]
            fn += bool(answered[i])
    return Metrics(
        method=method,
        threshold=threshold,
        top1=top1 / n_in if n_in else 0.0,
        top3=top3 / n_in if n_in else 0.0,
        end_to_end=correct / len(items),
        fallback_precision=tp / (tp + fp) if tp + fp else 0.0,
        fallback_recall=tp / (tp + fn) if tp + fn else 0.0,
        answer_or_suggestion=(correct_in + suggestion_hits) / n_in if n_in else 0.0,
        n_in_scope=n_in,
        n_out_of_scope=len(items) - n_in,
        ms_per_query=ms,
    )


def tune_threshold(scores: np.ndarray, items: Sequence[EvalItem], faqs: Sequence[FAQ]) -> float:
    """Threshold in [0.05, 0.95] maximising end-to-end accuracy on *items*.

    Among equally good thresholds the median one is returned, which keeps a
    margin on both sides instead of sitting on the edge of a plateau.
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    accs = [_metrics("", scores, items, faqs, t, 0).end_to_end for t in grid]
    best = max(accs)
    plateau = [t for t, a in zip(grid, accs) if a == best]
    return float(plateau[len(plateau) // 2])


def evaluate(
    methods: Sequence[str],
    faq_path: str | Path = config.FAQ_PATH,
    eval_path: str | Path = config.EVAL_PATH,
) -> list[Metrics]:
    """Tune each method on the dev split and score it on the test split."""
    faqs = load_faqs(faq_path)
    items = load_eval(eval_path)
    _validate(items, faqs)
    dev = [it for it in items if it.split == "dev"]
    test = [it for it in items if it.split == "test"]
    results = []
    for method in methods:
        matcher = FAQMatcher(faqs, method=method)
        dev_scores, _ = _score_matrix(matcher, dev)
        threshold = tune_threshold(dev_scores, dev, faqs)
        test_scores, ms = _score_matrix(matcher, test)
        matcher.set_threshold(threshold)
        hits = sum(
            1
            for it in test
            if it.expected
            and (r := matcher.respond(it.question)).status == "suggest"
            and any(c.faq.question in it.expected for c in r.suggestions)
        )
        results.append(_metrics(method, test_scores, test, faqs, threshold, ms, hits))
    return results


def format_table(results: Sequence[Metrics]) -> str:
    """Render results as a Markdown table."""
    lines = [
        "| Method | Top-1 | Top-3 | End-to-end | Fallback precision | Fallback recall "
        "| Answer or suggestion | Threshold (dev-tuned) | ms/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in results:
        lines.append(
            f"| `{m.method}` | {m.top1:.1%} | {m.top3:.1%} | {m.end_to_end:.1%} | "
            f"{m.fallback_precision:.1%} | {m.fallback_recall:.1%} | "
            f"{m.answer_or_suggestion:.1%} | {m.threshold:.2f} | {m.ms_per_query:.1f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=["keyword", "tfidf", "hybrid"])
    parser.add_argument("--faqs", default=str(config.FAQ_PATH))
    parser.add_argument("--eval", default=str(config.EVAL_PATH))
    args = parser.parse_args(argv)
    results = evaluate(args.methods, args.faqs, args.eval)
    m = results[0]
    print(
        f"Test split: {m.n_in_scope} in-scope + {m.n_out_of_scope} out-of-scope questions "
        f"(thresholds tuned on the dev split)\n"
    )
    print(format_table(results))


if __name__ == "__main__":
    main()
