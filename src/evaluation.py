"""
Part D — Evaluation harness.

Runs the pipeline against the assignment's fixed 8-question set and records,
per question: the retrieved chunk(s) (heading + page + score), the generated
answer, and a "correct?" verdict slot.

Note on the "correct?" column: filling it in requires comparing the
generated answer against the real document's actual content. Per this
project's constraint (the assistant must not read or reproduce the source
PDF's content), that column is intentionally left as a `to_review` sentinel
here rather than hand-filled - run `mark_correctness()` yourself after
reading the generated answers against the source document, or use the
notebook's evaluation cell interactively.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FIXED_QUESTIONS: list[dict] = [
    {"id": 1, "question": "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?"},
    {"id": 2, "question": "What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?"},
    {"id": 3, "question": "What are the two software load-lines supported by the 1830 PSS system?"},
    {"id": 4, "question": "Which fan units are supported on the 1830 PSS-32 shelf?"},
    {"id": 5, "question": "Which fan unit(s) are used on the 1830 PSS-16II shelf?"},
    {"id": 6, "question": "Name the power filter cards supported on the 1830 PSS-8 shelf."},
    {"id": 7, "question": "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported?"},
    {
        "id": 8,
        "question": "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?",
        "note": "Trick question: this spec is not expected to be in the provided page range (Ch.1-2) - a correct pipeline should refuse rather than guess a number.",
    },
]


@dataclass
class EvalRow:
    id: int
    question: str
    retrieved: list[dict]     # [{heading, page_start, page_end, score}, ...]
    answer: str
    provider: str
    model: str
    refused: bool
    has_citation: bool
    correct: str = "TO_REVIEW"   # user fills in after checking against the source doc
    notes: str = ""


def run_evaluation(pipeline, **retrieve_kwargs) -> list[EvalRow]:
    rows = []
    for item in FIXED_QUESTIONS:
        result = pipeline.answer(item["question"], **retrieve_kwargs)
        retrieved = [
            {
                "heading": c.heading_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "score": round(score, 4),
            }
            for c, score in result.retrieval.chunks_with_scores
        ]
        rows.append(
            EvalRow(
                id=item["id"],
                question=item["question"],
                retrieved=retrieved,
                answer=result.answer,
                provider=result.provider,
                model=result.model,
                refused=result.refused,
                has_citation=result.has_citation,
                notes=item.get("note", ""),
            )
        )
    return rows


def to_markdown_table(rows: list[EvalRow]) -> str:
    header = "| # | Question | Top source(s) | Answer | Refused? | Cited? | Correct? | Notes |\n"
    header += "|---|----------|----------------|--------|----------|--------|----------|-------|\n"
    lines = [header]
    for r in rows:
        sources = "<br>".join(f"{d['heading']} (p.{d['page_start']}, s={d['score']:.3f})" for d in r.retrieved[:3])
        answer = r.answer.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.id} | {r.question} | {sources} | {answer} | {r.refused} | {r.has_citation} | {r.correct} | {r.notes} |"
        )
    return "\n".join(lines)


def to_dicts(rows: list[EvalRow]) -> list[dict]:
    return [r.__dict__ for r in rows]
