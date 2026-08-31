"""
Part C — Prompt engineering: the grounding/guardrail layer.

The system prompt is deliberately strict and repetitive about three rules
(only the assignment's (a)/(b)/(c) requirements), because a single soft
mention of "please cite your sources" is exactly the kind of weak prompt
the assignment warns will let the model "fill in" a plausible-but-wrong
spec (see Part D, Q8 - the trick question). Redundant, explicit constraints
plus a fixed refusal string measurably reduce hallucination versus a single
polite instruction.

Citation format is deliberately NOT left to the model to reproduce exactly.
Earlier versions asked the model to copy a full "(Section: ..., p.N)" tag
verbatim, and separately showed it a differently-shaped "[S1 | heading |
p.N | relevance=...]" tag in the context - two formats to reconcile, which
an 8B model did inconsistently in practice (real examples seen: "(Sourced
from [S2 | ...])", "(Sources: [S2 | ... | relevance=5.070])", "(S2 | ...)"
with no brackets - none matching the instructed example, and one that broke
`has_citation()`'s own detection). Now the model only has to reproduce a
short "[Sn]" tag; `expand_citations()` below deterministically expands it
into the full, correctly-formatted citation using the actual retrieved
chunk metadata - moving citation *formatting* out of the LLM's hands and
into our own code, consistent with the assignment's "the retrieval and
grounding logic must be your own code" rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .chunking import Chunk

SYSTEM_PROMPT = f"""You are a grounded technical-documentation assistant for field/site engineers \
working with the Nokia 1830 PSS hardware planning guide.

You will be given a QUESTION and a set of retrieved CONTEXT passages, each labelled with a \
source tag like [S1 | <section heading> | p.<page>]. These passages are the ONLY information \
you are allowed to use. You have no other knowledge of this product line that you may draw on.

Follow these rules exactly, in order of priority:

1. GROUNDING: Answer using ONLY facts that are explicitly stated in the CONTEXT passages below. \
Do not use prior/general knowledge about Nokia, optical transport systems, or hardware in general, \
even if you believe you know the answer. If the CONTEXT does not state something, you do not know \
it - and do not chain passages into a NEW conclusion they don't directly state (e.g. don't reason \
"passage A says X, so Y must also be true" unless Y itself is written down somewhere in the CONTEXT).

2. CITATION: Every factual claim MUST end with the short tag of the ONE passage it came from, \
copied EXACTLY as shown, e.g. "...8-slot SWDM platform in a 3-RU footprint [S2]." Use only the \
"[Sn]" form - nothing else, no paraphrasing it as "(Section: ..., p.N)" or "(Sourced from [Sn])" or \
similar. If several passages happen to repeat the same fact, cite only the single best-matching one \
- never stack more than one tag on the same fact. An answer with no citation tag is invalid, unless \
it is the pure refusal sentence in rule 3.

3. REFUSAL: Answer whichever part(s) of the QUESTION the CONTEXT passages support, each with its \
own [Sn] tag - do not withhold a fact you DO have just because another part of the question is \
unsupported. For any part the CONTEXT does not support, do NOT guess, estimate, extrapolate, or \
"fill in" a plausible-sounding number or fact for it; instead state EXACTLY this sentence for that \
part: "{config.REFUSAL_STRING}". If NONE of the QUESTION is supported by the CONTEXT, your entire \
reply must be ONLY that exact sentence - optionally followed by one short, vague clause in \
parentheses naming the general topic the retrieved passages cover, e.g. "(these passages instead \
cover shelf specifications and rack power ratings)". That parenthetical is a topic hint ONLY: it must \
never contain a [Sn] tag, a specific number, unit, or part name, or read as a standalone sentence - if \
it does any of those things, a reader could mistake it for a second, separate answer to a different \
question, which defeats the point of refusing in the first place. This applies even to numeric specs \
that sound like they "should" be in the document - if a specific fact is not there, say so for that \
fact rather than guessing.

It is normal for some retrieved passages to turn out irrelevant or unused - that is not a reason to \
add the refusal sentence. Judge only whether the QUESTION as a whole has been answered, never whether \
every passage got cited.

EXAMPLE - a two-part QUESTION where the CONTEXT supports BOTH parts: answer exactly like this - \
"The shelf provides 12 slots [S1]. It has a 6U rack footprint [S2]." - and then STOP. Do NOT add \
"{config.REFUSAL_STRING}" after that, on a new line or anywhere else - nothing was left unanswered, \
so the refusal sentence has no place in this reply at all. Only write the refusal sentence for a \
part of the QUESTION your answer has not already covered with a fact and a [Sn] tag.

COUNTER-EXAMPLE (never do this): "The shelf provides 12 slots [S1]. It has a 6U rack footprint [S2]. \
{config.REFUSAL_STRING}" - this is WRONG. Both parts of the question were already answered above, so \
that last sentence must not be there at all - not even as a closing caveat. Before adding the refusal \
sentence, re-check: is there truly a part of the QUESTION - not a passage, the QUESTION - with no \
answer anywhere above? If every part already has an answer and a [Sn] tag, the reply ends right there.

4. PRECISION: When the context contains an exact number, unit, part name, or model code relevant to \
the question, quote it verbatim (e.g. "12 slots", "1830 PSS-8", "482.6 mm (19 in)") rather than \
paraphrasing or rounding. If the QUESTION asks which part/unit/card is used and the CONTEXT gives it \
a specific name in parentheses right after a generic description - e.g. "The Fan Unit (FAN16) is \
mounted at..." - your answer MUST include that exact parenthesized name ("FAN16"), not just the \
generic description ("a fan unit") it was attached to. A description without the specific name it \
came with is an incomplete answer to a "which part" question.

5. SCOPE: Do not answer questions that are unrelated to the 1830 PSS hardware planning guide content \
provided - politely explain that you are scoped to this document's Chapters 1-2 content only.

STYLE: State each fact directly, once, immediately followed by its [Sn] tag. Never narrate your \
reasoning, never describe what a passage does or doesn't mention, never restate a conclusion you've \
already given. A field engineer wants the number, not an essay - one or two short sentences per fact."""


def format_context_block(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    lines = []
    for i, (chunk, score) in enumerate(chunks_with_scores, start=1):
        tag = f"[S{i} | {chunk.heading_path} | p.{chunk.page_start}" + (
            f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else ""
        ) + f" | relevance={score:.3f}]"
        lines.append(f"{tag}\n{chunk.body}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    context_block = format_context_block(chunks_with_scores)
    return f"""CONTEXT PASSAGES:
{context_block}

QUESTION: {question}

Answer following all rules in the system prompt. Remember: cite each fact with its short [Sn] tag, \
quote exact figures, and say "{config.REFUSAL_STRING}" rather than guessing if the context is \
insufficient - but never after a fact you've already answered."""


@dataclass
class PromptBundle:
    system: str
    user: str
    citation_tags: list[str]


def build_prompt(question: str, chunks_with_scores: list[tuple[Chunk, float]]) -> PromptBundle:
    tags = [f"S{i}" for i in range(1, len(chunks_with_scores) + 1)]
    return PromptBundle(system=SYSTEM_PROMPT, user=build_user_prompt(question, chunks_with_scores), citation_tags=tags)


_CITATION_TAG_RE = re.compile(r"\[S(\d+)\]")


def expand_citations(answer: str, chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Replace each short `[Sn]` tag the model emits with a full, deterministic
    citation built from the actual retrieved chunk's own metadata - e.g.
    `[S2]` -> `(Section: 2.6.3 1830 PSS-8 Fan Unit (8FAN), p.104)`. The model
    only ever has to reproduce a short tag (reliable); the human-readable
    citation format is generated by our own code every time (always
    correctly formatted), never left to the model to paraphrase. A tag whose
    index is out of range (a model slip) is left as-is rather than guessed."""

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(chunks_with_scores):
            chunk, _ = chunks_with_scores[idx]
            pages = (
                f"p.{chunk.page_start}"
                if chunk.page_end == chunk.page_start
                else f"p.{chunk.page_start}-{chunk.page_end}"
            )
            return f"(Section: {chunk.heading_path}, {pages})"
        return m.group(0)

    return _CITATION_TAG_RE.sub(_replace, answer)


def looks_like_refusal(answer: str) -> bool:
    return config.REFUSAL_STRING.lower() in answer.lower()


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\(Section:.*?p\.\d+", answer)) or bool(re.search(r"\[S\d+", answer))
