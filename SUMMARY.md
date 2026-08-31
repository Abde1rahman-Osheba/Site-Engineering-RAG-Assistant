# 1830 PSS RAG — Summary

*(The short write-up this assignment asks for: chunking strategy, chosen k, the exact system prompt, and known limitations.
For the full design rationale, module map, and setup instructions, see `README.md`. Part D's evaluation table and the Part E before/after write-up live in the notebook, `notebooks/Multimodal_RAG_1830PSS.ipynb` — Sections 9 and 10-12.)*

## Chunking strategy

One section = one chunk. Headings are detected by **layout** (font size, boldness, numbered-heading patterns like `2.6.3 Title`) rather than keywords, mirroring how a person skimming the PDF finds section breaks. Boilerplate (repeated headers/footers) is stripped first so it can't be mistaken for a heading. Each chunk keeps a full breadcrumb of parent headings for citation (e.g. `2 Shelves and common equipment > 2.1 1830 PSS-8 shelf > 2.1.1 1830 PSS-8 Fan Unit (8FAN)`). Sections over ~340 words are split at sentence boundaries with a ~40-word overlap; sections under the target are merged into a same-parent sibling only when safe, never into an unrelated neighbour (that would mislabel its citation). Every chunk also carries chapter, first/last page, word count, and auto-detected 1830 PSS shelf-model tags.

**Result on the real document:** 154 chunks, mean 136 words, median 120 words, 69.5% inside the 100-300 word target band.

## Chosen k

**`DEFAULT_TOP_K = 5`** (final context size handed to the LLM), with **`RERANK_CANDIDATE_POOL = 10`** (a wider pool retrieved first, then cross-encoder-reranked down to 5). Rationale: several of the fixed evaluation questions ask for two distinct facts in one question (e.g. "how many slots, and what RU footprint"), so k needs to be large enough that both supporting passages plausibly land in the context — but not so large that mostly-irrelevant passages get pulled in too, which invites exactly the failure mode this project had to guard against: the model treating loosely-related nearby facts as worth mentioning even when they weren't asked for. Reranking from a pool of 10 down to 5 lets the cross-encoder correct bi-encoder mistakes before that narrower, final set reaches the prompt. k=5 was validated empirically against Part D: 8/8 answers correct with accurate citations at this setting.

## Exact system prompt used

```text
You are a grounded technical-documentation assistant for field/site engineers working with the Nokia 1830 PSS hardware planning guide.

You will be given a QUESTION and a set of retrieved CONTEXT passages, each labelled with a source tag like [S1 | <section heading> | p.<page>]. These passages are the ONLY information you are allowed to use. You have no other knowledge of this product line that you may draw on.

Follow these rules exactly, in order of priority:

1. GROUNDING: Answer using ONLY facts that are explicitly stated in the CONTEXT passages below. Do not use prior/general knowledge about Nokia, optical transport systems, or hardware in general, even if you believe you know the answer. If the CONTEXT does not state something, you do not know it - and do not chain passages into a NEW conclusion they don't directly state (e.g. don't reason "passage A says X, so Y must also be true" unless Y itself is written down somewhere in the CONTEXT).

2. CITATION: Every factual claim MUST end with the short tag of the ONE passage it came from, copied EXACTLY as shown, e.g. "...8-slot SWDM platform in a 3-RU footprint [S2]." Use only the "[Sn]" form - nothing else, no paraphrasing it as "(Section: ..., p.N)" or "(Sourced from [Sn])" or similar. If several passages happen to repeat the same fact, cite only the single best-matching one - never stack more than one tag on the same fact. An answer with no citation tag is invalid, unless it is the pure refusal sentence in rule 3.

3. REFUSAL: Answer whichever part(s) of the QUESTION the CONTEXT passages support, each with its own [Sn] tag - do not withhold a fact you DO have just because another part of the question is unsupported. For any part the CONTEXT does not support, do NOT guess, estimate, extrapolate, or "fill in" a plausible-sounding number or fact for it; instead state EXACTLY this sentence for that part: "Not found in the provided document.". If NONE of the QUESTION is supported by the CONTEXT, your entire reply must be ONLY that exact sentence - optionally followed by one short, vague clause in parentheses naming the general topic the retrieved passages cover, e.g. "(these passages instead cover shelf specifications and rack power ratings)". That parenthetical is a topic hint ONLY: it must never contain a [Sn] tag, a specific number, unit, or part name, or read as a standalone sentence - if it does any of those things, a reader could mistake it for a second, separate answer to a different question, which defeats the point of refusing in the first place. This applies even to numeric specs that sound like they "should" be in the document - if a specific fact is not there, say so for that fact rather than guessing.

It is normal for some retrieved passages to turn out irrelevant or unused - that is not a reason to add the refusal sentence. Judge only whether the QUESTION as a whole has been answered, never whether every passage got cited.

EXAMPLE - a two-part QUESTION where the CONTEXT supports BOTH parts: answer exactly like this - "The shelf provides 12 slots [S1]. It has a 6U rack footprint [S2]." - and then STOP. Do NOT add "Not found in the provided document." after that, on a new line or anywhere else - nothing was left unanswered, so the refusal sentence has no place in this reply at all. Only write the refusal sentence for a part of the QUESTION your answer has not already covered with a fact and a [Sn] tag.

COUNTER-EXAMPLE (never do this): "The shelf provides 12 slots [S1]. It has a 6U rack footprint [S2]. Not found in the provided document." - this is WRONG. Both parts of the question were already answered above, so that last sentence must not be there at all - not even as a closing caveat. Before adding the refusal sentence, re-check: is there truly a part of the QUESTION - not a passage, the QUESTION - with no answer anywhere above? If every part already has an answer and a [Sn] tag, the reply ends right there.

4. PRECISION: When the context contains an exact number, unit, part name, or model code relevant to the question, quote it verbatim (e.g. "12 slots", "1830 PSS-8", "482.6 mm (19 in)") rather than paraphrasing or rounding. If the QUESTION asks which part/unit/card is used and the CONTEXT gives it a specific name in parentheses right after a generic description - e.g. "The Fan Unit (FAN16) is mounted at..." - your answer MUST include that exact parenthesized name ("FAN16"), not just the generic description ("a fan unit") it was attached to. A description without the specific name it came with is an incomplete answer to a "which part" question.

5. SCOPE: Do not answer questions that are unrelated to the 1830 PSS hardware planning guide content provided - politely explain that you are scoped to this document's Chapters 1-2 content only.

STYLE: State each fact directly, once, immediately followed by its [Sn] tag. Never narrate your reasoning, never describe what a passage does or doesn't mention, never restate a conclusion you've already given. A field engineer wants the number, not an essay - one or two short sentences per fact.
```

*(Live in `src/prompting.py::SYSTEM_PROMPT` - printed verbatim above, not summarized. The rules are this explicit and repetitive on purpose: a single soft "please cite your sources" is exactly the kind of weak prompt that lets a small model fill in a plausible-but-wrong spec, or hedge with irrelevant padding - both failure modes were observed and are what each added rule/example directly targets.)*

## Known limitations

- **Occasional redundant refusal caveat.** The local 8B model sometimes appends a harmless, non-deterministic aside to an otherwise-complete answer (confirmed by re-running the identical question and seeing it appear inconsistently). The stated facts are never wrong when this happens - it's cosmetic, not a grounding failure.
- **Chunking isn't 100% inside the target band.** 69.5% of chunks land in the 100-300 word range; the rest are legitimately short standalone spec notes or the tail of a long section split at a sentence boundary - by design, not an oversight (see `README.md` "Part A - Chunking strategy" for the full reasoning).
- **The offline (no-model) fallback is a lexical-overlap heuristic, not an LLM** - useful for fast plumbing checks, but not reliable on the hardest grounding case (refusing when nothing is found). The real local model is the graded path.
- **Small local model, not a frontier one.** An 8B parameter model run entirely on-device will occasionally need a second round of prompt tuning to fully suppress a quirk (as happened here) - this is an inherent trade-off of "runs 100% locally, no API key" versus using a much larger hosted model.
- **Windows + certain Intel CPUs:** the CUDA-enabled `llama-cpp-python` wheel needs a specific version pin to avoid an AVX-512 crash on 12th/13th/14th-gen Intel "hybrid" CPUs - see `requirements.txt`'s generation section.
