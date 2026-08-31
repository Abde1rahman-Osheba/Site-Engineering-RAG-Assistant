"""
"Blind" dry-run of the full RAGPipeline against the REAL document and the
REAL fixed 8-question evaluation set (Part D), using the offline extractive
provider (no API key needed/available in this environment).

This is intentionally "blind": it prints ONLY structural/metadata facts
(page numbers touched, similarity scores, answer length, refusal/citation
flags) - never the retrieved passage text nor the generated answer text
itself, since both are derived from the confidential source document. This
lets us confirm the full retrieval -> prompt -> generation -> refusal
pipeline runs correctly end to end on the real evaluation questions without
the assistant ever reading/displaying the document's actual content.

The REAL, human-readable evaluation table (Part D deliverable) is produced
by the notebook when the user runs it themselves.

Run: py -m scripts.blind_eval_dryrun
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction import load_extracted
from src.chunking import build_chunks
from src.embedding_index import TextEmbedder, build_or_load_index, FaissIndex
from src.bm25_hybrid import build_bm25_index
from src.pipeline import RAGPipeline
from src.evaluation import FIXED_QUESTIONS


def main():
    pages = load_extracted()
    chunks = build_chunks(pages)
    embedder = TextEmbedder()
    text_index, _ = build_or_load_index(chunks, embedder=embedder)
    faiss_index = FaissIndex(embeddings=text_index.embeddings, chunk_ids=text_index.chunk_ids)
    bm25_index = build_bm25_index(chunks)

    pipeline = RAGPipeline(
        chunks=chunks,
        text_index=text_index,
        embedder=embedder,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
    )

    print(f"{'#':<3}{'pages touched':<16}{'top score':<11}{'#chunks':<9}{'refused':<9}{'cited':<7}{'answer_len':<11}")
    for item in FIXED_QUESTIONS:
        for use_hybrid, use_rerank, label in [(False, False, "manual"), (True, False, "hybrid")]:
            result = pipeline.answer(item["question"], provider="offline", use_hybrid=use_hybrid, use_rerank=use_rerank)
            pages_touched = sorted({c.page_start for c, _ in result.retrieval.chunks_with_scores})
            top_score = result.retrieval.chunks_with_scores[0][1] if result.retrieval.chunks_with_scores else 0.0
            print(
                f"{item['id']:<3}{str(pages_touched):<16}{top_score:<11.3f}"
                f"{len(result.retrieval.chunks_with_scores):<9}{str(result.refused):<9}"
                f"{str(result.has_citation):<7}{len(result.answer):<11}  [{label}]"
            )

    print("\n(No passage text or answer text is printed above - only page numbers, scores, and flags,")
    print(" per this project's constraint that the assistant never reads/displays the source document's content.)")


if __name__ == "__main__":
    main()
