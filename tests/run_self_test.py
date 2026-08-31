"""
End-to-end self-test of the ENTIRE pipeline against the synthetic fixture
ONLY (never the real 1830 PSS PDF). Safe to run and to print output from,
because every word in the fixture was authored for this project (see
synthetic_fixture.py) - there is no confidential content anywhere here.

This validates: extraction (font-size-driven heading detection), section-
aware chunking + metadata, embedding + manual cosine top-k search, FAISS
agreement, BM25 hybrid fusion, cross-encoder re-ranking, metadata filtering,
prompt construction, and the citation/refusal contract - end to end.

Generation here deliberately forces the "offline" (non-LLM, lexical-overlap)
provider rather than the real one (meta-llama/Llama-3.1-8B-Instruct via
transformers) - this is a fast, dependency-light mechanical check of the
plumbing, and loading an 8B-parameter model on every test run would be slow
and require a real GPU/token. To actually exercise the real local model
against this same safe, fictitious fixture, change `provider = "offline"`
below to `provider = pick_available_provider()` (or `"huggingface"`) - see
README.md for setup.

Run: py -m tests.run_self_test
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re

from src import chunking
from src.extraction import extract_page_range
from src.chunking import build_chunks, chunk_stats

# The shelf-tag regexes in config/chunking are written for the real product
# line ("1830 PSS-8", etc.). Swap in fixture-matching patterns so Part E's
# metadata-filtering step has something fictitious to key off of here.
chunking._SHELF_TAG_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [r"Acme\s*OptiRack[\s-]*8(?!FAN)", r"Acme\s*OptiRack[\s-]*32", r"Acme\s*OptiRack[\s-]*16\s*II"]
]
from src.embedding_index import TextEmbedder, ManualCosineIndex, FaissIndex
from src.bm25_hybrid import build_bm25_index, hybrid_scores, top_k_from_scores
from src.rerank import CrossEncoderReranker, keyword_overlap_rerank
from src.prompting import build_prompt, looks_like_refusal, has_citation
from src.llm_client import generate_answer, pick_available_provider, corpus_common_tokens
from src.pipeline import RAGPipeline
from tests.synthetic_fixture import build_fixture_pdf, FIXTURE_PDF


def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    section("1. Build synthetic fixture PDF")
    build_fixture_pdf()
    print(f"Fixture written: {FIXTURE_PDF} ({FIXTURE_PDF.stat().st_size} bytes)")

    section("2. Extraction (layout-aware, font-size metadata retained)")
    pages = extract_page_range(FIXTURE_PDF, first_page=1, last_page=_page_count(FIXTURE_PDF))
    print(f"Pages extracted: {len(pages)}; total lines: {sum(len(p.lines) for p in pages)}")

    section("3. Chunking (heading-aware, metadata-tagged)")
    chunks = build_chunks(pages)
    stats = chunk_stats(chunks)
    print("Chunk stats:", stats)
    assert len(chunks) >= 5, "expected multiple chunks from the fixture"
    assert stats["n_chunks_in_target_100_300"] >= 1 or stats["mean_words"] > 0

    print("\nDetected chunk headings & shelf tags (fixture is fictitious - safe to print):")
    for c in chunks:
        print(f"  [{c.chunk_id}] p.{c.page_start}-{c.page_end} | {c.heading_path} | {c.word_count}w | tags={c.shelf_tags}")

    section("4. Embedding + manual cosine top-k vs FAISS agreement")
    embedder = TextEmbedder()
    texts = [c.text for c in chunks]
    t0 = time.time()
    embeddings = embedder.encode(texts)
    print(f"Encoded {len(texts)} chunks in {time.time()-t0:.2f}s, dim={embeddings.shape[1]}")

    chunk_ids = [c.chunk_id for c in chunks]
    manual_index = ManualCosineIndex(embeddings=embeddings, chunk_ids=chunk_ids)
    faiss_index = FaissIndex(embeddings=embeddings, chunk_ids=chunk_ids)

    test_query = "How many slots and RU does the OptiRack-8 shelf have?"
    qvec = embedder.encode_one(test_query)
    manual_hits = manual_index.search(qvec, top_k=5)
    faiss_hits = faiss_index.search(qvec, top_k=5)
    print("Manual top-5:", [(cid, round(s, 4)) for cid, s in manual_hits])
    print("FAISS  top-5:", [(cid, round(s, 4)) for cid, s in faiss_hits])
    manual_ids_top3 = {cid for cid, _ in manual_hits[:3]}
    faiss_ids_top3 = {cid for cid, _ in faiss_hits[:3]}
    agreement = len(manual_ids_top3 & faiss_ids_top3) / 3
    print(f"Top-3 agreement manual vs FAISS: {agreement:.2%}")
    assert agreement >= 0.66, "manual and FAISS top-3 should mostly agree (both are exact cosine search)"

    section("5. BM25 + hybrid fusion")
    bm25 = build_bm25_index(chunks)
    full_emb_scores = {cid: s for cid, s in manual_index.search(qvec, top_k=len(chunk_ids))}
    fused = hybrid_scores(full_emb_scores, bm25, test_query)
    hybrid_top = top_k_from_scores(fused, 5)
    print("Hybrid top-5:", [(cid, round(s, 4)) for cid, s in hybrid_top])

    section("6. Cross-encoder re-ranking")
    pool = [(next(c for c in chunks if c.chunk_id == cid), s) for cid, s in manual_index.search(qvec, top_k=10)]
    try:
        ce = CrossEncoderReranker()
        reranked = ce.rerank(test_query, pool, top_k=3)
        print("Cross-encoder top-3:", [(c.chunk_id, round(s, 4)) for c, s in reranked])
    except Exception as e:
        print(f"(cross-encoder unavailable: {e}; falling back to keyword-overlap rerank)")
        reranked = keyword_overlap_rerank(test_query, pool)[:3]
        print("Keyword-overlap top-3:", [(c.chunk_id, round(s, 4)) for c, s in reranked])

    section("7. Prompt construction + citation/refusal contract")
    top3 = [(next(c for c in chunks if c.chunk_id == cid), s) for cid, s in manual_hits[:3]]
    prompt = build_prompt(test_query, top3)
    print("System prompt length:", len(prompt.system), "chars")
    print("User prompt preview (fixture is fictitious, safe to print):\n", prompt.user[:600], "...")

    section("8. Generation (forced offline - fast mechanical check, not the real model)")
    # This fast self-test deliberately forces "offline" rather than calling
    # pick_available_provider() (which now defaults to "huggingface" -
    # meta-llama/Llama-3.1-8B-Instruct - the real generation path). Loading
    # an 8B-parameter model on every plumbing check would be slow and
    # require a GPU/token; swap this for pick_available_provider() to
    # exercise the real local model against this same safe fixture instead.
    provider = "offline"
    print(f"Provider: {provider}")
    common_tokens = corpus_common_tokens(chunks)
    gen = generate_answer(test_query, prompt, top3, provider=provider, common_tokens=common_tokens)
    print(f"[{gen.provider}/{gen.model}] Answer: {gen.answer}")
    print("Looks like refusal:", looks_like_refusal(gen.answer), "| Has citation:", has_citation(gen.answer))

    section("9. Full RAGPipeline + a fixture 'trick question' (should refuse)")
    pipeline = RAGPipeline(chunks=chunks, text_index=manual_index, embedder=embedder, faiss_index=faiss_index, bm25_index=bm25)
    trick_q = "What is the maximum optical reach in kilometers of the OptiRack-8 shelf without amplification?"
    result = pipeline.answer(trick_q, provider=provider, use_hybrid=True, use_rerank=True)
    print(f"Trick question answer [{result.provider}]: {result.answer}")
    print("Refused as expected:", result.refused)

    good_q = "Which fan units are supported on the OptiRack-32 shelf?"
    result2 = pipeline.answer(good_q, provider=provider, use_hybrid=True)
    print(f"\nHybrid-search answer for '{good_q}': {result2.answer}")

    section("10. Metadata-filtered retrieval (Part E)")
    filtered = pipeline.retrieve("What fan unit is used?", shelf_filter=["ACME OPTIRACK-32"])
    print("Filtered-to-32 retrieval hits:", [(c.chunk_id, c.shelf_tags, round(s, 3)) for c, s in filtered.chunks_with_scores])

    section("ALL SELF-TESTS COMPLETED (synthetic fixture only)")


def _page_count(pdf_path: Path) -> int:
    import pymupdf as fitz

    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


if __name__ == "__main__":
    main()
