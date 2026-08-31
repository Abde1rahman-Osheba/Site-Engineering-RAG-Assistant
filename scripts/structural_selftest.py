"""
Structural-only validation against the REAL 1830 PSS PDF.

IMPORTANT / CONFIDENTIALITY: this script prints ONLY aggregate numbers,
counts, timings, and hashes about the extracted/chunked/embedded content -
never the actual extracted text, headings, or generated answers. This lets
us confirm the pipeline runs correctly end-to-end against the real,
restricted page range (47-166) without the assistant (or this transcript)
ever displaying the source document's content, per this project's
confidentiality constraint.

Run: py -m scripts.structural_selftest
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.extraction import extract_page_range, save_extracted, extraction_is_fresh, extract_images, render_page_thumbnails
from src.chunking import build_chunks, save_chunks, chunk_stats
from src.embedding_index import TextEmbedder, ManualCosineIndex, FaissIndex, build_or_load_index
from src.bm25_hybrid import build_bm25_index


def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    section("0. Source file check (path/size/hash only - never content)")
    print(f"Source PDF exists: {config.SOURCE_PDF.exists()}")
    print(f"Source PDF size (bytes): {config.SOURCE_PDF.stat().st_size}")
    print(f"Configured page range: {config.PAGE_START}-{config.PAGE_END} (Chapters 1-2)")

    section("1. Extraction (physical pages 47-166)")
    t0 = time.time()
    pages = extract_page_range(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
    manifest = save_extracted(pages)
    print(f"Extracted in {time.time()-t0:.1f}s")
    print("Extraction manifest (counts only):", manifest)
    assert manifest["num_pages_extracted"] == (config.PAGE_END - config.PAGE_START + 1)

    section("2. Chunking (heading-aware + metadata)")
    t0 = time.time()
    chunks = build_chunks(pages)
    save_chunks(chunks)
    stats = chunk_stats(chunks)
    print(f"Chunked in {time.time()-t0:.1f}s")
    print("Chunk stats (counts/word-counts only):", stats)
    pct_in_target = 100 * stats["n_chunks_in_target_100_300"] / max(stats["n_chunks"], 1)
    print(f"% of chunks within the 100-300 word target: {pct_in_target:.1f}%")

    section("3. Embedding + indexing (manual cosine index, cached to disk)")
    embedder = TextEmbedder()
    text_index, idx_manifest = build_or_load_index(chunks, embedder=embedder)
    print("Index manifest (no text):", {k: v for k, v in idx_manifest.items() if k != "chunk_stats"})
    print(f"Embeddings shape: {text_index.embeddings.shape}")

    section("4. FAISS second implementation - agreement check with manual index")
    faiss_index = FaissIndex(embeddings=text_index.embeddings, chunk_ids=text_index.chunk_ids)
    faiss_index.save()
    # sanity probe with a random vector (not real query text) purely to confirm
    # both backends return the SAME chunk ids / scores for the SAME vector -
    # this never touches document semantics, just numerical agreement.
    import numpy as np

    rng = np.random.default_rng(config.RANDOM_SEED)
    probe = rng.normal(size=text_index.embeddings.shape[1]).astype("float32")
    probe /= np.linalg.norm(probe)
    manual_hits = text_index.search(probe, top_k=10)
    faiss_hits = faiss_index.search(probe, top_k=10)
    agreement = len(set(cid for cid, _ in manual_hits) & set(cid for cid, _ in faiss_hits)) / 10
    max_score_diff = max(abs(m[1] - f[1]) for m, f in zip(manual_hits, faiss_hits))
    print(f"Manual vs FAISS top-10 id agreement on random probe vector: {agreement:.0%}")
    print(f"Max score difference: {max_score_diff:.6f}")
    assert agreement == 1.0 and max_score_diff < 1e-4, "manual and FAISS (both exact) must agree"

    section("5. BM25 index build (Part E hybrid ingredient)")
    t0 = time.time()
    bm25 = build_bm25_index(chunks)
    print(f"BM25 built over {len(bm25.chunk_ids)} chunks in {time.time()-t0:.1f}s")

    section("6. Cache round-trip: re-running build_or_load_index should hit cache")
    t0 = time.time()
    text_index_2, manifest_2 = build_or_load_index(chunks, embedder=embedder)
    print(f"Second call took {time.time()-t0:.2f}s, cache_hit={manifest_2.get('_cache_hit')}")
    assert manifest_2.get("_cache_hit") is True, "second call should load from disk, not re-embed"

    section("7. Multimodal: extract embedded figures + page renders (counts only)")
    t0 = time.time()
    figures = extract_images(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
    print(f"Embedded raster images extracted: {len(figures)} in {time.time()-t0:.1f}s")
    # Page renders are the heavier fallback (1 PNG per page) - only sample a
    # handful of pages here to keep this structural check fast; the notebook
    # runs the full range when the user executes it.
    sample_pages = list(range(config.PAGE_START, min(config.PAGE_START + 5, config.PAGE_END + 1)))
    renders = render_page_thumbnails(config.SOURCE_PDF, sample_pages[0], sample_pages[-1])
    print(f"Sample page renders: {len(renders)} (full range built by the notebook)")

    section("ALL STRUCTURAL SELF-TESTS PASSED (real PDF - counts/hashes only, no content displayed)")


if __name__ == "__main__":
    main()
