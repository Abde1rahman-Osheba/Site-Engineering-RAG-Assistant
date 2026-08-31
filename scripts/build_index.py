"""
Rebuilds the entire persisted index from scratch: extraction -> chunking ->
embedding -> manual/FAISS index -> BM25 -> (best-effort) CLIP image index.
Prints per-stage and total timing so the "rebuilds in under a minute"
deliverable is directly checkable, not just claimed.

Nothing here is committed to git (see .gitignore - data/ is all
regenerated-at-runtime, re-derived from the source PDF, never hand-edited) -
this script is how you get it back.

Usage:
    py -m scripts.build_index
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: E402  (loads .env)
from src import config  # noqa: E402
from src.extraction import extract_page_range, save_extracted, extract_images  # noqa: E402
from src.chunking import build_chunks, save_chunks  # noqa: E402
from src.embedding_index import TextEmbedder, build_or_load_index, FaissIndex  # noqa: E402
from src.bm25_hybrid import build_bm25_index  # noqa: E402


def main() -> None:
    if not config.SOURCE_PDF.exists():
        print(f"ERROR: {config.SOURCE_PDF.name} not found at the project root. "
              f"Place your own copy there first (see README.md).")
        sys.exit(1)

    t_start = time.time()

    t0 = time.time()
    pages = extract_page_range(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
    save_extracted(pages)
    print(f"[1/5] Extraction:  {time.time()-t0:6.2f}s  ({len(pages)} pages)")

    t0 = time.time()
    chunks = build_chunks(pages)
    save_chunks(chunks)
    print(f"[2/5] Chunking:    {time.time()-t0:6.2f}s  ({len(chunks)} chunks)")

    t0 = time.time()
    embedder = TextEmbedder()
    text_index, manifest = build_or_load_index(chunks, embedder=embedder)
    print(f"[3/5] Embedding:   {time.time()-t0:6.2f}s  (model load + encode; "
          f"cache_hit={manifest['_cache_hit']})")

    t0 = time.time()
    FaissIndex(embeddings=text_index.embeddings, chunk_ids=text_index.chunk_ids).save()
    bm25_index = build_bm25_index(chunks)
    print(f"[4/5] FAISS+BM25:  {time.time()-t0:6.2f}s  ({len(bm25_index.chunk_ids)} chunks indexed)")

    t0 = time.time()
    try:
        from src.multimodal import ClipEmbedder, records_from_extraction, build_or_load_image_index

        figures = extract_images(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
        image_records = records_from_extraction(figures)
        clip_embedder = ClipEmbedder()
        _, image_manifest = build_or_load_image_index(image_records, embedder=clip_embedder)
        print(f"[5/5] CLIP images: {time.time()-t0:6.2f}s  ({len(figures)} figures; "
              f"cache_hit={image_manifest['_cache_hit']})")
    except Exception as e:  # pragma: no cover - multimodal is a bonus extension, not required
        print(f"[5/5] CLIP images: skipped ({e})")

    total = time.time() - t_start
    print(f"\nTotal rebuild time: {total:.1f}s")
    if total < 60:
        print("(under a minute, as required)")


if __name__ == "__main__":
    main()
