"""
Structural-only validation of the multimodal (CLIP) image index against the
REAL extracted figures. Prints only counts/dimensions/timings - never
displays or describes any image content (this script does not call Read on
any image, and CLIP embeddings are numeric vectors, not descriptions).

Run: py -m scripts.multimodal_selftest
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.extraction import extract_images
from src.multimodal import ClipEmbedder, records_from_extraction, build_or_load_image_index, search_images


def main():
    print("Extracting embedded figures (counts only)...")
    figures = extract_images(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
    print(f"Found {len(figures)} embedded raster images in the page range.")
    records = records_from_extraction(figures)

    print("\nBuilding CLIP image index (this downloads clip-ViT-B-32 on first run)...")
    t0 = time.time()
    embedder = ClipEmbedder()
    image_index, manifest = build_or_load_image_index(records, embedder=embedder)
    print(f"Done in {time.time()-t0:.1f}s")
    print("Image index manifest:", manifest)
    print("Embeddings shape:", image_index.embeddings.shape)

    print("\nSanity probe: text->image retrieval mechanics (query is generic, not doc-derived)")
    hits = search_images("a photo of an electronic equipment shelf with slots", image_index, embedder, top_k=3)
    print("Top-3 image ids + scores (ids only, not images):", [(iid, round(s, 4)) for iid, s in hits])

    print("\nCache round-trip check...")
    t0 = time.time()
    image_index_2, manifest_2 = build_or_load_image_index(records, embedder=embedder)
    print(f"Second call took {time.time()-t0:.2f}s, cache_hit={manifest_2.get('_cache_hit')}")
    assert manifest_2.get("_cache_hit") is True

    print("\nALL MULTIMODAL STRUCTURAL CHECKS PASSED")


if __name__ == "__main__":
    main()
