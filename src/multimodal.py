"""
Multimodal extension — images join the retrieval index.

The planning guide's shelf/fan-unit sections are full of diagrams and
front-panel photos; a text-only RAG throws that information away. This
module extracts embedded figures (and, as a fallback, full-page renders for
pages whose key content is a vector-drawn diagram/table rather than a raster
image) and embeds them with CLIP (`clip-ViT-B-32`, loaded through
sentence-transformers) - the same model family can embed *both* images and
text into one shared vector space, so a plain-text query can retrieve the
most relevant figure without any separate captioning step.

This keeps the same "no black-box `ask-the-doc` library, implement top-k
search yourself" spirit as Part B: image retrieval reuses the exact same
`ManualCosineIndex` used for text (see embedding_index.py) - only the
embedding model differs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config
from .embedding_index import ManualCosineIndex

IMAGE_EMBEDDINGS_NPY = config.INDEX_DIR / "image_embeddings.npy"
IMAGE_MANIFEST = config.INDEX_DIR / "image_index_manifest.json"
IMAGE_RECORDS_JSON = config.INDEX_DIR / "image_records.json"


@dataclass
class ImageRecord:
    image_id: str
    path: str
    page_number: int
    width: int
    height: int
    kind: str  # "embedded_figure" | "page_render"


class ClipEmbedder:
    """Wraps sentence-transformers' CLIP integration. `.encode_images` and
    `.encode_text` share one embedding space, which is exactly what lets a
    natural-language query retrieve a relevant diagram directly."""

    def __init__(self, model_name: str = config.CLIP_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_images(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        vecs = self.model.encode(images, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def encode_text_one(self, text: str) -> np.ndarray:
        return self.encode_text([text])[0]


def records_from_extraction(embedded_figures: list[dict], page_renders: list[dict] | None = None) -> list[ImageRecord]:
    records = [
        ImageRecord(
            image_id=f"fig_{r['page_number']:04d}_{i:02d}",
            path=r["path"],
            page_number=r["page_number"],
            width=r["width"],
            height=r["height"],
            kind="embedded_figure",
        )
        for i, r in enumerate(embedded_figures)
    ]
    if page_renders:
        records += [
            ImageRecord(
                image_id=f"page_{r['page_number']:04d}",
                path=r["path"],
                page_number=r["page_number"],
                width=r["width"],
                height=r["height"],
                kind="page_render",
            )
            for r in page_renders
        ]
    return records


def _records_fingerprint(records: list[ImageRecord]) -> str:
    return str(hash(tuple((r.image_id, r.path) for r in records)))


def build_or_load_image_index(
    records: list[ImageRecord],
    embedder: ClipEmbedder | None = None,
    force_rebuild: bool = False,
) -> tuple[ManualCosineIndex, dict]:
    """Same cache/version pattern as the text index (embedding_index.py)."""
    if not records:
        raise ValueError("No image records to index.")

    fp = _records_fingerprint(records)
    if not force_rebuild and IMAGE_EMBEDDINGS_NPY.exists() and IMAGE_MANIFEST.exists():
        with open(IMAGE_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("fingerprint") == fp and manifest.get("model_name") == config.CLIP_MODEL:
            embeddings = np.load(IMAGE_EMBEDDINGS_NPY)
            with open(IMAGE_RECORDS_JSON, "r", encoding="utf-8") as f:
                saved_records = json.load(f)
            ids = [r["image_id"] for r in saved_records]
            manifest["_cache_hit"] = True
            return ManualCosineIndex(embeddings=embeddings, chunk_ids=ids), manifest

    if embedder is None:
        embedder = ClipEmbedder()

    t0 = time.time()
    embeddings = embedder.encode_images([r.path for r in records])
    elapsed = time.time() - t0

    ids = [r.image_id for r in records]
    np.save(IMAGE_EMBEDDINGS_NPY, embeddings)
    with open(IMAGE_RECORDS_JSON, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in records], f, indent=2)

    manifest = {
        "model_name": config.CLIP_MODEL,
        "n_images": len(records),
        "fingerprint": fp,
        "encode_seconds": round(elapsed, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_cache_hit": False,
    }
    with open(IMAGE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return ManualCosineIndex(embeddings=embeddings, chunk_ids=ids), manifest


def load_image_records() -> dict[str, ImageRecord]:
    with open(IMAGE_RECORDS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {r["image_id"]: ImageRecord(**r) for r in raw}


def search_images(query_text: str, index: ManualCosineIndex, embedder: ClipEmbedder, top_k: int = 3) -> list[tuple[str, float]]:
    q = embedder.encode_text_one(query_text)
    return index.search(q, top_k=top_k)
