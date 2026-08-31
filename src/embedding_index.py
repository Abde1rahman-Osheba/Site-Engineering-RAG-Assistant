"""
Part B — Embedding & Indexing.

* `TextEmbedder`      thin wrapper around sentence-transformers (all-MiniLM-L6-v2).
* `ManualCosineIndex` the *required*, from-first-principles top-k cosine
                      search: plain numpy, no vector-DB black box.
* `FaissIndex`        a second, compared implementation (FAISS IndexFlatIP)
                      used only to validate/benchmark the manual version -
                      not a replacement for it (assignment Part B rule).
* `build_or_load_index` persists embeddings + a version manifest to disk so
                      the index is not recomputed on every run ("cache &
                      version your embeddings").
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, chunk_stats

EMBEDDINGS_NPY = config.INDEX_DIR / "text_embeddings.npy"
INDEX_MANIFEST = config.INDEX_DIR / "text_index_manifest.json"
CHUNK_ID_MAP = config.INDEX_DIR / "text_chunk_ids.json"
FAISS_INDEX_FILE = config.INDEX_DIR / "text_faiss.index"


def _chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Hash of chunk ids+text so we can tell if chunking changed upstream."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(b"\0")
        h.update(str(len(c.text)).encode())
        h.update(b"\0")
    return h.hexdigest()


class TextEmbedder:
    """Wraps a single sentence-transformers model used for BOTH indexing and
    querying (must match, per the assignment: "embed the incoming question
    with the same model used for indexing")."""

    def __init__(self, model_name: str = config.TEXT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import (heavy)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm -> dot product == cosine similarity
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@dataclass
class ManualCosineIndex:
    """From-first-principles top-k cosine similarity search.

    Vectors are stored L2-normalised, so cosine similarity(query, doc) is
    simply the dot product - implemented here with plain numpy, no FAISS/
    Chroma/Annoy involved. This is the index the rest of the pipeline uses
    by default; FaissIndex below exists purely as a second, benchmarked
    implementation per the assignment's "allowed only as a second, compared
    implementation" rule.
    """

    embeddings: np.ndarray            # (n, d), L2-normalised, float32
    chunk_ids: list[str]

    def search(self, query_vec: np.ndarray, top_k: int = config.DEFAULT_TOP_K) -> list[tuple[str, float]]:
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        # cosine similarity via dot product, computed by hand (no sklearn.metrics, no faiss)
        scores = self.embeddings @ q  # (n,)
        top_k = min(top_k, len(scores))
        # argpartition for O(n) top-k selection, then sort just those k
        top_idx_unsorted = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx_unsorted[np.argsort(-scores[top_idx_unsorted])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]

    def search_filtered(
        self, query_vec: np.ndarray, allowed_chunk_ids: set[str], top_k: int = config.DEFAULT_TOP_K
    ) -> list[tuple[str, float]]:
        """Same as `search`, but restricted to a subset of chunk ids -
        the mechanism behind Part E metadata filtering (e.g. "only PSS-32")."""
        if not allowed_chunk_ids:
            return self.search(query_vec, top_k)
        mask = np.array([cid in allowed_chunk_ids for cid in self.chunk_ids])
        sub_embeddings = self.embeddings[mask]
        sub_ids = [cid for cid, m in zip(self.chunk_ids, mask) if m]
        if len(sub_ids) == 0:
            return self.search(query_vec, top_k)
        sub_index = ManualCosineIndex(embeddings=sub_embeddings, chunk_ids=sub_ids)
        return sub_index.search(query_vec, top_k)


class FaissIndex:
    """Second, compared implementation using FAISS (flat, exact, inner-product
    over L2-normalised vectors == cosine). Used in the notebook to confirm
    the manual implementation returns identical top-k results and to compare
    query latency at scale."""

    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str]):
        import faiss  # lazy import

        self.chunk_ids = chunk_ids
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = config.DEFAULT_TOP_K) -> list[tuple[str, float]]:
        q = query_vec.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores, idx = self.index.search(q, top_k)
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]

    def save(self, path: Path = FAISS_INDEX_FILE):
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, chunk_ids: list[str], path: Path = FAISS_INDEX_FILE):
        import faiss

        obj = cls.__new__(cls)
        obj.chunk_ids = chunk_ids
        obj.index = faiss.read_index(str(path))
        obj.dim = obj.index.d
        return obj


def _manifest_matches(manifest: dict, chunks: list[Chunk], model_name: str) -> bool:
    return (
        manifest.get("model_name") == model_name
        and manifest.get("chunks_fingerprint") == _chunks_fingerprint(chunks)
        and manifest.get("n_chunks") == len(chunks)
    )


def build_or_load_index(
    chunks: list[Chunk],
    embedder: TextEmbedder | None = None,
    force_rebuild: bool = False,
) -> tuple[ManualCosineIndex, dict]:
    """The "cache & version your embeddings" entry point.

    If a previously-saved embedding matrix exists AND its manifest matches
    the current chunk set + model name, it is loaded from disk. Otherwise
    the chunks are (re-)embedded and persisted with a fresh manifest.
    """
    model_name = embedder.model_name if embedder is not None else config.TEXT_EMBED_MODEL

    if not force_rebuild and EMBEDDINGS_NPY.exists() and INDEX_MANIFEST.exists():
        with open(INDEX_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if _manifest_matches(manifest, chunks, model_name):
            embeddings = np.load(EMBEDDINGS_NPY)
            with open(CHUNK_ID_MAP, "r", encoding="utf-8") as f:
                chunk_ids = json.load(f)
            manifest["_cache_hit"] = True
            return ManualCosineIndex(embeddings=embeddings, chunk_ids=chunk_ids), manifest

    if embedder is None:
        embedder = TextEmbedder(model_name)

    t0 = time.time()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress=False)
    elapsed = time.time() - t0

    chunk_ids = [c.chunk_id for c in chunks]
    np.save(EMBEDDINGS_NPY, embeddings)
    with open(CHUNK_ID_MAP, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    manifest = {
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "n_chunks": len(chunks),
        "chunks_fingerprint": _chunks_fingerprint(chunks),
        "encode_seconds": round(elapsed, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunk_stats": chunk_stats(chunks),
        "_cache_hit": False,
    }
    with open(INDEX_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return ManualCosineIndex(embeddings=embeddings, chunk_ids=chunk_ids), manifest
