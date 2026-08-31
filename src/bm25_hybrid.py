"""
Part E (stretch) — Hybrid search.

Combines the dense embedding similarity score with a sparse, lexical
BM25 score. Pure embedding search sometimes under-ranks passages that share
an *exact* rare token with the query (a part number, an acronym like "8FAN",
a precise unit like "RU") because all-MiniLM-L6-v2 was trained for general
semantic similarity, not exact-token matching. BM25 is the classic fix, and
combining the two ("hybrid") is the standard mitigation - see README.md for
a worked example question where hybrid outperforms pure embedding search.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk

BM25_PICKLE = config.INDEX_DIR / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def save(self, path=BM25_PICKLE):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path=BM25_PICKLE) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(bm25=d["bm25"], chunk_ids=d["chunk_ids"])


def build_bm25_index(chunks: list[Chunk]) -> BM25Index:
    corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    return BM25Index(bm25=bm25, chunk_ids=[c.chunk_id for c in chunks])


def _min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_scores(
    embedding_scores: dict[str, float],
    bm25_index: BM25Index,
    query: str,
    alpha: float = config.HYBRID_ALPHA,
) -> dict[str, float]:
    """Fuse dense + sparse scores over the full corpus (min-max normalised
    independently, then weighted-summed) and return a {chunk_id: score} map.

    `embedding_scores` should already cover every chunk id known to
    `bm25_index` (i.e. call this over the *full* corpus, not just a
    pre-filtered top-k, so the fusion is meaningful).
    """
    all_ids = bm25_index.chunk_ids
    bm25_raw = bm25_index.scores(query)
    emb_raw = np.array([embedding_scores.get(cid, 0.0) for cid in all_ids], dtype=np.float32)

    bm25_norm = _min_max(bm25_raw)
    emb_norm = _min_max(emb_raw)
    fused = alpha * emb_norm + (1 - alpha) * bm25_norm
    return {cid: float(s) for cid, s in zip(all_ids, fused)}


def top_k_from_scores(score_map: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    return sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
