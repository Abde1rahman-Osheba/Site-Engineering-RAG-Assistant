"""
Part E (stretch) — Cross-encoder re-ranking.

Retrieve a larger candidate pool (RERANK_CANDIDATE_POOL, e.g. 10) with the
cheap bi-encoder (all-MiniLM-L6-v2 cosine search), then re-score each
(query, candidate) pair jointly with a cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2), which is slower per-pair but far
more accurate because it attends over the query and passage together
instead of comparing two independently-computed vectors. The top
DEFAULT_TOP_K after re-ranking is what actually goes into the prompt.

A pure keyword-overlap re-ranker is also provided as a zero-dependency
fallback/comparison signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .chunking import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def keyword_overlap_rerank(query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """Cheap, dependency-free re-ranking signal: fraction of query tokens
    that literally appear in the candidate. Useful as a sanity baseline
    against the cross-encoder, and as a fallback if the cross-encoder model
    can't be downloaded."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return candidates
    rescored = []
    for chunk, orig_score in candidates:
        c_tokens = _tokenize(chunk.text)
        overlap = len(q_tokens & c_tokens) / len(q_tokens)
        rescored.append((chunk, overlap))
    return sorted(rescored, key=lambda cs: cs[1], reverse=True)


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = config.DEFAULT_TOP_K) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c, _ in candidates]
        ce_scores = self.model.predict(pairs)
        rescored = list(zip([c for c, _ in candidates], (float(s) for s in ce_scores)))
        rescored.sort(key=lambda cs: cs[1], reverse=True)
        return rescored[:top_k]
