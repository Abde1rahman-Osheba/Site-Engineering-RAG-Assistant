"""
Ties every part of the assignment together into one callable pipeline:

  question
    -> embed (TextEmbedder, same model as indexing)
    -> retrieve top-k (ManualCosineIndex, optionally FAISS instead - Part B)
    -> optional metadata filter by shelf tag (Part E)
    -> optional hybrid fusion with BM25 (Part E)
    -> optional cross-encoder re-ranking over a larger candidate pool (Part E)
    -> build grounded/guardrailed prompt (Part C)
    -> call an LLM for generation (Part C, step 7)
    -> package retrieved evidence + answer + validation flags (Part D input)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .chunking import Chunk
from .embedding_index import TextEmbedder, ManualCosineIndex, FaissIndex
from .bm25_hybrid import BM25Index, hybrid_scores, top_k_from_scores
from .rerank import CrossEncoderReranker, keyword_overlap_rerank
from .prompting import build_prompt, looks_like_refusal, has_citation, expand_citations
from .llm_client import generate_answer, pick_available_provider, corpus_common_tokens, GenerationResult


@dataclass
class RetrievalResult:
    chunks_with_scores: list[tuple[Chunk, float]]
    backend: str
    used_hybrid: bool
    used_rerank: bool
    shelf_filter: list[str] | None


@dataclass
class AnswerResult:
    question: str
    retrieval: RetrievalResult
    answer: str
    provider: str
    model: str
    refused: bool
    has_citation: bool


class RAGPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        text_index: ManualCosineIndex,
        embedder: TextEmbedder,
        faiss_index: FaissIndex | None = None,
        bm25_index: BM25Index | None = None,
        cross_encoder: CrossEncoderReranker | None = None,
    ):
        self.chunks = chunks
        self.chunk_by_id = {c.chunk_id: c for c in chunks}
        self.text_index = text_index
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.cross_encoder = cross_encoder
        self._common_tokens = corpus_common_tokens(chunks)  # used only by the offline fallback provider

    # ------------------------------------------------------------------
    # Metadata filtering (Part E)
    # ------------------------------------------------------------------
    def _allowed_ids_for_shelf(self, shelf_filter: list[str]) -> set[str]:
        norm_targets = {s.upper().replace(" ", "") for s in shelf_filter}
        allowed = set()
        for c in self.chunks:
            tags_norm = {t.replace(" ", "") for t in c.shelf_tags}
            if any(any(target in tag or tag in target for tag in tags_norm) for target in norm_targets):
                allowed.add(c.chunk_id)
        return allowed

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
        backend: str = "manual",           # "manual" | "faiss"
        use_hybrid: bool = False,
        use_rerank: bool = False,
        shelf_filter: list[str] | None = None,
    ) -> RetrievalResult:
        query_vec = self.embedder.encode_one(question)
        allowed_ids = self._allowed_ids_for_shelf(shelf_filter) if shelf_filter else None

        pool_k = config.RERANK_CANDIDATE_POOL if use_rerank else top_k

        if use_hybrid:
            # Hybrid needs a full-corpus embedding score map, then fuses with BM25.
            if allowed_ids is not None:
                sub_ids = [cid for cid in self.text_index.chunk_ids if cid in allowed_ids]
            else:
                sub_ids = self.text_index.chunk_ids
            full_hits = self.text_index.search(query_vec, top_k=len(self.text_index.chunk_ids))
            emb_score_map = {cid: score for cid, score in full_hits if cid in set(sub_ids)}
            fused = hybrid_scores(emb_score_map, self.bm25_index, question)
            fused = {cid: s for cid, s in fused.items() if cid in set(sub_ids)}
            hits = top_k_from_scores(fused, pool_k)
        elif backend == "faiss":
            if allowed_ids is not None:
                hits = self.faiss_index.search(query_vec, top_k=len(self.text_index.chunk_ids))
                hits = [(cid, s) for cid, s in hits if cid in allowed_ids][:pool_k]
            else:
                hits = self.faiss_index.search(query_vec, top_k=pool_k)
        else:
            if allowed_ids is not None:
                hits = self.text_index.search_filtered(query_vec, allowed_ids, top_k=pool_k)
            else:
                hits = self.text_index.search(query_vec, top_k=pool_k)

        candidates = [(self.chunk_by_id[cid], score) for cid, score in hits]

        if use_rerank:
            if self.cross_encoder is not None:
                candidates = self.cross_encoder.rerank(question, candidates, top_k=top_k)
            else:
                candidates = keyword_overlap_rerank(question, candidates)[:top_k]
        else:
            candidates = candidates[:top_k]

        return RetrievalResult(
            chunks_with_scores=candidates,
            backend=backend,
            used_hybrid=use_hybrid,
            used_rerank=use_rerank,
            shelf_filter=shelf_filter,
        )

    # ------------------------------------------------------------------
    # End-to-end answer
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
        backend: str = "manual",
        use_hybrid: bool = False,
        use_rerank: bool = False,
        shelf_filter: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AnswerResult:
        retrieval = self.retrieve(
            question,
            top_k=top_k,
            backend=backend,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            shelf_filter=shelf_filter,
        )
        prompt = build_prompt(question, retrieval.chunks_with_scores)
        provider = provider or pick_available_provider()
        gen: GenerationResult = generate_answer(
            question, prompt, retrieval.chunks_with_scores, provider=provider, model=model,
            common_tokens=self._common_tokens,
        )
        # The LLM only emits short "[Sn]" tags (rule 2 in the system prompt);
        # expand them into full "(Section: ..., p.N)" citations here, deterministically,
        # from the actual retrieved-chunk metadata - see prompting.expand_citations().
        answer_text = expand_citations(gen.answer, retrieval.chunks_with_scores)
        return AnswerResult(
            question=question,
            retrieval=retrieval,
            answer=answer_text,
            provider=gen.provider,
            model=gen.model,
            refused=looks_like_refusal(answer_text),
            has_citation=has_citation(answer_text),
        )
