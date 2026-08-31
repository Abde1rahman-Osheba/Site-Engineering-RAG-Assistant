"""
Local web chat interface for the 1830 PSS Multimodal RAG assistant.

Runs entirely on your own machine: Flask serves a single-page chat UI
(`web/index.html`) and a small JSON API that calls the exact same
`RAGPipeline` used by the notebook - same retrieval, same grounded prompt,
same local GGUF/llama.cpp generation backend (see README.md / src/llm_client.py).
No data leaves your machine; no Anthropic/OpenAI account involved.

Usage:
    python app.py
    # then open http://127.0.0.1:5000 in your browser

Env vars (optional):
    RAG_USE_FIXTURE=1     build the pipeline against the safe, fictitious
                          "Acme OptiRack" synthetic fixture instead of the
                          real document - useful for a quick, fully-safe demo
                          without needing the real PDF at all.
    RAG_FORCE_OFFLINE=1   use the fast, non-LLM offline heuristic instead of
                          the real local model (see llm_client.py) - useful
                          for checking the UI/plumbing without waiting on
                          real generation.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: E402  (loads .env - HF_TOKEN, etc.)
from src import config  # noqa: E402
from src.extraction import extract_page_range, load_extracted, save_extracted, extraction_is_fresh  # noqa: E402
from src.chunking import build_chunks  # noqa: E402
from src.embedding_index import TextEmbedder, build_or_load_index  # noqa: E402
from src.bm25_hybrid import build_bm25_index  # noqa: E402
from src.rerank import CrossEncoderReranker  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.llm_client import pick_available_provider  # noqa: E402

from flask import Flask, jsonify, request

app = Flask(__name__, static_folder="web", static_url_path="")

_pipeline: RAGPipeline | None = None
_provider: str | None = None
_model_label: str = ""


def _build_real_pipeline() -> RAGPipeline:
    print(f"[1/4] Extracting pages {config.PAGE_START}-{config.PAGE_END} from {config.SOURCE_PDF.name} ...")
    if extraction_is_fresh():
        pages = load_extracted()
        print("      (loaded cached extraction)")
    else:
        pages = extract_page_range(config.SOURCE_PDF, config.PAGE_START, config.PAGE_END)
        save_extracted(pages)

    print("[2/4] Chunking ...")
    chunks = build_chunks(pages)
    print(f"      {len(chunks)} chunks")

    print("[3/4] Embedding + building indexes (manual cosine + BM25) ...")
    embedder = TextEmbedder()
    text_index, manifest = build_or_load_index(chunks, embedder=embedder)
    bm25_index = build_bm25_index(chunks)
    print(f"      index ready (cache_hit={manifest['_cache_hit']})")

    print("[4/4] Loading cross-encoder re-ranker ...")
    try:
        cross_encoder = CrossEncoderReranker()
    except Exception as e:  # pragma: no cover - optional component
        print(f"      cross-encoder unavailable ({e}); falling back to keyword-overlap re-ranking")
        cross_encoder = None

    return RAGPipeline(
        chunks=chunks,
        text_index=text_index,
        embedder=embedder,
        bm25_index=bm25_index,
        cross_encoder=cross_encoder,
    )


def _build_fixture_pipeline() -> RAGPipeline:
    """Safe demo mode: builds the pipeline against a small, entirely
    fictitious "Acme OptiRack" document instead of the real one - useful
    for trying the chat UI with no real document required at all."""
    from tests.synthetic_fixture import build_fixture_pdf, FIXTURE_PDF
    from src import chunking as chunking_mod
    import pymupdf as fitz

    print("[demo mode] Building the safe synthetic 'Acme OptiRack' fixture ...")
    chunking_mod._SHELF_TAG_RES = [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"Acme\s*OptiRack[\s-]*8(?!FAN)",
            r"Acme\s*OptiRack[\s-]*32",
            r"Acme\s*OptiRack[\s-]*16\s*II",
        )
    ]
    build_fixture_pdf()
    doc = fitz.open(FIXTURE_PDF)
    n_pages = doc.page_count
    doc.close()
    pages = extract_page_range(FIXTURE_PDF, 1, n_pages)
    chunks = build_chunks(pages)
    embedder = TextEmbedder()
    emb = embedder.encode([c.text for c in chunks])
    from src.embedding_index import ManualCosineIndex

    text_index = ManualCosineIndex(embeddings=emb, chunk_ids=[c.chunk_id for c in chunks])
    bm25_index = build_bm25_index(chunks)
    print(f"      {len(chunks)} chunks (fixture)")
    return RAGPipeline(chunks=chunks, text_index=text_index, embedder=embedder, bm25_index=bm25_index)


def get_pipeline() -> RAGPipeline:
    global _pipeline, _provider, _model_label
    if _pipeline is None:
        use_fixture = bool(os.environ.get("RAG_USE_FIXTURE"))
        if use_fixture:
            _pipeline = _build_fixture_pipeline()
        else:
            if not config.SOURCE_PDF.exists():
                raise FileNotFoundError(
                    f"{config.SOURCE_PDF.name} not found at the project root. Place your own copy there "
                    f"(see README.md), or set RAG_USE_FIXTURE=1 to try the UI against the safe synthetic "
                    f"fixture instead."
                )
            _pipeline = _build_real_pipeline()
        _provider = pick_available_provider()
        _model_label = (
            f"{config.GGUF_REPO} ({config.GGUF_FILENAME})" if _provider == "huggingface" else _provider
        )
        print(f"\nReady. Provider: {_provider}  Model: {_model_label}")
    return _pipeline


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    ready = _pipeline is not None
    return jsonify({
        "ready": ready,
        "provider": _provider,
        "model": _model_label,
        "fixture_mode": bool(os.environ.get("RAG_USE_FIXTURE")),
    })


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Message is empty."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Message is too long (max 1000 characters)."}), 400

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    t0 = time.time()
    try:
        result = pipeline.answer(question, use_hybrid=True, use_rerank=True, provider=_provider)
    except Exception as e:  # pragma: no cover - surfaced to the UI as a chat error bubble
        return jsonify({"error": f"Generation failed: {e}"}), 500
    latency = time.time() - t0

    sources = [
        {
            "heading": c.heading_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "score": round(float(score), 3),
        }
        for c, score in result.retrieval.chunks_with_scores
    ]

    return jsonify({
        "answer": result.answer,
        "refused": result.refused,
        "cited": result.has_citation,
        "provider": result.provider,
        "model": result.model,
        "sources": sources,
        "latency_s": round(latency, 1),
    })


if __name__ == "__main__":
    print("=" * 70)
    print("1830 PSS Assistant - local chat UI")
    print("=" * 70)
    get_pipeline()  # build/load everything up front, before serving requests
    print("\nOpen http://127.0.0.1:5000 in your browser.\n")
    # threaded=False: a single llama.cpp model instance is not safe to call
    # concurrently from multiple request threads - this app serves one
    # question at a time, which is the right behaviour for a local single-user chat.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
