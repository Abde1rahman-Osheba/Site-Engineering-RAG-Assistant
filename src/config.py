"""
Central configuration for the 1830 PSS Multimodal RAG assistant.

All paths, page ranges, model names and tunable parameters live here so the
rest of the codebase (and the notebook) never hard-codes a magic number.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_PDF = PROJECT_ROOT / "1830_Technical_Description.pdf"

DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"          # raw per-page text + markdown dump
PROCESSED_DIR = DATA_DIR / "processed"          # chunked, metadata-tagged passages
INDEX_DIR = DATA_DIR / "index"                  # persisted embeddings / FAISS / BM25 / CLIP

for _d in (EXTRACTED_DIR, PROCESSED_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EXTRACTED_MARKDOWN = EXTRACTED_DIR / "chapters_1_2_pages_47_166.md"
EXTRACTED_PAGES_JSON = EXTRACTED_DIR / "pages_raw.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"

# ---------------------------------------------------------------------------
# Scope: assignment restricts the pipeline to Chapters 1-2 only
# (System concept  +  Shelves/common equipment through Power filters).
# Page numbers are 1-indexed *physical* page numbers in the PDF file itself
# (the same convention `pdftotext -f -l` and PyPDF2/PyMuPDF `page index + 1` use).
# ---------------------------------------------------------------------------
CHAPTER_RANGES = {
    1: {"title": "System concept", "first_page": 47, "last_page": 72},
    2: {"title": "Shelves and common equipment/cards (through Power filters)", "first_page": 73, "last_page": 166},
}
PAGE_START = min(c["first_page"] for c in CHAPTER_RANGES.values())   # 47
PAGE_END = max(c["last_page"] for c in CHAPTER_RANGES.values())      # 166

# ---------------------------------------------------------------------------
# Chunking targets (Part A)
# ---------------------------------------------------------------------------
CHUNK_TARGET_MIN_WORDS = 100
CHUNK_TARGET_MAX_WORDS = 300
CHUNK_HARD_MAX_WORDS = 340          # allow a little slack before forcing a split
CHUNK_OVERLAP_WORDS = 40            # sliding-window overlap when a section must be split
MIN_STANDALONE_SECTION_WORDS = 100  # sections shorter than this get merged with a same-parent
                                     # neighbour where safe (see chunking._merge_tiny_sections);
                                     # tuned against the real document - see README "Chunking strategy"

# ---------------------------------------------------------------------------
# Embedding / retrieval models
# ---------------------------------------------------------------------------
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # Session 4 model
CLIP_MODEL = "clip-ViT-B-32"                                  # multimodal (image+text) embeddings
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Part E: re-ranking

DEFAULT_TOP_K = 5
RERANK_CANDIDATE_POOL = 10   # Part E: retrieve this many before re-ranking down to DEFAULT_TOP_K

# hybrid search fusion weight: final = ALPHA * embedding_score + (1-ALPHA) * bm25_score
# (both min-max normalised to [0, 1] first)
HYBRID_ALPHA = 0.55

# ---------------------------------------------------------------------------
# Known shelf identifiers used for metadata tagging + metadata-filtered retrieval
# (Part E). Detected automatically from chunk text via regex - not hand-typed
# from the document, just the family of names the 1830 PSS product line uses.
# ---------------------------------------------------------------------------
KNOWN_SHELF_PATTERNS = [
    r"1830\s*PSS[\s-]*4",
    r"1830\s*PSS[\s-]*8",
    r"1830\s*PSS[\s-]*16\s*II",
    r"1830\s*PSS[\s-]*16",
    r"1830\s*PSS[\s-]*32",
    r"1830\s*PSS[\s-]*36",
]

# ---------------------------------------------------------------------------
# LLM generation - local, self-hosted only (no Anthropic/OpenAI hosted APIs).
# Everything below runs entirely on your own machine/GPU, no API key or
# network call needed at query time beyond the one-time weight download.
#
# Default backend: GGUF quantization via `llama-cpp-python` (the `llama_cpp`
# package), loading straight from the Hugging Face Hub. Chosen over
# transformers+bitsandbytes as the default because bitsandbytes' 4-bit
# format only runs its quantized layers on a CUDA GPU - splitting it across
# a small GPU and CPU needs an explicit escape hatch
# (`llm_int8_enable_fp32_cpu_offload`) that keeps the CPU-resident portion
# in **fp32**, ballooning its RAM footprint ~8x versus its 4-bit size - not
# safe on a small-VRAM/tight-RAM laptop (confirmed by hitting exactly this
# wall in practice on a 4GB-VRAM/16GB-RAM machine). GGUF's CPU-offloaded
# layers stay in the *same* quantized format (no ballooning), and
# `n_gpu_layers` lets you tune exactly how many layers live on GPU vs CPU -
# the standard, well-supported approach for small-VRAM consumer GPUs, and
# it still works (slower) with `n_gpu_layers=0`, i.e. no GPU at all.
# ---------------------------------------------------------------------------
# See requirements.txt for the llama-cpp-python install command (a specific
# version pin matters on Windows + certain Intel CPUs - see that file).
GGUF_REPO = "unsloth/Llama-3.1-8B-Instruct-GGUF"   # same model family as the bnb-4bit repo below, GGUF-quantized
GGUF_FILENAME = "*Q4_K_M.gguf"                     # ~4.9GB - the standard quality/size "sweet spot" quant
GGUF_N_CTX = 4096
# How many transformer layers to place on the GPU. -1 = try to offload all
# (fastest if it fits); if that overflows VRAM, `_load_gguf_model()` retries
# with progressively fewer GPU layers, down to 0 (CPU-only, always works -
# no CUDA required for GGUF/llama.cpp, unlike bitsandbytes).
GGUF_N_GPU_LAYERS_CANDIDATES = [-1, 28, 20, 12, 6, 0]

HF_MAX_NEW_TOKENS = 512
HF_DO_SAMPLE = False  # greedy decoding - deterministic, precision-favouring answers for spec lookups

# ---------------------------------------------------------------------------
# Alternate backend: transformers + bitsandbytes 4-bit (select explicitly via
# provider="transformers"). Kept for machines with enough VRAM to hold the
# whole 4-bit model on GPU without CPU offload (roughly >=6GB) - not the
# default on typical small-GPU laptops, per the GGUF rationale above.
# ---------------------------------------------------------------------------
HF_MODEL_DEFAULT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
HF_DEVICE_MAP = "auto"

# 4-bit quantization (via bitsandbytes) when a CUDA GPU is available. An 8B
# model needs ~16GB in fp16 - far more than a typical single consumer GPU
# (e.g. 4-8GB laptop cards); 4-bit gets weights down to ~5GB. Ignored
# entirely on CPU-only machines (no CUDA -> no bitsandbytes path).
HF_QUANTIZE_4BIT = True

REFUSAL_STRING = "Not found in the provided document."

RANDOM_SEED = 42
