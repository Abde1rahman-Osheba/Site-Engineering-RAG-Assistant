"""
Part C, step 7 — generation.

"you may call any LLM you have access to for the generation step; the
retrieval and grounding logic must be your own code" - this project's choice
is a **locally-run, open-weight model** (Llama-3.1-8B-Instruct), not a
hosted API - no Anthropic, no OpenAI. This module is the *only* place that
loads/calls that model. Everything upstream (chunking, embedding, similarity
search, hybrid fusion, re-ranking, prompt construction, refusal/citation
checking) is our own code.

Three backends, chosen by `provider`:
  - "huggingface" (default): the real generation path - a locally-run,
    open-weight model loaded straight from the Hugging Face Hub via
    `llama-cpp-python` (GGUF quantization: default
    `unsloth/Llama-3.1-8B-Instruct-GGUF`, Q4_K_M, ~4.9GB). Chosen as the
    default over the "transformers" backend below because GGUF's
    `n_gpu_layers` lets a small consumer GPU offload only as many layers as
    fit in VRAM while the rest stay quantized (not fp32-ballooned) on CPU -
    see `config.py`'s comment for the full rationale. No API key, no
    hosted-provider account, no network call at query time beyond the
    one-time download - runs entirely on your own machine/GPU (or CPU only,
    slower, via `n_gpu_layers=0`). This is what `pick_available_provider()`
    returns by default.
  - "transformers": an alternate real-generation path via the `transformers`
    library (default: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, a
    pre-quantized bitsandbytes 4-bit re-upload), for machines with enough
    VRAM to hold the whole 4-bit model on GPU without CPU offload. Select
    it explicitly with `provider="transformers"`.
  - "offline":      a zero-dependency, non-LLM extractive fallback used only
    for fast mechanical self-tests (see tests/run_self_test.py) so the
    retrieval/prompt/citation/refusal plumbing can be checked without
    loading an 8B-parameter model every time. Clearly labelled in its
    output; not meant for real answers.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config
from .prompting import PromptBundle


@dataclass
class GenerationResult:
    answer: str
    provider: str
    model: str
    raw: object | None = None


# Loaded lazily and cached by (repo, filename), since an 8B-parameter model
# can take tens of seconds to a few minutes to load onto a GPU/CPU - we do
# not want to reload it on every single question (e.g. across the Part D
# evaluation loop).
_GGUF_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _preload_windows_llama_cpp_dlls() -> None:
    """`llama-cpp-python`'s CUDA-enabled Windows wheel has DLL dependencies a
    stock Windows install doesn't ship: the OpenMP runtime (`vcomp140.dll` -
    part of the VC++ redistributable) and the CUDA runtime/cuBLAS libraries
    (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` - normally
    installed via the full CUDA Toolkit, which this project doesn't
    otherwise require). Rather than asking every user to install the CUDA
    Toolkit or track down the VC++ redistributable separately, this preloads
    those exact DLLs from copies this project's own *other* dependencies
    already bundle on Windows (`faiss-cpu`/`scikit-learn` ship `vcomp140.dll`
    for their own OpenMP use; the CUDA-enabled `torch` wheel ships the CUDA
    runtime/cuBLAS libraries) - so `import llama_cpp` resolves its
    transitive DLL dependencies against these already-loaded copies instead
    of failing to find them on PATH/System32. Once a DLL is loaded into the
    process from an explicit path, Windows resolves later unqualified
    lookups for that DLL name against the already-loaded copy - that's the
    mechanism this relies on. A no-op on non-Windows platforms; harmless
    (each piece is individually try/except-guarded) if a piece isn't found -
    `import llama_cpp` then raises its own clear error instead.
    """
    if sys.platform != "win32":
        return

    import ctypes

    def _try_load(path: Path) -> bool:
        try:
            if path.exists():
                ctypes.CDLL(str(path))
                return True
        except OSError:
            pass
        return False

    # vcomp140.dll (OpenMP) - bundled by faiss-cpu and/or scikit-learn (both
    # already project dependencies - see requirements.txt) in one of two
    # common on-disk layouts.
    numpy_spec = importlib.util.find_spec("numpy")
    if numpy_spec and numpy_spec.origin:
        site_packages = Path(numpy_spec.origin).resolve().parent.parent
        for candidate in (*site_packages.glob("*.libs/vcomp140.dll"), *site_packages.glob("*/.libs/vcomp140.dll")):
            if _try_load(candidate):
                break

    # CUDA runtime/cuBLAS - bundled by the CUDA-enabled `torch` wheel.
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        for dll_name in ("cublasLt64_12.dll", "cublas64_12.dll", "cudart64_12.dll"):
            _try_load(torch_lib / dll_name)
    except ImportError:
        pass


def _load_gguf_model(repo_id: str, filename: str):
    """Load a GGUF model straight from the Hugging Face Hub via
    `llama-cpp-python`. Tries `config.GGUF_N_GPU_LAYERS_CANDIDATES` in order
    (most GPU layers first) and falls back to fewer/zero GPU layers if a
    candidate doesn't fit in VRAM - `n_gpu_layers=0` (pure CPU) always works,
    since GGUF/llama.cpp needs no CUDA at all, unlike the bitsandbytes
    "transformers" backend.
    """
    cache_key = (repo_id, filename)
    if cache_key not in _GGUF_MODEL_CACHE:
        _preload_windows_llama_cpp_dlls()
        from llama_cpp import Llama

        last_error: Exception | None = None
        for n_gpu_layers in config.GGUF_N_GPU_LAYERS_CANDIDATES:
            try:
                llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=config.GGUF_N_CTX,
                    verbose=False,
                )
                _GGUF_MODEL_CACHE[cache_key] = (llm, n_gpu_layers)
                break
            except Exception as e:  # noqa: BLE001 - genuinely want to catch/retry any load failure (OOM, etc.)
                last_error = e
        else:
            raise RuntimeError(
                f"Could not load GGUF model {repo_id}/{filename} at any GPU-layer count "
                f"(tried {config.GGUF_N_GPU_LAYERS_CANDIDATES}); last error: {last_error}"
            )
    return _GGUF_MODEL_CACHE[cache_key]


def _llama_cpp_generate(
    prompt: PromptBundle,
    question: str,
    repo_id: str,
    filename: str,
    max_new_tokens: int = config.HF_MAX_NEW_TOKENS,
    do_sample: bool = config.HF_DO_SAMPLE,
) -> GenerationResult:
    """Local, self-hosted generation via `llama-cpp-python` - no API key, no
    network call at query time beyond the one-time GGUF download. Uses the
    model's own chat template (llama.cpp applies it internally from the
    GGUF's embedded chat_template metadata) via `create_chat_completion`, so
    the system prompt's grounding/citation/refusal rules land exactly where
    this model expects them.

    Known, deliberately-accepted limitation: despite the system prompt's
    explicit rule + worked example (see prompting.py) telling the model not
    to, it sometimes tacks the fixed refusal sentence onto an answer that
    already fully covers the question - a redundant trailing caveat, not a
    wrong or missing fact. A code-level fix was tried (a cheap follow-up
    yes/no self-check asking the model whether every part was already
    answered) and reverted: on a partial-support question it incorrectly
    answered "yes" and stripped a *legitimate* refusal, silently hiding that
    one part genuinely wasn't found - a worse failure mode than the
    cosmetic extra sentence it was meant to remove. An occasional harmless
    "answered anyway" caveat is safer than ever risking a false impression
    of completeness, so this is left as a known 8B-model calibration quirk
    rather than "fixed" at the cost of that risk. `question` is accepted
    here (unused directly) to keep the call signature stable for that
    now-reverted check, should a safer version of it be revisited later.
    """
    llm, n_gpu_layers_used = _load_gguf_model(repo_id, filename)

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        max_tokens=max_new_tokens,
        temperature=0.0 if not do_sample else 0.7,
    )
    text = response["choices"][0]["message"]["content"].strip()
    return GenerationResult(
        answer=text,
        provider="huggingface",
        model=f"{repo_id}/{filename} (n_gpu_layers={n_gpu_layers_used})",
    )


# Loaded lazily and cached by model name, since an 8B-parameter model can take
# tens of seconds to a few minutes to load onto a GPU/CPU - we do not want to
# reload it on every single question (e.g. across the Part D evaluation loop).
_HF_MODEL_CACHE: dict[str, tuple] = {}


def _load_hf_model(model_name: str):
    if model_name not in _HF_MODEL_CACHE:
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

        # Explicit token pickup (in addition to huggingface_hub's own implicit
        # HF_TOKEN/cached-login detection) so a token placed in `.env` (see
        # `src/__init__.py`, which loads it) or exported in the shell is used
        # for gated models like the Llama-3.1 default - never hardcode a
        # token as a string literal here or anywhere else in the codebase.
        hf_token = os.environ.get("HF_TOKEN") or None

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # Some repos (e.g. pre-quantized community re-uploads, like the
        # bnb-4bit default below) already ship their own quantization_config
        # in config.json - applying a second, our-own BitsAndBytesConfig on
        # top of an already-quantized model errors out, so detect and skip
        # our injection in that case.
        hf_config = AutoConfig.from_pretrained(model_name, token=hf_token)
        already_quantized = getattr(hf_config, "quantization_config", None) is not None

        load_kwargs = {"device_map": config.HF_DEVICE_MAP, "token": hf_token}
        if not already_quantized:
            if torch.cuda.is_available() and config.HF_QUANTIZE_4BIT:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    # bitsandbytes not installed - fall back to plain fp16 on
                    # GPU (may not fit on a small card, but at least tries GPU first)
                    load_kwargs["torch_dtype"] = torch.float16
            elif torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        _HF_MODEL_CACHE[model_name] = (tokenizer, model)
    return _HF_MODEL_CACHE[model_name]


def _huggingface_generate(
    prompt: PromptBundle,
    model_name: str,
    max_new_tokens: int = config.HF_MAX_NEW_TOKENS,
    do_sample: bool = config.HF_DO_SAMPLE,
) -> GenerationResult:
    """Local, self-hosted generation via `transformers` - no API key, no
    network call at query time. Requires the model weights to be downloaded
    once (and, for a gated model like Llama-3.1-8B-Instruct, accepting
    Meta's license on Hugging Face and authenticating - `huggingface-cli
    login` or an `HF_TOKEN` env var - before the first call).

    Uses the model's own chat template (`apply_chat_template`) so the system
    prompt's grounding/citation/refusal rules are placed exactly where that
    model expects them, rather than hand-formatting a prompt string.
    """
    tokenizer, model = _load_hf_model(model_name)

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode only the newly-generated continuation, not the echoed prompt
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return GenerationResult(answer=text.strip(), provider="huggingface", model=model_name)


_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Generic/stopword-ish tokens that don't count as "evidence" for the offline
# baseline: a chunk about the right shelf will always share these with the
# question (that's what got it retrieved), so requiring overlap on them
# would let the baseline "answer" questions about facts it never actually
# found - exactly the false-confidence failure mode Part D Q8 is designed
# to catch. Only overlap on words *outside* this set counts as real evidence.
_GENERIC_TOKENS = {
    "what", "is", "the", "of", "and", "or", "a", "an", "to", "for", "in", "on",
    "does", "do", "which", "who", "how", "many", "much", "are", "its", "it",
    "shelf", "shelves", "unit", "units", "system", "card", "cards", "provide",
    "provides", "supported", "supports", "used", "use", "name", "required",
    "required", "with", "without", "at",
}


def _informative_tokens(text: str, extra_stop: set[str] = frozenset()) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _GENERIC_TOKENS - extra_stop


def corpus_common_tokens(chunks, top_n: int = 20) -> set[str]:
    """Words that recur across many chunks (e.g. "optical", "transport",
    "system" in a transport-hardware manual) carry little discriminating
    power for the offline extractive baseline - a chunk merely *sharing
    the manual's vocabulary* with the question isn't evidence it answers
    it. Computed once per corpus and passed into the offline generator so
    it doesn't count these as overlap "evidence" (mirrors an IDF-style
    downweighting without the extra dependency)."""
    from collections import Counter

    counts: Counter[str] = Counter()
    n_chunks = max(len(chunks), 1)
    for c in chunks:
        counts.update(set(_informative_tokens(c.body)))
    # A token appearing in a large share of chunks is "part of the manual's
    # background vocabulary" (e.g. "optical", "transport", "system" recur on
    # nearly every page of a transport-hardware manual), not evidence for any
    # one specific question. The higher floor (vs. a plain 15% cut) keeps
    # this from over-firing on small corpora, where a handful of chunks
    # legitimately sharing one topic word (e.g. several different fan-unit
    # sections all saying "fan") would otherwise get wrongly stripped too.
    threshold = max(5, int(0.30 * n_chunks))
    common = {tok for tok, n in counts.items() if n >= threshold}
    if len(common) < top_n:
        common |= {tok for tok, n in counts.most_common(top_n) if n >= threshold}
    return common


def _offline_extractive_generate(question: str, chunks_with_scores, common_tokens: set[str] = frozenset()) -> GenerationResult:
    """Deterministic, non-LLM baseline: scans every retrieved chunk (not
    just the top-scored one - the answer is often in the #2/#3 hit) for the
    sentence with the highest *informative* lexical overlap with the
    question, or refuses if no chunk has any. Useful for testing the
    pipeline mechanics without any API key/network call.

    Deliberately conservative: a chunk being topically about the right
    shelf/component is NOT enough to answer - it must also share an
    informative (non-generic, non-heading) token with the question, e.g.
    the actual unit/quantity being asked about. This is what lets it
    correctly refuse on a trick question like "max optical reach in km"
    even when a top-retrieved chunk is topically about the right shelf.
    """
    if not chunks_with_scores or max(s for _, s in chunks_with_scores) < 0.15:
        return GenerationResult(
            answer=f"[offline extractive mode] {config.REFUSAL_STRING}",
            provider="offline",
            model="extractive-baseline",
        )

    # NOTE: earlier we also stripped each candidate's own heading tokens
    # from the question before matching. That over-corrects: when the
    # heading itself names the exact fact being asked about (e.g. a
    # "<Shelf> fan units" section answering "which fan units does <Shelf>
    # use"), the heading match *is* the evidence, so it must not be
    # discarded - only the corpus-wide common-vocabulary filter applies.
    q_tokens = _informative_tokens(question, extra_stop=common_tokens)
    if not q_tokens:
        return GenerationResult(
            answer=f"[offline extractive mode] {config.REFUSAL_STRING}",
            provider="offline",
            model="extractive-baseline",
        )

    best_chunk, best_sentence, best_overlap = None, "", -1
    for chunk, _score in chunks_with_scores:
        for s in re.split(r"(?<=[.!?])\s+", chunk.body):
            overlap_tokens = q_tokens & _informative_tokens(s)
            # A bare shared digit (e.g. "8") is weak/ambiguous evidence in a
            # manual with many similarly-numbered products/variants ("-8",
            # "-16", "-32", ...) - it only counts alongside at least one
            # overlapping non-numeric word, never on its own.
            if overlap_tokens and not any(not t.isdigit() for t in overlap_tokens):
                continue
            overlap = len(overlap_tokens)
            if overlap > best_overlap:
                best_chunk, best_sentence, best_overlap = chunk, s, overlap

    if best_overlap <= 0 or best_chunk is None:
        return GenerationResult(
            answer=f"[offline extractive mode] {config.REFUSAL_STRING}",
            provider="offline",
            model="extractive-baseline",
        )

    citation = f"(Section: {best_chunk.heading_path}, p.{best_chunk.page_start})"
    answer = f"[offline extractive mode] {best_sentence.strip()} {citation}"
    return GenerationResult(answer=answer, provider="offline", model="extractive-baseline")


def generate_answer(
    question: str,
    prompt: PromptBundle,
    chunks_with_scores,
    provider: str = "huggingface",
    model: str | None = None,
    common_tokens: set[str] = frozenset(),
) -> GenerationResult:
    if provider in ("huggingface", "hf", "local", "llama_cpp", "gguf"):
        repo_id = model or os.environ.get("LOCAL_LLM_MODEL") or config.GGUF_REPO
        filename = os.environ.get("LOCAL_LLM_GGUF_FILE") or config.GGUF_FILENAME
        return _llama_cpp_generate(prompt, question, repo_id, filename)
    if provider == "transformers":
        return _huggingface_generate(prompt, model or os.environ.get("LOCAL_LLM_MODEL") or config.HF_MODEL_DEFAULT)
    if provider == "offline":
        return _offline_extractive_generate(question, chunks_with_scores, common_tokens=common_tokens)
    raise ValueError(f"Unknown provider: {provider!r}. This project only supports 'huggingface' (the real, "
                     f"GGUF/llama.cpp generation path - default), 'transformers' (alternate real path, needs "
                     f"more VRAM), and 'offline' (the fast, non-LLM self-test fallback) - no Anthropic/OpenAI.")


def pick_available_provider() -> str:
    """This project's default real generation backend is a local, open-weight
    model (Llama-3.1-8B-Instruct, GGUF-quantized) loaded via `llama-cpp-python`
    - no Anthropic, no OpenAI. Always returns "huggingface" unless
    RAG_FORCE_OFFLINE is set, which forces the zero-dependency offline
    heuristic instead (used by this project's own fast self-tests against
    the synthetic fixture, so they don't need to load an 8B-parameter model
    just to check plumbing)."""
    if os.environ.get("RAG_FORCE_OFFLINE"):
        return "offline"
    return "huggingface"
