"""
Part A, steps 2-3 — section-aware chunking + metadata.

Design (see README.md "Chunking strategy" for the full write-up):

1.  Boilerplate removal: running headers/footers (e.g. a repeated document
    title, "Nokia 1830 PSS Planning Guide", bare page numbers) are detected
    by *frequency* - any short line that repeats near-identically across an
    unusually large fraction of pages is noise, not content, and is dropped
    before anything else happens. This prevents every chunk from being
    polluted with the same boilerplate line and prevents boilerplate from
    being mis-detected as a heading.

2.  Heading detection is layout-driven, not keyword-driven: a line is a
    heading candidate if its font is meaningfully larger than the
    document's modal body-text size, or it is bold-and-short, or it matches
    a numbered-heading pattern ("2.6.3 <Title>"). This is exactly the same
    signal a human skimming the PDF uses ("bigger/bolder text = a new
    topic"), so it naturally keeps a component's own heading glued to its
    own description - e.g. "1830 PSS-8 Fan Unit (8FAN)" is detected as a
    heading and everything until the *next* heading belongs to it alone.

3.  One section (heading -> next heading) becomes one chunk whenever its
    length is within/near the 100-300 word target. Sections that are too
    long are split at sentence boundaries with a small sliding overlap
    (never mid-sentence, never mid-number/unit). Sections that are too
    short (a stray heading with almost no body text) are folded into the
    following section rather than left as a near-empty, useless chunk.

4.  Every chunk carries metadata: heading (with breadcrumb of parent
    headings), chapter, first/last physical page, word count, and any
    1830 PSS shelf identifiers mentioned in it (auto-tagged via regex) -
    used later for metadata-filtered retrieval (Part E).
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict

from . import config
from .extraction import PageContent

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+(.{3,90})$")
_PURE_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9(“\"])")
_SHELF_TAG_RES = [re.compile(p, re.IGNORECASE) for p in config.KNOWN_SHELF_PATTERNS]


@dataclass
class RawLine:
    text: str
    size: float
    bold: bool
    page_number: int


@dataclass
class Chunk:
    chunk_id: str
    heading: str
    heading_path: str          # breadcrumb, e.g. "2.6 Shelves > 2.6.3 1830 PSS-8 Fan Unit (8FAN)"
    chapter: int | None
    page_start: int
    page_end: int
    word_count: int
    text: str                  # heading_path + body - what actually gets embedded
    body: str                  # body only - what gets shown/cited to the LLM
    shelf_tags: list[str] = field(default_factory=list)
    part_index: int = 1
    part_total: int = 1

    def to_json(self) -> dict:
        return asdict(self)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _detect_boilerplate(pages: list[PageContent]) -> set[str]:
    """Lines that repeat (near-)verbatim across many pages are running
    headers/footers, not content, and are removed before heading detection.
    """
    counts: Counter[str] = Counter()
    for p in pages:
        seen_this_page: set[str] = set()
        for line in p.lines:
            norm = _normalize_ws(line).lower()
            if not norm or len(norm.split()) > 10:
                continue
            if norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            counts[norm] += 1
    n_pages = max(len(pages), 1)
    boilerplate = {
        norm
        for norm, c in counts.items()
        if c >= max(6, int(0.25 * n_pages))  # repeats on at least a quarter of pages
    }
    return boilerplate


def _flatten_lines(pages: list[PageContent], boilerplate: set[str]) -> list[RawLine]:
    out: list[RawLine] = []
    for p in pages:
        for span in p.spans:
            text = _normalize_ws(span.text)
            if not text:
                continue
            if _PURE_PAGE_NUMBER_RE.match(text):
                continue
            if text.lower() in boilerplate:
                continue
            out.append(RawLine(text=text, size=span.size, bold=span.bold, page_number=p.page_number))
    return out


def _body_font_size(lines: list[RawLine]) -> float:
    sizes = [round(l.size, 1) for l in lines if len(l.text.split()) >= 6]
    if not sizes:
        sizes = [round(l.size, 1) for l in lines] or [10.0]
    return statistics.mode(sizes)


def _heading_level(text: str) -> int | None:
    """Return a heading nesting level (1 = chapter-ish, higher = deeper) if
    `text` looks like a numbered heading, else None."""
    m = _NUMBERED_HEADING_RE.match(text)
    if not m:
        return None
    numbering = m.group(1)
    return numbering.count(".") + 1


def _is_heading(line: RawLine, body_size: float) -> tuple[bool, int]:
    """Heuristic heading classifier. Returns (is_heading, level)."""
    n_words = len(line.text.split())
    numbered_level = _heading_level(line.text)
    if numbered_level is not None and n_words <= 16:
        return True, numbered_level
    size_ratio = line.size / body_size if body_size else 1.0
    ends_sentence = line.text.rstrip().endswith((".", ",", ";"))
    if size_ratio >= 1.15 and n_words <= 14 and not ends_sentence:
        return True, 2
    if line.bold and size_ratio >= 0.98 and n_words <= 10 and not ends_sentence:
        return True, 3
    return False, 0


def _extract_shelf_tags(text: str) -> list[str]:
    tags: set[str] = set()
    for rx in _SHELF_TAG_RES:
        for m in rx.finditer(text):
            norm = re.sub(r"\s+", " ", m.group(0)).upper()
            norm = re.sub(r"\s*-\s*", "-", norm)
            tags.add(norm)
    return sorted(tags)


def _chapter_for_page(page_number: int) -> int | None:
    for ch, rng in config.CHAPTER_RANGES.items():
        if rng["first_page"] <= page_number <= rng["last_page"]:
            return ch
    return None


@dataclass
class _Section:
    heading_path: str
    heading: str
    lines: list[RawLine]

    @property
    def page_start(self) -> int:
        return self.lines[0].page_number if self.lines else -1

    @property
    def page_end(self) -> int:
        return self.lines[-1].page_number if self.lines else -1

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _build_sections(lines: list[RawLine], body_size: float) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []   # (level, text)
    current: _Section | None = None

    def start_section(level: int, heading_text: str):
        nonlocal current
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))
        path = " > ".join(h for _, h in heading_stack)
        current = _Section(heading_path=path, heading=heading_text, lines=[])
        sections.append(current)

    # Preface (content before the very first detected heading in range)
    start_section(1, "Introduction / preamble")

    for line in lines:
        is_head, level = _is_heading(line, body_size)
        if is_head:
            start_section(level, line.text)
        else:
            current.lines.append(line)  # type: ignore[union-attr]

    return [s for s in sections if s.lines]  # drop empty preface if nothing preceded first heading


def _parent_path(heading_path: str) -> str:
    parts = heading_path.split(" > ")
    return " > ".join(parts[:-1])


_NOISE_FLOOR_WORDS = 15  # a heading with next-to-no body is almost certainly a
                          # detection artifact (running title, stray caption),
                          # not a genuine short spec - always safe to fold away.


def _merge_tiny_sections(sections: list[_Section]) -> list[_Section]:
    """Fold sections with too little body text into a neighbour - but ONLY
    when that's safe:

    - Below `_NOISE_FLOOR_WORDS`: almost certainly a heading-detection
      artifact, not real content -> always folded into the next section.
    - Below `MIN_STANDALONE_SECTION_WORDS` AND sharing the same immediate
      parent heading as the next section (e.g. "2.1.1 <Shelf> Fan Unit" and
      "2.1.2 <Shelf> power filter cards" are both children of "2.1 <Shelf>"):
      folding these is safe because they're still about the same component -
      it just consolidates a shelf's own sub-notes into fewer chunks.
    - Below `MIN_STANDALONE_SECTION_WORDS` but the NEXT section belongs to a
      *different* parent (e.g. the tail end of one shelf's section vs. the
      next shelf's heading): left standalone even though short, rather than
      folding it forward and mislabelling it under an unrelated heading/page
      citation. This is the direct fix for the assignment's own example -
      a short "1830 PSS-8 Fan Unit (8FAN)" section must never end up folded
      into, and cited as, an unrelated following section.
    """
    merged: list[_Section] = []
    carry: _Section | None = None
    for sec in sections:
        if carry is not None:
            same_parent = _parent_path(carry.heading_path) == _parent_path(sec.heading_path)
            is_noise = carry.word_count < _NOISE_FLOOR_WORDS
            if same_parent or is_noise or carry.heading == "Introduction / preamble":
                sec = _Section(heading_path=sec.heading_path, heading=sec.heading, lines=carry.lines + sec.lines)
            else:
                merged.append(carry)  # different topic - keep standalone even though short
            carry = None
        if sec.word_count < config.MIN_STANDALONE_SECTION_WORDS and sec is not sections[-1]:
            carry = sec
            continue
        merged.append(sec)
    if carry is not None:
        if merged and _parent_path(merged[-1].heading_path) == _parent_path(carry.heading_path):
            merged[-1].lines.extend(carry.lines)
        else:
            merged.append(carry)
    return merged


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack_sentences(sentences: list[str], target_max: int, overlap_words: int) -> list[str]:
    """Greedily pack sentences into ~target_max-word windows with a small
    trailing-sentence overlap between consecutive windows, never breaking a
    sentence in half."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        w = len(sent.split())
        if current and current_words + w > target_max:
            chunks.append(current)
            # start next window with overlap: carry trailing sentences
            overlap: list[str] = []
            ow = 0
            for s in reversed(current):
                sw = len(s.split())
                if ow + sw > overlap_words:
                    break
                overlap.insert(0, s)
                ow += sw
            current = overlap.copy()
            current_words = ow
        current.append(sent)
        current_words += w
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def _section_to_chunks(sec: _Section, chapter_hint: int | None) -> list[dict]:
    body_text = sec.text
    word_count = sec.word_count
    pieces: list[str]
    if word_count <= config.CHUNK_HARD_MAX_WORDS:
        pieces = [body_text]
    else:
        sentences = _split_sentences(body_text)
        pieces = _pack_sentences(sentences, config.CHUNK_TARGET_MAX_WORDS, config.CHUNK_OVERLAP_WORDS)
        if not pieces:
            pieces = [body_text]

    out = []
    for i, piece in enumerate(pieces, start=1):
        # page range for this piece: approximate by locating which of the
        # section's lines the piece's first/last words fall under.
        out.append(
            {
                "heading": sec.heading,
                "heading_path": sec.heading_path,
                "chapter": chapter_hint,
                "page_start": sec.page_start,
                "page_end": sec.page_end,
                "body": piece,
                "part_index": i,
                "part_total": len(pieces),
            }
        )
    return out


def build_chunks(pages: list[PageContent]) -> list[Chunk]:
    boilerplate = _detect_boilerplate(pages)
    lines = _flatten_lines(pages, boilerplate)
    body_size = _body_font_size(lines)
    sections = _build_sections(lines, body_size)
    sections = _merge_tiny_sections(sections)

    chunks: list[Chunk] = []
    counter = 0
    for sec in sections:
        chapter_hint = _chapter_for_page(sec.page_start)
        for piece in _section_to_chunks(sec, chapter_hint):
            counter += 1
            heading_path = piece["heading_path"]
            body = piece["body"]
            text_for_embedding = f"{heading_path}. {body}"
            shelf_tags = _extract_shelf_tags(heading_path + " " + body)
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter:04d}",
                    heading=piece["heading"],
                    heading_path=heading_path,
                    chapter=piece["chapter"],
                    page_start=piece["page_start"],
                    page_end=piece["page_end"],
                    word_count=len(body.split()),
                    text=text_for_embedding,
                    body=body,
                    shelf_tags=shelf_tags,
                    part_index=piece["part_index"],
                    part_total=piece["part_total"],
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk], path=config.CHUNKS_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")


def load_chunks(path=config.CHUNKS_JSONL) -> list[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Purely numeric summary - safe to print/inspect without exposing the
    document's actual sentences."""
    wc = [c.word_count for c in chunks]
    n_split = sum(1 for c in chunks if c.part_total > 1)
    shelves = Counter(tag for c in chunks for tag in c.shelf_tags)
    return {
        "n_chunks": len(chunks),
        "min_words": min(wc) if wc else 0,
        "max_words": max(wc) if wc else 0,
        "mean_words": round(statistics.mean(wc), 1) if wc else 0,
        "median_words": statistics.median(wc) if wc else 0,
        "n_chunks_in_target_100_300": sum(1 for w in wc if 100 <= w <= 300),
        "n_sections_split_into_multiple_chunks": n_split,
        "n_chapter_1": sum(1 for c in chunks if c.chapter == 1),
        "n_chapter_2": sum(1 for c in chunks if c.chapter == 2),
        "shelf_tag_counts": dict(shelves),
    }
