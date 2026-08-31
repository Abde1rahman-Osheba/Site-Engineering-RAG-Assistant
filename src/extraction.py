"""
Part A, step 1 — PDF extraction.

Extracts ONLY the assigned page range (Chapters 1-2, physical pages
PAGE_START..PAGE_END) from the source PDF, using PyMuPDF (fitz) so that we
keep line-level layout metadata (font size / boldness / position). That
metadata is what lets `chunking.py` detect section headings automatically
instead of guessing from character counts alone.

Two artefacts are produced:
  1. `EXTRACTED_PAGES_JSON` - one JSON line per page with its layout-tagged
     text spans (input to the heading-aware chunker).
  2. `EXTRACTED_MARKDOWN`   - a human-readable markdown dump of the same
     range, satisfying the assignment's "extract to plain-text/markdown
     first" requirement.

This module also extracts embedded raster images per page (Part E / the
multimodal extension) so diagrams/photos in the planning guide can be
indexed alongside text.

Note on confidentiality: this module is meant to be *run by the user*
against the real source PDF. It never prints extracted text to stdout -
only page/line/image counts - so that running it does not surface the
document's content in a terminal transcript.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias is deprecated upstream)

from . import config


@dataclass
class TextSpan:
    text: str
    size: float
    bold: bool
    font: str


@dataclass
class PageContent:
    page_number: int          # 1-indexed physical page number in the PDF
    spans: list[TextSpan] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)   # plain text lines, same order as spans->joined
    image_count: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @property
    def plain_text(self) -> str:
        return "\n".join(self.lines)


def _file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """Content hash used purely for cache-versioning - never used to inspect content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprint() -> dict:
    """A small manifest fragment identifying exactly which source + page range
    produced an artefact, so downstream caches can detect staleness."""
    return {
        "source_pdf": config.SOURCE_PDF.name,
        "source_sha256": _file_sha256(config.SOURCE_PDF),
        "page_start": config.PAGE_START,
        "page_end": config.PAGE_END,
    }


def _line_from_spans(spans: list[dict]) -> tuple[str, float, bool, str]:
    text = "".join(s["text"] for s in spans).strip()
    size = max((s["size"] for s in spans), default=0.0)
    font = spans[0]["font"] if spans else ""
    bold = any(("bold" in s["font"].lower()) or (s["flags"] & 2**4) for s in spans)
    return text, size, bold, font


def extract_page_range(
    pdf_path: Path = config.SOURCE_PDF,
    first_page: int = config.PAGE_START,
    last_page: int = config.PAGE_END,
) -> list[PageContent]:
    """Extract physical pages [first_page, last_page] (1-indexed, inclusive).

    Uses PyMuPDF's structured ("dict") text extraction to retain font size /
    weight per line, which downstream heading detection relies on.
    """
    doc = fitz.open(pdf_path)
    try:
        if last_page > doc.page_count:
            raise ValueError(
                f"last_page={last_page} exceeds document page count={doc.page_count}"
            )
        pages: list[PageContent] = []
        for page_number in range(first_page, last_page + 1):
            page = doc[page_number - 1]  # fitz is 0-indexed
            raw = page.get_text("dict")
            spans: list[TextSpan] = []
            lines: list[str] = []
            for block in raw.get("blocks", []):
                if block.get("type") != 0:  # not a text block (e.g. image block)
                    continue
                for line in block.get("lines", []):
                    line_spans = line.get("spans", [])
                    if not line_spans:
                        continue
                    text, size, bold, font = _line_from_spans(line_spans)
                    if not text:
                        continue
                    spans.append(TextSpan(text=text, size=size, bold=bold, font=font))
                    lines.append(text)
            image_count = len(page.get_images(full=True))
            pages.append(
                PageContent(page_number=page_number, spans=spans, lines=lines, image_count=image_count)
            )
        return pages
    finally:
        doc.close()


def save_extracted(pages: list[PageContent]) -> dict:
    """Persist the extracted page range as JSONL (structured) + Markdown (readable).

    Returns a manifest dict (fingerprint + counts) that is also written to
    disk so later stages can detect whether re-extraction is needed.
    """
    with open(config.EXTRACTED_PAGES_JSON, "w", encoding="utf-8") as jf:
        for p in pages:
            jf.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")

    with open(config.EXTRACTED_MARKDOWN, "w", encoding="utf-8") as mf:
        mf.write(f"# 1830 PSS Planning Guide - Chapters 1-2 (pages {pages[0].page_number}-{pages[-1].page_number})\n\n")
        for p in pages:
            mf.write(f"\n\n<!-- page {p.page_number} -->\n\n")
            mf.write(p.plain_text)
            mf.write("\n")

    manifest = {
        **source_fingerprint(),
        "num_pages_extracted": len(pages),
        "total_lines": sum(len(p.lines) for p in pages),
        "total_words": sum(len(p.plain_text.split()) for p in pages),
        "total_embedded_images": sum(p.image_count for p in pages),
    }
    with open(config.EXTRACTED_DIR / "extraction_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_extracted() -> list[PageContent]:
    pages = []
    with open(config.EXTRACTED_PAGES_JSON, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            spans = [TextSpan(**s) for s in d["spans"]]
            pages.append(PageContent(page_number=d["page_number"], spans=spans, lines=d["lines"], image_count=d.get("image_count", 0)))
    return pages


def extraction_is_fresh() -> bool:
    """True if a previous extraction exists and matches the current source
    file + page range (so we can skip re-extracting)."""
    manifest_path = config.EXTRACTED_DIR / "extraction_manifest.json"
    if not (manifest_path.exists() and config.EXTRACTED_PAGES_JSON.exists()):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError:
        return False
    fp = source_fingerprint()
    return all(manifest.get(k) == v for k, v in fp.items())


def extract_images(
    pdf_path: Path = config.SOURCE_PDF,
    first_page: int = config.PAGE_START,
    last_page: int = config.PAGE_END,
    out_dir: Path = config.EXTRACTED_DIR / "images",
    min_side_px: int = 120,
) -> list[dict]:
    """Extract embedded raster images (figures/diagrams/photos) from the page
    range for the multimodal index. Tiny images (<min_side_px on both sides -
    typically bullets/rule lines/logos) are skipped.

    Returns a list of metadata dicts {path, page_number, width, height, xref}.
    Image bytes are written to disk; this function does not display them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    records: list[dict] = []
    try:
        for page_number in range(first_page, last_page + 1):
            page = doc[page_number - 1]
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                except Exception:
                    continue
                if pix.width < min_side_px and pix.height < min_side_px:
                    pix = None
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK -> convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out_path = out_dir / f"p{page_number:04d}_img{img_idx:02d}_x{xref}.png"
                pix.save(out_path)
                records.append(
                    {
                        "path": str(out_path),
                        "page_number": page_number,
                        "width": pix.width,
                        "height": pix.height,
                        "xref": xref,
                    }
                )
                pix = None
        return records
    finally:
        doc.close()


def render_page_thumbnails(
    pdf_path: Path = config.SOURCE_PDF,
    first_page: int = config.PAGE_START,
    last_page: int = config.PAGE_END,
    out_dir: Path = config.EXTRACTED_DIR / "page_renders",
    dpi: int = 110,
) -> list[dict]:
    """Render each page in the range to a PNG. Useful for pages whose key
    information is a *diagram/table* rather than embedded raster images
    (common in PDFs where tables are drawn as vector graphics) - these
    page-level renders become additional multimodal index entries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    records: list[dict] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    try:
        for page_number in range(first_page, last_page + 1):
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=mat)
            out_path = out_dir / f"page_{page_number:04d}.png"
            pix.save(out_path)
            records.append({"path": str(out_path), "page_number": page_number, "width": pix.width, "height": pix.height})
        return records
    finally:
        doc.close()
