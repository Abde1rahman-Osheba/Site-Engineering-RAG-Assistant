"""
Synthetic decoy corpus for SAFE self-testing.

Everything in this file is fictitious, authored from scratch for this
project - a fake "Acme OptiRack" product line with made-up shelves, fan
units, power filter cards, and specs. It deliberately mirrors the
*structure* of the real assignment (numbered headings, a "trick question"
fact that is intentionally absent) so the full pipeline (extraction ->
chunking -> embedding -> retrieval -> hybrid -> rerank -> prompting ->
generation -> refusal) can be exercised and its *output inspected* without
ever touching the real, confidential 1830 PSS PDF.

Run standalone: `py -m tests.synthetic_fixture` builds `tests/fixture.pdf`.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

FIXTURE_PDF = Path(__file__).resolve().parent / "fixture.pdf"

# (heading_or_None, text, font_size, bold)
# font size 16 = chapter heading, 13 = section heading, 11.5 = body text
DOCUMENT: list[tuple[str | None, str, float, bool]] = [
    (None, "1  System concept", 16, True),
    (
        None,
        "Overview. The Acme OptiRack family is a modular transport platform built around a "
        "common shelf architecture, shared fan units, and two supported software load-lines that "
        "control the feature set available on a given deployment.",
        11.5,
        False,
    ),
    (
        None,
        "1.1  Software load-lines",
        13,
        True,
    ),
    (
        None,
        "The Acme OptiRack system supports exactly two software load-lines: Load-Line Alpha, intended "
        "for metro aggregation deployments, and Load-Line Beta, intended for long-haul DWDM "
        "deployments. Every shelf in the product family ships pre-loaded with Load-Line Alpha by "
        "default; Load-Line Beta must be ordered as a separate activation key.",
        11.5,
        False,
    ),
    (None, "2  Shelves and common equipment", 16, True),
    (
        None,
        "2.1  Acme OptiRack-8 shelf",
        13,
        True,
    ),
    (
        None,
        "The Acme OptiRack-8 shelf is the compact member of the family. It provides 12 card slots in a "
        "3 RU (rack unit) footprint, making it suitable for space-constrained central offices. The "
        "shelf requires a horizontal rack aperture of 482.6 mm (19 in); the 23 in wide aperture "
        "variant, common in some legacy central-office racks, is explicitly NOT supported.",
        11.5,
        False,
    ),
    (
        None,
        "2.1.1  Acme OptiRack-8 Fan Unit (8FAN-A)",
        13,
        True,
    ),
    (
        None,
        "The 8FAN-A fan unit is the sole fan tray supported on the OptiRack-8 shelf. It mounts in the "
        "lower rear of the shelf and provides front-to-back airflow rated for the shelf's full card "
        "complement at 40 degrees C ambient.",
        11.5,
        False,
    ),
    (
        None,
        "2.1.2  Acme OptiRack-8 power filter cards",
        13,
        True,
    ),
    (
        None,
        "Two power filter cards are supported on the OptiRack-8 shelf: the PF-8A (single -48V feed) "
        "and the PF-8B (dual-feed, redundant). Exactly one power filter card must be installed per "
        "shelf.",
        11.5,
        False,
    ),
    (
        None,
        "2.2  Acme OptiRack-32 shelf",
        13,
        True,
    ),
    (
        None,
        "The Acme OptiRack-32 shelf is the high-density member of the family, providing 32 card slots "
        "in a 16 RU footprint. It is intended for large central-office deployments where slot density "
        "matters more than shelf compactness.",
        11.5,
        False,
    ),
    (
        None,
        "2.2.1  Acme OptiRack-32 fan units",
        13,
        True,
    ),
    (
        None,
        "Two fan units are supported on the OptiRack-32 shelf, depending on deployment density: the "
        "32FAN-X (standard airflow, for typical card fills) and the 32FAN-Y (high-airflow variant, "
        "required when more than 24 of the 32 slots are populated with high-power amplifier cards).",
        11.5,
        False,
    ),
    (
        None,
        "2.3  Acme OptiRack-16II shelf",
        13,
        True,
    ),
    (
        None,
        "The Acme OptiRack-16II shelf is a mid-density variant offering 16 card slots. It uses a "
        "single fan unit, the 16FAN-Z, which is not interchangeable with any fan unit from the -8 or "
        "-32 shelves due to a different mounting bracket.",
        11.5,
        False,
    ),
]

# NOTE: deliberately no mention anywhere of unamplified optical reach in km -
# this is the fictitious analogue of assignment Q8, to test refusal behaviour.


def build_fixture_pdf(path: Path = FIXTURE_PDF) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    y = 60.0
    left_margin = 50.0
    max_width = 495.0

    def wrap(text: str, size: float, width: float) -> list[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if fitz.get_text_length(trial, fontsize=size) > width:
                if cur:
                    lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    for _, text, size, bold in DOCUMENT:
        font = "helv" if not bold else "hebo"
        for line in wrap(text, size, max_width):
            if y > 780:
                page = doc.new_page(width=595, height=842)
                y = 60.0
            page.insert_text((left_margin, y), line, fontsize=size, fontname=font)
            y += size * 1.5
        y += size * 0.8  # paragraph gap

    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = build_fixture_pdf()
    print(f"Synthetic fixture PDF written to {out} ({out.stat().st_size} bytes)")
