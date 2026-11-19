"""Survey the layout of the downloaded PDFs before writing extraction rules.

For each PDF it reports what the extraction will need to handle: pages without
a text layer (OCR), the body font size, how much text is in smaller fonts
(edition header and footnotes), lines that repeat across pages (running
headers or footers) and the vertical gaps between lines (paragraph breaks).

Usage:
    uv run python -m machado_gpt.corpus.inspect_pdfs            # all PDFs
    uv run python -m machado_gpt.corpus.inspect_pdfs romance     # one genre
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber

from machado_gpt.corpus.download import PROJECT_ROOT, RAW_DIR

PDF_DIR = RAW_DIR / "pdf"
REPORT = PROJECT_ROOT / "data" / "interim" / "pdf_survey.csv"

DIGITS = re.compile(r"\d+")
EDITION_HEADER = re.compile(
    r"textos?[- ]?(fonte|de referência)|edição (de )?referência", re.IGNORECASE
)


@dataclass
class PdfSurvey:
    path: str
    pages: int
    pages_without_text: int
    body_size: float
    small_text_pct: float
    repeated_lines: int
    line_gap: int
    paragraph_gap: int
    hyphen_line_ends: int
    edition_header: bool


def has_edition_header(first_page_text: str) -> bool:
    """Detect the modern edition note by its label (wording varies between PDFs)."""
    return bool(EDITION_HEADER.search(first_page_text))


def body_font_size(sizes: Counter[float]) -> float:
    """The font size used by most characters is the body text."""
    return sizes.most_common(1)[0][0] if sizes else 0.0


def repeated_line_count(lines_per_page: list[list[str]], min_share: float = 0.3) -> int:
    """Count lines that appear on many pages (digits ignored, so page numbers match)."""
    if len(lines_per_page) < 5:
        return 0
    seen: Counter[str] = Counter()
    for lines in lines_per_page:
        seen.update({DIGITS.sub("#", line.strip()) for line in lines if line.strip()})
    threshold = max(3, int(min_share * len(lines_per_page)))
    return sum(1 for count in seen.values() if count >= threshold)


def two_most_common_gaps(gaps: Counter[int]) -> tuple[int, int]:
    """Smallest of the two most common gaps is the line gap, the other the paragraph gap."""
    top = sorted(gap for gap, _ in gaps.most_common(2))
    if not top:
        return 0, 0
    if len(top) == 1:
        return top[0], 0
    return top[0], top[1]


def survey(pdf_path: Path) -> PdfSurvey:
    sizes: Counter[float] = Counter()
    gaps: Counter[int] = Counter()
    lines_per_page: list[list[str]] = []
    pages_without_text = 0
    hyphens = 0
    first_page_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            chars = [c for c in page.chars if c["text"].strip()]
            if not chars:
                pages_without_text += 1
                lines_per_page.append([])
                continue
            sizes.update(round(c["size"], 1) for c in chars)
            lines = page.extract_text_lines()
            texts = [line["text"] for line in lines]
            lines_per_page.append(texts)
            if i == 0:
                first_page_text = "\n".join(texts)
            hyphens += sum(1 for t in texts if t.rstrip().endswith("-"))
            gaps.update(round(b["top"] - a["top"]) for a, b in zip(lines, lines[1:]))
        n_pages = len(pdf.pages)

    body = body_font_size(sizes)
    total = sum(sizes.values())
    small = sum(n for size, n in sizes.items() if size < body - 0.5)
    line_gap, paragraph_gap = two_most_common_gaps(gaps)
    return PdfSurvey(
        path=str(pdf_path.relative_to(PDF_DIR)),
        pages=n_pages,
        pages_without_text=pages_without_text,
        body_size=body,
        small_text_pct=round(100 * small / total, 2) if total else 0.0,
        repeated_lines=repeated_line_count(lines_per_page),
        line_gap=line_gap,
        paragraph_gap=paragraph_gap,
        hyphen_line_ends=hyphens,
        edition_header=has_edition_header(first_page_text),
    )


def main(argv: list[str]) -> None:
    root = PDF_DIR / argv[0] if argv else PDF_DIR
    rows = [survey(p) for p in sorted(root.rglob("*.pdf"))]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"{len(rows)} PDFs surveyed -> {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
