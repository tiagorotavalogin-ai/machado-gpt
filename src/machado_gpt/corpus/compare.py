"""Compare our extraction with the luxedo TXT baseline, word by word.

Differences are classified so that only the unexpected ones need reading:
- `hyphen`: same word, luxedo glued the hyphen ("admitime" vs "admiti-me");
- `other`: anything else (missing, extra or reordered words).

Usage:
    uv run python -m machado_gpt.corpus.compare romance conto
"""

from __future__ import annotations

import difflib
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from machado_gpt.corpus.download import RAW_DIR
from machado_gpt.corpus.extract import OUT_DIR

TXT_DIR = RAW_DIR / "txt"


@dataclass
class Comparison:
    work: str
    ours_words: int
    baseline_words: int
    equal_words: int
    hyphen_diffs: int
    other_diffs: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def equal_pct(self) -> float:
        return 100 * self.equal_words / max(self.baseline_words, 1)


def is_glued_hyphen(ours: list[str], baseline: list[str]) -> bool:
    """True when the only difference is hyphens that the baseline removed.

    Covers partial gluing too: "explicou-se-me" vs "explicou-seme".
    """
    return len(ours) == len(baseline) and all(
        a.count("-") > b.count("-") and a.replace("-", "") == b.replace("-", "")
        for a, b in zip(ours, baseline)
    )


def compare_words(work: str, ours: list[str], baseline: list[str]) -> Comparison:
    matcher = difflib.SequenceMatcher(None, ours, baseline, autojunk=False)
    result = Comparison(work, len(ours), len(baseline), 0, 0)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.equal_words += i2 - i1
        elif tag == "replace" and is_glued_hyphen(ours[i1:i2], baseline[j1:j2]):
            result.hyphen_diffs += i2 - i1
        else:
            result.other_diffs.append((tag, " ".join(ours[i1:i2]), " ".join(baseline[j1:j2])))
    return result


def compare(work: str) -> Comparison:
    ours = (OUT_DIR / work).with_suffix(".txt").read_text(encoding="utf-8")
    baseline = (TXT_DIR / work).with_suffix(".txt").read_text(encoding="utf-8")
    return compare_words(work, ours.split(), unicodedata.normalize("NFC", baseline).split())


def main(argv: list[str]) -> None:
    works = sorted(
        str(p.relative_to(OUT_DIR).with_suffix(""))
        for genre in argv
        for p in (OUT_DIR / genre).glob("*.txt")
    )
    print(f"{'obra':34} {'nossas':>7} {'luxedo':>7} {'iguais':>7} {'hífen':>6} {'outras':>6}")
    for work in works:
        c = compare(work)
        print(
            f"{work:34} {c.ours_words:7d} {c.baseline_words:7d} {c.equal_pct:6.2f}% "
            f"{c.hyphen_diffs:6d} {len(c.other_diffs):6d}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
