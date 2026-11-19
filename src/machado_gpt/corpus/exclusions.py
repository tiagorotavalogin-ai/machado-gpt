"""Apply the curated list of passages that are not Machado's own text.

Some passages cannot be detected safely by layout rules: poems Machado
translated, poems by other authors included in his books, a few epigraphs set
without indentation and author's notes printed in the body font. They are listed
by hand in `data/catalog/exclusions.csv`, each with the evidence for it.

Each entry names a range of blocks of the extracted text:
- `inicio`: the first excluded block starts with this text (it must be unique);
- `fim_antes_de`: the first kept block after the range starts with this text
  (the first match after `inicio`); empty means "to the end of the work";
- `acao`: `excluir` removes the range; `revisar` keeps it and reports it, until
  the owner decides.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from machado_gpt.corpus.download import PROJECT_ROOT

EXCLUSIONS = PROJECT_ROOT / "data" / "catalog" / "exclusions.csv"
ACTIONS = {"excluir", "revisar"}


@dataclass(frozen=True)
class Exclusion:
    obra: str
    inicio: str
    fim_antes_de: str
    tipo: str
    autor_original: str
    fonte: str
    acao: str
    observacao: str


def load(path: Path = EXCLUSIONS) -> list[Exclusion]:
    with path.open(encoding="utf-8", newline="") as f:
        entries = [Exclusion(**row) for row in csv.DictReader(f)]
    for e in entries:
        if e.acao not in ACTIONS:
            raise ValueError(f"unknown acao {e.acao!r} for {e.obra}: {e.inicio}")
        if not e.inicio or not e.fonte:
            raise ValueError(f"entry without inicio or fonte in {e.obra}")
    return entries


def locate(blocks: list[str], entry: Exclusion) -> range:
    """Indexes of the blocks covered by an entry; fails loudly if ambiguous."""
    starts = [i for i, b in enumerate(blocks) if b.startswith(entry.inicio)]
    if len(starts) != 1:
        raise ValueError(f"{entry.obra}: {entry.inicio!r} matches {len(starts)} blocks, expected 1")
    start = starts[0]
    if not entry.fim_antes_de:
        return range(start, len(blocks))
    end = next((i for i in range(start + 1, len(blocks)) if blocks[i].startswith(entry.fim_antes_de)), None)
    if end is None:
        raise ValueError(f"{entry.obra}: end {entry.fim_antes_de!r} not found after {entry.inicio!r}")
    return range(start, end)


def apply(
    work: str, blocks: list[str], entries: list[Exclusion]
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    """Remove the `excluir` ranges of a work; report both removed and to-review ranges."""
    drop: set[int] = set()
    removed, review = [], []
    for entry in entries:
        if entry.obra != work:
            continue
        span = locate(blocks, entry)
        record = {
            "inicio": entry.inicio,
            "tipo": entry.tipo,
            "autor_original": entry.autor_original,
            "blocos": len(span),
            "palavras": sum(len(blocks[i].split()) for i in span),
        }
        if entry.acao == "excluir":
            drop.update(span)
            removed.append(record)
        else:
            review.append(record)
    kept = [b for i, b in enumerate(blocks) if i not in drop]
    return kept, removed, review
