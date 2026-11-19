"""Build the final corpus: catalog, deduplication, split by work and statistics.

1. Group works that share text (8-word sequences in common). A group always
   lands in a single split, so no version of a work leaks across splits.
2. Drop a work that is almost entirely contained in another (duplicate text
   would weigh twice in training); the larger work is kept.
3. Split by group, per genre: groups are ordered by SHA-256 of "seed:group" and
   fill test and validation up to ~10% of the genre's words each; a group that
   would overshoot a target goes to training. If a split is left empty for a
   genre, it takes the first group of at most 25% of the genre. Deterministic
   and independent of file order.
4. Write the catalog (versioned) and the split texts (not versioned).

Usage:
    uv run python -m machado_gpt.corpus.build
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.corpus.download import MANIFEST as SOURCE_MANIFEST
from machado_gpt.corpus.extract import OUT_DIR as EXTRACTED_DIR

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CATALOG = PROJECT_ROOT / "data" / "catalog" / "works.csv"

SEED = "machado-gpt-split-v1"
SPLITS = ("train", "val", "test")
TARGET_SHARE = {"val": 0.10, "test": 0.10}
OVERSHOOT = 1.25  # a split may end up to 25% above its target
FORCED_MAX_SHARE = 0.25  # largest group that may fill an otherwise empty split
SHINGLE = 8  # words per sequence when looking for shared text
SHARED_MIN = 50  # shared sequences that make two works one group
CONTAINED_MAX = 0.90  # share of a work found in another above which it is a duplicate
EXCLUDED_GENRES = {"tradução"}
WORK_SEPARATOR = "\n\n\n"  # paragraphs are separated by one blank line; works by two
LICENSE = "domínio público (Machado de Assis, 1839-1908)"

WORD = re.compile(r"\w+")


@dataclass
class Work:
    name: str  # "romance/domCasmurro"
    text: str

    @property
    def genre(self) -> str:
        return self.name.split("/")[0]

    @property
    def words(self) -> int:
        return len(self.text.split())


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def shared_text(works: list[Work]) -> list[tuple[str, str, int, float]]:
    """Pairs of works sharing at least SHARED_MIN sequences: (a, b, shared, share of the smaller)."""
    sets = {w.name: shingles(w.text) for w in works}
    pairs = []
    for a, b in itertools.combinations(sorted(sets), 2):
        shared = len(sets[a] & sets[b])
        if shared >= SHARED_MIN:
            smaller = min(len(sets[a]), len(sets[b])) or 1
            pairs.append((a, b, shared, shared / smaller))
    return pairs


def duplicates(pairs: list[tuple[str, str, int, float]], words: dict[str, int]) -> dict[str, str]:
    """Works almost entirely contained in another: {dropped: kept}."""
    dropped = {}
    for a, b, _, share in pairs:
        if share >= CONTAINED_MAX:
            small, large = sorted((a, b), key=lambda n: (words[n], n))
            dropped[small] = large
    return dropped


def group_works(names: list[str], pairs: list[tuple[str, str, int, float]]) -> dict[str, str]:
    """Union-find: map each work to its group key (sorted member names joined by '+')."""
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b, _, _ in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    members: dict[str, list[str]] = defaultdict(list)
    for n in names:
        members[find(n)].append(n)
    return {n: "+".join(sorted(members[find(n)])) for n in names}


def order_key(group: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def assign_splits(works: list[Work], group_of: dict[str, str]) -> dict[str, str]:
    """Split groups per genre; a group's genre is that of its largest work."""
    group_words: Counter[str] = Counter()
    group_genre: dict[str, tuple[int, str]] = {}
    for w in works:
        g = group_of[w.name]
        group_words[g] += w.words
        group_genre[g] = max(group_genre.get(g, (0, "")), (w.words, w.genre))
    by_genre: dict[str, list[str]] = defaultdict(list)
    for g, (_, genre) in group_genre.items():
        by_genre[genre].append(g)

    split_of_group = {}
    for genre, groups in by_genre.items():
        total = sum(group_words[g] for g in groups)
        filled = Counter()
        for g in sorted(groups, key=order_key):
            size = group_words[g]
            chosen = "train"
            for split in ("test", "val"):
                target = TARGET_SHARE[split] * total
                if filled[split] < target and filled[split] + size <= OVERSHOOT * target:
                    chosen = split
                    break
            filled[chosen] += size
            split_of_group[g] = chosen
        # Every genre should be measured in validation and test. When all groups
        # of a genre are too large for the target (e.g. poetry books of 15-20k
        # words for a 10k target), take the first group, in hash order, that is
        # at most a quarter of the genre.
        for split in ("test", "val"):
            if filled[split] == 0:
                for g in sorted(groups, key=order_key):
                    if split_of_group[g] == "train" and group_words[g] <= FORCED_MAX_SHARE * total:
                        split_of_group[g] = split
                        filled[split] += group_words[g]
                        break
    return {w.name: split_of_group[group_of[w.name]] for w in works}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_works(extracted: Path = EXTRACTED_DIR) -> list[Work]:
    works = []
    for path in sorted(extracted.glob("*/*.txt")):
        name = str(path.relative_to(extracted).with_suffix(""))
        if name.split("/")[0] in EXCLUDED_GENRES:
            continue
        works.append(Work(name, path.read_text(encoding="utf-8").strip()))
    return works


def main() -> None:
    all_works = load_works()
    with SOURCE_MANIFEST.open(encoding="utf-8") as f:
        pdf_sha = {r["path"]: r["sha256"] for r in csv.DictReader(f)}

    excluded_whole = {w.name for w in all_works if not w.text}  # removed entirely by the exclusion list
    works = [w for w in all_works if w.text]
    words = {w.name: w.words for w in works}
    pairs = shared_text(works)
    dropped = duplicates(pairs, words)
    kept = [w for w in works if w.name not in dropped]
    group_of = group_works([w.name for w in kept], pairs)
    split_of = assign_splits(kept, group_of)

    rows = []
    for w in all_works:
        meta = json.loads((EXTRACTED_DIR / f"{w.name}.json").read_text(encoding="utf-8"))
        if w.name in excluded_whole:
            status, split = "excluída: lista manual (data/catalog/exclusions.csv)", ""
        elif w.name in dropped:
            status, split = f"excluída: duplicata contida em {dropped[w.name]}", ""
        else:
            status, split = "incluída", split_of[w.name]
        rows.append(
            {
                "obra": w.name,
                "genero": w.genre,
                "conjunto": split,
                "situacao": status,
                "grupo": group_of.get(w.name, ""),
                "palavras": w.words,
                "caracteres": len(w.text),
                "edicao_referencia": " ".join(meta["edition_header"]),
                "fonte": f"luxedo/dataset-machado-de-assis, pdf/{w.name}.pdf",
                "licenca": LICENSE,
                "sha256_pdf": pdf_sha[f"pdf/{w.name}.pdf"],
                "sha256_texto": sha256_text(w.text),
            }
        )
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        texts = [w.text for w in kept if split_of[w.name] == split]
        (PROCESSED_DIR / f"{split}.txt").write_text(WORK_SEPARATOR.join(texts) + "\n", encoding="utf-8")

    print(f"{len(all_works)} works, {len(kept)} included, {len(dropped)} duplicates, "
          f"{len(excluded_whole)} excluded by the list")
    print(f"catalog: {CATALOG.relative_to(PROJECT_ROOT)}")
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.txt"
        n = sum(words[w.name] for w in kept if split_of[w.name] == split)
        print(f"{split:5} {n:8} words -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
