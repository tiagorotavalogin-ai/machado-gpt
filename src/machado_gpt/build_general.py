"""Build the general pretraining corpus from Portuguese Wikisource (stage 6).

1. Read the pinned dump: works typed in the main namespace, and scanned books
   rebuilt by joining their "Página:" pages in page order.
2. Keep scanned pages proofread at least once (quality 3 or 4): level 1 is raw
   OCR, level 0 has no text, level 2 is marked as problematic.
3. Remove Machado de Assis, so that the pretraining never reads the validation
   or test works (nor the training works, which come only in fine-tuning):
   - by author: the work's header or the scanned book's index;
   - by text: any paragraph touched by 8-word sequences shared with the
     Machado corpus, compared after normalizing the spelling (lower case, no
     accents, ph->f, th->t, y->i, z->s, double letters single, silent h), so
     that an edition in the original spelling still matches the modernized
     one. Sequences run across paragraph boundaries, so that a poem typed one
     verse per paragraph is still caught.
4. Hold out ~1% of the documents (by SHA-256 of the title) as a general
   validation set, to follow the pretraining itself.
5. Write the corpus (not versioned) and the statistics (versioned).

Usage:
    uv run python -m machado_gpt.build_general
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mwparserfromhell as mw

from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.wikisource import INDEX, MAIN, PAGE, download, iter_pages, wikitext_to_text

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "general"  # not versioned
RESULTS = PROJECT_ROOT / "results" / "06-wikisource.json"
MACHADO = "machado de assis"
SHINGLE = 8
MIN_QUALITY = 3
MIN_SHARED = 2  # a paragraph sharing this many 8-word sequences with Machado is removed
DOC_SEPARATOR = "\n\n\n"
VAL_FRACTION = 0.01


def is_val(title: str) -> bool:
    digest = hashlib.sha256(f"machado-gpt-general-v1:{title}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < VAL_FRACTION


# ---- spelling-insensitive 8-word sequences --------------------------------------


def normalize_word(word: str) -> str:
    w = "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))
    w = w.replace("ph", "f").replace("th", "t").replace("y", "i").replace("z", "s")
    w = re.sub(r"(?<![cln])h", "", w)  # silent h ("hontem", "sahir"), keeping ch, lh, nh
    return re.sub(r"(.)\1", r"\1", w)  # "ella", "succede", "commercio"


def words(text: str) -> list[str]:
    return [normalize_word(w) for w in re.findall(r"\w+", text)]


def shingles(text: str) -> set[int]:
    ws = words(text)
    return {hash(tuple(ws[i : i + SHINGLE])) for i in range(len(ws) - SHINGLE + 1)}


# ---- reading --------------------------------------------------------------------


QUALITY = re.compile(r'<pagequality\s+level="(\d)"')


def header_author(wikitext: str) -> str:
    """The "autor"/"Autor" parameter of the first top-level template that has one:
    the header of a work ({{navegar|autor=...}}) or the index of a scanned book."""
    for t in mw.parse(wikitext).filter_templates(recursive=False):
        for param in t.params:
            if str(param.name).strip().lower() == "autor":
                return wikitext_to_text(str(param.value)).strip()
    return ""


def page_key(title: str) -> tuple[str, int]:
    """'Página:Book.djvu/23' -> ('Book.djvu', 23); a page without a number is its own book."""
    name = title.split(":", 1)[1]
    book, _, number = name.rpartition("/")
    return (book, int(number)) if book and number.isdigit() else (name, 0)


@dataclass
class Document:
    title: str
    source: str  # "main" or "scanned"
    author: str
    paragraphs: list[str] = field(default_factory=list)


def read_documents(dump: Path, stats: Counter) -> list[Document]:
    docs: list[Document] = []
    index_author: dict[str, str] = {}
    pages: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for raw in iter_pages(dump):
        if raw.ns == INDEX:
            index_author[raw.title.split(":", 1)[1]] = header_author(raw.text)
        elif raw.ns == PAGE:
            m = QUALITY.search(raw.text)
            level = int(m.group(1)) if m else -1
            text = wikitext_to_text(raw.text)
            stats[f"scanned_words_quality_{level}"] += len(text.split())
            if level >= MIN_QUALITY and text:
                book, number = page_key(raw.title)
                pages[book].append((number, text))
        elif raw.ns == MAIN:
            text = wikitext_to_text(raw.text)
            if text:
                docs.append(Document(raw.title, "main", header_author(raw.text), split_paragraphs(text)))
    for book, numbered in pages.items():
        numbered.sort()
        text = "\n\n".join(t for _, t in numbered)  # a paragraph cut by a page break stays cut in two
        docs.append(Document(book, "scanned", index_author.get(book, ""), split_paragraphs(text)))
    return docs


def split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---- Machado filter -------------------------------------------------------------


def shared_per_paragraph(paragraphs: list[str], reference: set[int]) -> list[int]:
    """How many 8-word sequences shared with the reference touch each paragraph.

    The sequences run over the whole document, across paragraph boundaries: a
    poem with one verse per paragraph has few sequences inside each verse, but
    many that join consecutive verses."""
    ws, owner = [], []
    for i, p in enumerate(paragraphs):
        w = words(p)
        ws += w
        owner += [i] * len(w)
    counts = [0] * len(paragraphs)
    for start in range(len(ws) - SHINGLE + 1):
        if hash(tuple(ws[start : start + SHINGLE])) in reference:
            for i in set(owner[start : start + SHINGLE]):
                counts[i] += 1
    return counts


def remove_machado(docs: list[Document], machado_text: str, stats: Counter) -> list[Document]:
    reference = shingles(machado_text)
    kept = []
    for doc in docs:
        n_words = sum(len(p.split()) for p in doc.paragraphs)
        if MACHADO in doc.author.lower():
            stats["machado_by_author_docs"] += 1
            stats["machado_by_author_words"] += n_words
            continue
        paragraphs = []
        for p, shared in zip(doc.paragraphs, shared_per_paragraph(doc.paragraphs, reference)):
            if shared >= MIN_SHARED:
                stats["machado_by_text_paragraphs"] += 1
                stats["machado_by_text_words"] += len(p.split())
            else:
                paragraphs.append(p)
        if paragraphs:
            kept.append(Document(doc.title, doc.source, doc.author, paragraphs))
    return kept


def load_split(name: str) -> str:
    from machado_gpt.bigram import load_split as load  # imported here: bigram pulls in torch

    return load(name)


def main() -> None:
    dump = download()
    stats: Counter = Counter()
    docs = read_documents(dump, stats)
    stats["documents_read"] = len(docs)
    stats["words_read"] = sum(len(p.split()) for d in docs for p in d.paragraphs)
    machado = "\n\n".join(load_split(s) for s in ("train", "val", "test"))
    docs = remove_machado(docs, machado, stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_words, characters = {}, 0
    for split in ("train", "val"):
        part = [d for d in docs if is_val(d.title) == (split == "val")]
        text = DOC_SEPARATOR.join("\n\n".join(d.paragraphs) for d in part) + "\n"
        (OUT_DIR / f"{split}.txt").write_text(text, encoding="utf-8")
        split_words[split] = sum(len(p.split()) for d in part for p in d.paragraphs)
        characters += len(text)

    by_source = Counter()
    by_author = Counter()
    for d in docs:
        n = sum(len(p.split()) for p in d.paragraphs)
        by_source[d.source] += n
        by_author[d.author or "(sem autor)"] += n
    results = {
        "dump": dump.name,
        "min_quality": MIN_QUALITY,
        "documents": len(docs),
        "words": sum(by_source.values()),
        "characters": characters,
        "words_by_split": split_words,
        "words_by_source": dict(by_source),
        "top_authors": by_author.most_common(25),
        "stats": dict(sorted(stats.items())),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in results.items():
        if k != "top_authors":
            print(f"{k}: {v}")
    print(f"corpus -> {OUT_DIR.relative_to(PROJECT_ROOT)}/train.txt, val.txt")


if __name__ == "__main__":
    main()
