"""Encode the general corpus with the stage 4 tokenizer (stage 6).

The tokenizer is the one trained on the Machado training split, unchanged: only
one thing changes from stage 5 (the pretraining), so the comparison stays clean.
Encoding runs in parallel over documents; the exact round trip is checked on
both splits, as in stage 4.

Usage:
    uv run python -m machado_gpt.tokenize_general
"""

from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy

from machado_gpt.bpe import BPE
from machado_gpt.build_general import DOC_SEPARATOR, OUT_DIR, RESULTS
from machado_gpt.corpus.download import PROJECT_ROOT

TOKENIZER = PROJECT_ROOT / "runs" / "04-bpe" / "bpe-4096.json"
RUN_DIR = PROJECT_ROOT / "runs" / "06-general"  # not versioned


def sha256_file(path: Path) -> str:  # not imported from train_bpe, which pulls in torch
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


MAX_PROCESSES = 6  # each worker holds its own copy of the tokenizer; the machine has little RAM

_bpe: BPE | None = None


def _init(tokenizer: Path) -> None:
    global _bpe
    _bpe = BPE.load(tokenizer)


def _encode(piece: str) -> numpy.ndarray:
    """Encode one document and check its exact round trip right away, so the
    main process never rebuilds the whole text. If every piece comes back
    exact, so does their concatenation."""
    ids = _bpe.encode(piece)
    if _bpe.decode(ids) != piece:
        raise AssertionError(f"round trip failed on a document starting {piece[:60]!r}")
    return numpy.array(ids, dtype=numpy.uint16)  # 2 bytes per token instead of a Python int


def pieces(text: str) -> list[str]:
    """Documents, each but the first preceded by the separator; joined, they
    give back the text."""
    docs = text.split(DOC_SEPARATOR)
    return docs[:1] + [DOC_SEPARATOR + d for d in docs[1:]]


def encode_parallel(text: str, processes: int, tokenizer: Path = TOKENIZER) -> numpy.ndarray:
    """Documents are encoded independently; joining their ids in order equals
    encoding the whole text when no document starts or ends with whitespace
    (true for the general corpus). The separator goes at the start of the next
    document: in the whole text, "\n\n\nWord" pre-tokenizes as "\n\n" + "\n" +
    "Word", and a piece starting with the separator does the same, while a
    piece ending with it would give a single "\n\n\n"."""
    with Pool(processes, initializer=_init, initargs=(tokenizer,)) as pool:
        parts = list(pool.imap(_encode, pieces(text), chunksize=64))
    return numpy.concatenate(parts) if parts else numpy.array([], dtype=numpy.uint16)


def main() -> None:
    assert BPE.load(TOKENIZER).vocab_size <= 2**16  # ids fit in uint16
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    processes = max(1, min(MAX_PROCESSES, (os.cpu_count() or 2) - 1))
    out = {"tokenizer_sha256": sha256_file(TOKENIZER), "splits": {}}
    for split in ("train", "val"):
        text = (OUT_DIR / f"{split}.txt").read_text(encoding="utf-8")
        ids = encode_parallel(text, processes)
        path = RUN_DIR / f"{split}.bin"
        ids.tofile(path)
        n_bytes = len(text.encode("utf-8"))
        del text
        out["splits"][split] = {"tokens": len(ids), "bytes": n_bytes, "bytes_per_token": n_bytes / len(ids),
                                "sha256": sha256_file(path)}
        print(f"general {split}: {len(ids):,} tokens  {n_bytes / len(ids):.2f} bytes/token  round trip exact")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["tokens"] = out
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tokens -> {RUN_DIR.relative_to(PROJECT_ROOT)} | results -> {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
