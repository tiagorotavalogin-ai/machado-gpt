"""Train the BPE tokenizer on the training split and encode the three splits.

- trains the tokenizer on train only (validation and test never influence it);
- checks the exact UTF-8 round trip on the whole of train, val and test;
- writes the token ids as uint16 binary files for stage 5;
- with --sweep, trains once to 16,384 tokens and measures every smaller size
  (merges are learned in order, so the first N merges are the size-(256+N)
  tokenizer), to justify the vocabulary size.

Usage:
    uv run python -m machado_gpt.tokenize_corpus           # V = 4096 (~2 min)
    uv run python -m machado_gpt.tokenize_corpus --sweep   # plus the size study (~4 min)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy

from machado_gpt.bigram import load_split
from machado_gpt.bpe import BPE, pre_tokenize
from machado_gpt.corpus.download import PROJECT_ROOT

VOCAB_SIZE = 4096
SWEEP_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)
RUN_DIR = PROJECT_ROOT / "runs" / "04-bpe"  # not versioned: tokenizer and encoded splits
RESULTS = PROJECT_ROOT / "results" / "04-bpe.json"
EXAMPLE = "Não consultes dicionários. Casmurro não está aqui no sentido que eles lhe dão."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep(train: str, val: str) -> list[dict[str, float]]:
    full = BPE.train(train, max(SWEEP_SIZES))
    val_bytes = len(val.encode("utf-8"))
    rows = []
    for size in SWEEP_SIZES:
        bpe = BPE(full.merges[: size - 256])
        counts = Counter(bpe.encode(train))
        merged = sorted(counts[i] for i in range(256, size))
        val_tokens = len(bpe.encode(val))
        rows.append({
            "vocab_size": size,
            "val_bytes_per_token": val_bytes / val_tokens,
            "bytes_in_128_tokens": 128 * val_bytes / val_tokens,
            "train_tokens": sum(counts.values()),
            "merges_under_50_occurrences": sum(1 for c in merged if c < 50) / len(merged) if merged else 0.0,
            "median_occurrences_per_merge": merged[len(merged) // 2] if merged else 0,
        })
        print(f"  V={size:6}  {rows[-1]['val_bytes_per_token']:.2f} bytes/token  "
              f"rare merges {100 * rows[-1]['merges_under_50_occurrences']:.1f}%")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    texts = {name: load_split(name) for name in ("train", "val", "test")}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    if args.sweep:
        print("vocabulary size study:")
        results["sweep"] = sweep(texts["train"], texts["val"])

    bpe = BPE.train(texts["train"], VOCAB_SIZE)
    tok_path = RUN_DIR / f"bpe-{VOCAB_SIZE}.json"
    bpe.save(tok_path)
    print(f"tokenizer: {bpe.vocab_size} tokens, trained on train only -> {tok_path.relative_to(PROJECT_ROOT)}")

    splits = {}
    for name, text in texts.items():
        ids = bpe.encode(text)
        if bpe.decode(ids) != text:
            raise AssertionError(f"round trip failed on {name}")
        path = RUN_DIR / f"{name}.bin"
        numpy.array(ids, dtype=numpy.uint16).tofile(path)
        n_bytes = len(text.encode("utf-8"))
        splits[name] = {"tokens": len(ids), "bytes": n_bytes, "bytes_per_token": n_bytes / len(ids), "sha256": sha256(path)}
        print(f"  {name:5} {len(ids):9,} tokens  {n_bytes / len(ids):.2f} bytes/token  round trip exact")

    table = bpe.token_bytes()
    example_ids = bpe.encode(EXAMPLE)
    example = [table[i].decode("utf-8", errors="replace") for i in example_ids]
    print(f"\nexample ({len(EXAMPLE.encode())} bytes -> {len(example_ids)} tokens):\n  {example}")

    unseen_chunks = Counter(c for c in pre_tokenize(texts["val"]) if len(bpe.encode(c)) > 1)
    results.update({
        "vocab_size": bpe.vocab_size,
        "tokenizer_sha256": sha256(tok_path),
        "splits": splits,
        "example": {"text": EXAMPLE, "tokens": example},
        "longest_tokens": sorted({t.decode("utf-8", errors="replace") for t in table[256:]}, key=len)[-10:],
        "val_chunks_split_in_several_tokens": sum(unseen_chunks.values()) / len(pre_tokenize(texts["val"])),
    })
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
