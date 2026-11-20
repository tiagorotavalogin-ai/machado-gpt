"""Byte-level BPE (Byte Pair Encoding) tokenizer, written by hand.

Training:
1. Split the text into chunks (pre-tokenization): words with their leading
   space, numbers, punctuation runs, whitespace. Merges never cross chunks.
2. Turn each chunk into its UTF-8 bytes: the initial vocabulary is the 256 byte
   values, so any text can be represented (there is no "unknown" token).
3. Repeatedly merge the most frequent pair of adjacent symbols into a new
   symbol, until the vocabulary has the requested size. The ordered list of
   merges is the tokenizer.

Encoding applies the merges to a chunk in the order they were learned; decoding
concatenates the bytes of each token and decodes UTF-8.

Pair counts are updated incrementally: after a merge, only the chunks that
contained the pair are recounted, which keeps training to seconds.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Pre-tokenization pattern (in the spirit of GPT-2's, without English contractions):
# - optional space + letters ("Capitu", " que", " ação")
# - optional space + 1 to 3 digits ("1", " 189", "9")
# - optional space + run of other symbols (" —", ".", "”.", "_"); the underscore is
#   a "word" character for `re` but neither letter nor digit, so it is listed
#   explicitly (without it, "_" fell through every branch and was lost)
# - whitespace, keeping the last space attached to the next word
PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pre_tokenize(text: str) -> list[str]:
    return PATTERN.findall(text)


Pair = tuple[int, int]


@dataclass
class BPE:
    merges: list[Pair] = field(default_factory=list)  # merge i creates token 256 + i

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def token_bytes(self) -> list[bytes]:
        """The bytes each token stands for."""
        table = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            table.append(table[a] + table[b])
        return table

    # ---- training -------------------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose_every: int = 0) -> BPE:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte values)")
        chunk_counts = Counter(pre_tokenize(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: Counter[Pair] = Counter()
        where: dict[Pair, set[int]] = defaultdict(set)  # pair -> words containing it
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)

        merges: list[Pair] = []
        while 256 + len(merges) < vocab_size and pair_counts:
            # most frequent pair; ties broken by the smallest pair, so training is deterministic
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for i in list(where[best]):
                w, f = words[i], freqs[i]
                for pair in zip(w, w[1:]):  # remove the word's old pairs
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(i)
                words[i] = w = merge_pair(w, best, new_id)
                for pair in zip(w, w[1:]):  # add its new pairs
                    pair_counts[pair] += f
                    where[pair].add(i)
            del where[best]
            if verbose_every and len(merges) % verbose_every == 0:
                print(f"  merge {len(merges):5}: {cls(merges).token_bytes()[new_id]!r}")
        return cls(merges)

    # ---- encoding / decoding --------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ranks = {pair: i for i, pair in enumerate(self.merges)}
        cache: dict[str, list[int]] = {}
        ids: list[int] = []
        for chunk in pre_tokenize(text):
            if chunk not in cache:
                cache[chunk] = encode_chunk(list(chunk.encode("utf-8")), ranks)
            ids.extend(cache[chunk])
        return ids

    def decode(self, ids: list[int]) -> str:
        table = self.token_bytes()
        return b"".join(table[i] for i in ids).decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------------

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPE:
        return cls([tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))["merges"]])


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair by new_id, left to right."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def encode_chunk(ids: list[int], ranks: dict[Pair, int]) -> list[int]:
    """Apply merges in the order they were learned (lowest rank first)."""
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, float("inf")))
        if pair not in ranks:
            break
        ids = merge_pair(ids, pair, 256 + ranks[pair])
    return ids
