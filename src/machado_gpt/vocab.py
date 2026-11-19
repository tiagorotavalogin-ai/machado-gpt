"""Character vocabulary, built from the training split only.

Every distinct character of the training text gets an integer id. A character
never seen in training (for example "½", which appears once in validation) maps
to a single "unknown" symbol, so the model is never given information from
validation or test through its vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

UNKNOWN = "�"  # "�", absent from the corpus


@dataclass(frozen=True)
class CharVocab:
    chars: tuple[str, ...]  # id -> character; id 0 is UNKNOWN

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        if UNKNOWN in text:
            raise ValueError("training text already contains the unknown symbol")
        return cls((UNKNOWN, *sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        index = {c: i for i, c in enumerate(self.chars)}
        return [index.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(list(self.chars), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        return cls(tuple(json.loads(path.read_text(encoding="utf-8"))))
