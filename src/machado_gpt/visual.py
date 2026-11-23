"""Build the visual explainer page docs/visual/machado-gpt.html.

Collects the real data of each stage (catalog, bigram counts, attention weights
of the stage 3 model, BPE merges, stage 5 training curves) and inlines it into
docs/visual/template.html. Needs the local corpus, runs/03-gpt-char/model.pt and
runs/04-bpe/bpe-4096.json.

Usage:
    uv run python -m machado_gpt.visual
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch

from machado_gpt.bigram import load_split
from machado_gpt.bpe import BPE
from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.inspect_attention import head_profiles, load_model

TEMPLATE = PROJECT_ROOT / "docs" / "visual" / "template.html"
PAGE = PROJECT_ROOT / "docs" / "visual" / "machado-gpt.html"


def collect() -> dict:
    out = {}
    # works
    works, excluded = [], Counter()
    for r in csv.DictReader(open(PROJECT_ROOT / "data/catalog/works.csv", encoding="utf-8")):
        name = r["obra"].split("/")[1]
        title = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name)
        title = title[0].upper() + title[1:]
        if r["situacao"] == "incluída":
            works.append({"t": title, "g": r["genero"], "s": r["conjunto"], "w": int(r["palavras"])})
        else:
            excluded["lista" if "lista" in r["situacao"] else "duplicata"] += 1
    out["works"] = works
    out["excluded"] = excluded

    # bigram counts over top chars
    train = load_split("train")
    freq = Counter(train)
    chars = [c for c, _ in freq.most_common(72)]
    idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    counts = [[0] * n for _ in range(n)]
    for a, b in zip(train, train[1:]):
        if a in idx and b in idx:
            counts[idx[a]][idx[b]] += 1
    out["bigram"] = {"chars": "".join(chars), "counts": counts, "coverage": sum(freq[c] for c in chars) / len(train)}
    r2 = json.load(open(PROJECT_ROOT / "results/02-bigram.json"))
    out["stage2"] = {"uniform": r2["uniform_val"]["bits_per_byte"], "unigram": r2["unigram_val"]["bits_per_byte"],
                     "bigram": r2["bigram_count"]["val"]["bits_per_byte"]}

    # stage 3
    r3 = json.load(open(PROJECT_ROOT / "results/03-gpt-char.json"))
    model, vocab = load_model()
    val = load_split("val")
    m = re.search(r"\n([A-ZÁÉÍÓÚ][^\n]{70,110}?[.!?])\n", val)
    sentence = m.group(1)
    ids = torch.tensor([vocab.encode(sentence)])
    model(ids)
    att = []
    for block in model.blocks:
        w = block.attn.last_weights[0]  # heads, T, T
        heads = []
        for h in range(w.shape[0]):
            rows = []
            for t in range(w.shape[1]):
                row = w[h, t, : t + 1]
                mx = float(row.max())
                rows.append("".join(str(min(9, round(9 * float(x) / mx))) for x in row))
            heads.append(rows)
        att.append(heads)
    T = model.config.block_size
    data = torch.tensor(vocab.encode(val[: 64 * T + 1]))
    prof = head_profiles(model, data[: 64 * T].view(64, T), vocab.encode(" ")[0])
    out["stage3"] = {"history": r3["history"], "val": r3["val_full"], "params": r3["parameters"],
                     "samples": {k: v[:420] for k, v in r3["samples"].items()},
                     "sentence": sentence, "attention": att,
                     "profiles": [[{k: round(v, 3) for k, v in h.items()} for h in layer] for layer in prof]}

    # stage 4
    r4 = json.load(open(PROJECT_ROOT / "results/04-bpe.json"))
    bpe = BPE.load(PROJECT_ROOT / "runs/04-bpe/bpe-4096.json")
    out["stage4"] = {"merges": bpe.merges, "sweep": r4["sweep"], "splits": r4["splits"], "example": r4["example"]}

    # stage 5
    out["stage5"] = {}
    for s in ("small", "medium"):
        r = json.load(open(PROJECT_ROOT / f"results/05-gpt-bpe-{s}.json"))
        out["stage5"][s] = {"history": [{k: round(p[k], 4) for k in ("step", "train", "val")} for p in r["history"]],
                            "val": r["val"]["bits_per_byte"], "test": r["test"]["bits_per_byte"],
                            "params": r["parameters"], "minutes": round(r["minutes"], 1), "best": r["best_step"],
                            "samples": {k: v[:600] for k, v in r["samples"].items()}}
    return out


def main() -> None:
    data = json.dumps(collect(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    PAGE.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")
    print(f"page -> {PAGE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
