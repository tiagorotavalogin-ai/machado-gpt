"""Bigram character model: predict the next character from the current one.

Two versions of the same model:
- counting: P(b | a) = (count(a, b) + alpha) / (count(a) + alpha * V);
- neural: a V x V table of logits trained by gradient descent (see `train_neural`).

Usage:
    uv run python -m machado_gpt.bigram
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy
import torch

from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.vocab import CharVocab

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results" / "02-bigram.json"
RUN_DIR = PROJECT_ROOT / "runs" / "02-bigram"  # not versioned
ALPHAS = (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
SEED = 1337
NEURAL_STEPS = 3000
NEURAL_BATCH = 32768
NEURAL_LR = 50.0  # large because W starts flat and each step sees a small batch
NEURAL_EVAL_EVERY = 300


@dataclass
class Score:
    """Average negative log-likelihood of a text, in several units."""

    nats_per_char: float
    bits_per_char: float
    bits_per_byte: float
    perplexity: float


def score(nll_sum_nats: float, n_chars: int, n_bytes: int) -> Score:
    nats = nll_sum_nats / n_chars
    return Score(
        nats_per_char=nats,
        bits_per_char=nats / math.log(2),
        bits_per_byte=nll_sum_nats / math.log(2) / n_bytes,
        perplexity=math.exp(nats),
    )


def count_bigrams(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """counts[a, b] = how many times character b follows character a."""
    pairs = ids[:-1] * vocab_size + ids[1:]
    return torch.bincount(pairs, minlength=vocab_size * vocab_size).view(vocab_size, vocab_size)


def bigram_probs(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add-alpha smoothing: every pair gets alpha extra counts, so none has probability 0."""
    smoothed = counts.double() + alpha
    return smoothed / smoothed.sum(dim=1, keepdim=True)


def bigram_nll_sum(probs: torch.Tensor, ids: torch.Tensor) -> float:
    """Sum of -ln P(next | current) over every position of a text."""
    return float(-torch.log(probs[ids[:-1], ids[1:]]).sum())


def unigram_nll_sum(train_ids: torch.Tensor, ids: torch.Tensor, vocab_size: int, alpha: float = 1.0) -> float:
    """Baseline that ignores context: P(next) is just the frequency of the character."""
    freq = torch.bincount(train_ids, minlength=vocab_size).double() + alpha
    probs = freq / freq.sum()
    return float(-torch.log(probs[ids[1:]]).sum())


def sample(probs: torch.Tensor, vocab: CharVocab, start: str, n: int, seed: int = SEED) -> str:
    """Generate text by drawing each next character from P(. | current)."""
    gen = torch.Generator().manual_seed(seed)
    ids = vocab.encode(start)
    for _ in range(n):
        ids.append(int(torch.multinomial(probs[ids[-1]].float(), 1, generator=gen)))
    return vocab.decode(ids)


def table_nll_sum(logits: torch.Tensor, counts: torch.Tensor) -> float:
    """Exact loss of a logit table on a text, from the text's pair counts.

    The loss only depends on how often each pair (a, b) occurs, so it is
    sum(counts * -log_softmax(logits)), without running every position.
    """
    return float(-(counts.double() * torch.log_softmax(logits.double(), dim=1)).sum())


@dataclass
class NeuralRun:
    steps: int
    batch_size: int
    learning_rate: float
    history: list[dict[str, float]]


def train_neural(
    train_ids: torch.Tensor,
    vocab_size: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_every: int,
    eval_counts: dict[str, torch.Tensor],
    seed: int = SEED,
) -> tuple[torch.Tensor, NeuralRun]:
    """Learn the bigram table by gradient descent on mini-batches of positions.

    W starts at zero (every next character equally likely). Each step takes a
    random batch of (current, next) pairs, computes the cross-entropy of
    softmax(W[current]) against next, and moves W against the gradient.
    """
    gen = torch.Generator().manual_seed(seed)
    W = torch.zeros(vocab_size, vocab_size, requires_grad=True)
    x_all, y_all = train_ids[:-1], train_ids[1:]
    history = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            with torch.no_grad():
                point = {"step": step}
                for name, counts in eval_counts.items():
                    point[name] = table_nll_sum(W, counts) / float(counts.sum())
                history.append(point)
        if step == steps:
            break
        idx = torch.randint(len(x_all), (batch_size,), generator=gen)
        logits = W[x_all[idx]]  # row lookup = one-hot(x) @ W
        loss = torch.nn.functional.cross_entropy(logits, y_all[idx])
        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= learning_rate * W.grad
    return W.detach(), NeuralRun(steps, batch_size, learning_rate, history)


def load_split(name: str) -> str:
    return (PROCESSED_DIR / f"{name}.txt").read_text(encoding="utf-8")


def main(argv: list[str]) -> None:
    train_text, val_text = load_split("train"), load_split("val")
    vocab = CharVocab.from_text(train_text)
    V = vocab.size
    train = torch.tensor(vocab.encode(train_text))
    val = torch.tensor(vocab.encode(val_text))
    n_val, bytes_val = len(val) - 1, len(val_text[1:].encode("utf-8"))
    print(f"vocabulary: {V} characters (from train only)")
    print(f"train: {len(train):,} characters | val: {len(val):,} characters")

    results: dict[str, object] = {"vocab_size": V}
    uniform = score(n_val * math.log(V), n_val, bytes_val)
    unigram = score(unigram_nll_sum(train, val, V), n_val, bytes_val)
    print(f"\nuniform  val: {uniform.nats_per_char:.4f} nats/char  {uniform.bits_per_byte:.4f} bits/byte  ppl {uniform.perplexity:.1f}")
    print(f"unigram  val: {unigram.nats_per_char:.4f} nats/char  {unigram.bits_per_byte:.4f} bits/byte  ppl {unigram.perplexity:.1f}")
    results["uniform_val"] = asdict(uniform)
    results["unigram_val"] = asdict(unigram)

    counts = count_bigrams(train, V)
    zeros = int((counts == 0).sum())
    print(f"\nbigram table: {V}x{V} = {V * V:,} pairs, {zeros:,} never seen in train ({100 * zeros / V / V:.1f}%)")
    by_alpha = {}
    for alpha in ALPHAS:
        s = score(bigram_nll_sum(bigram_probs(counts, alpha), val), n_val, bytes_val)
        by_alpha[alpha] = s
        print(f"  alpha={alpha:<6} val: {s.nats_per_char:.4f} nats/char  {s.bits_per_byte:.4f} bits/byte  ppl {s.perplexity:.2f}")
    best_alpha = min(by_alpha, key=lambda a: by_alpha[a].nats_per_char)
    probs = bigram_probs(counts, best_alpha)
    train_score = score(bigram_nll_sum(probs, train), len(train) - 1, len(train_text[1:].encode("utf-8")))
    print(f"best alpha={best_alpha}: train {train_score.nats_per_char:.4f} | val {by_alpha[best_alpha].nats_per_char:.4f} nats/char")
    results["bigram_count"] = {
        "alpha": best_alpha,
        "train": asdict(train_score),
        "val": asdict(by_alpha[best_alpha]),
        "val_by_alpha": {str(a): s.nats_per_char for a, s in by_alpha.items()},
    }

    text = sample(probs, vocab, "Capitu", 400)
    print(f"\nsample (seed {SEED}):\n{text}")
    results["bigram_count"]["sample"] = text

    print("\nneural bigram (gradient descent on a 144x144 logit table):")
    val_counts = count_bigrams(val, V)
    W, run = train_neural(
        train, V, steps=NEURAL_STEPS, batch_size=NEURAL_BATCH, learning_rate=NEURAL_LR,
        eval_every=NEURAL_EVAL_EVERY, eval_counts={"train": counts, "val": val_counts},
    )
    for point in run.history:
        print(f"  step {point['step']:5}  train {point['train']:.4f}  val {point['val']:.4f} nats/char")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    numpy.save(RUN_DIR / "neural_logits.npy", W.numpy())
    numpy.save(RUN_DIR / "train_counts.npy", counts.numpy())
    numpy.save(RUN_DIR / "val_counts.npy", val_counts.numpy())
    neural_val = score(table_nll_sum(W, val_counts), n_val, bytes_val)
    gap = neural_val.nats_per_char - by_alpha[best_alpha].nats_per_char
    print(f"neural val {neural_val.nats_per_char:.4f} vs counting {by_alpha[best_alpha].nats_per_char:.4f} (difference {gap:+.4f})")
    neural_sample = sample(torch.softmax(W.double(), dim=1), vocab, "Capitu", 400)
    print(f"\nneural sample (seed {SEED}):\n{neural_sample}")
    results["bigram_neural"] = {
        **asdict(run),
        "val": asdict(neural_val),
        "difference_to_counting": gap,
        "sample": neural_sample,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
