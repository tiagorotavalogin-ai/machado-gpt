"""Train the character-level GPT on the training split.

The loop is written by hand: sample a batch of windows, forward, loss, backward,
clip the gradient, AdamW step with a warmup + cosine learning rate. Every
`eval_every` steps the loss is estimated on random windows of train and val;
at the end the exact loss over the whole validation split is computed, in the
same units as the bigram baseline.

Usage:
    uv run python -m machado_gpt.train_char            # full run (~30 min on this CPU)
    uv run python -m machado_gpt.train_char --steps 50 # quick smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from machado_gpt.bigram import load_split, score
from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.vocab import CharVocab

RUN_DIR = PROJECT_ROOT / "runs" / "03-gpt-char"  # not versioned: checkpoints
RESULTS = PROJECT_ROOT / "results" / "03-gpt-char.json"


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337
    threads: int = 11  # measured fastest on this machine (22 is slower)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    """B random windows: x = data[i : i+T], y = data[i+1 : i+T+1] (the next character)."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from learning_rate to min_learning_rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> float:
    model.eval()
    losses = [
        model(*get_batch(data, model.config.block_size, cfg.batch_size, gen))[1].item()
        for _ in range(cfg.eval_batches)
    ]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def full_nll_sum(model: GPT, data: torch.Tensor, batch_size: int = 64) -> tuple[float, int]:
    """Exact sum of -ln P over a whole text, in non-overlapping windows of block_size.

    The first character of each window is predicted from a short context, which
    slightly penalizes the model; the comparison with the bigram stays fair.
    Returns (sum in nats, number of predicted characters).
    """
    model.eval()
    T = model.config.block_size
    n_windows = (len(data) - 1) // T
    total, count = 0.0, 0
    for start in range(0, n_windows, batch_size):
        idx = range(start, min(start + batch_size, n_windows))
        x = torch.stack([data[i * T : i * T + T] for i in idx])
        y = torch.stack([data[i * T + 1 : i * T + T + 1] for i in idx])
        logits, _ = model(x)
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        count += y.numel()
    return total, count


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    args = parser.parse_args()
    cfg = TrainConfig(steps=args.steps)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    train_text, val_text = load_split("train"), load_split("val")
    vocab = CharVocab.from_text(train_text)
    train = torch.tensor(vocab.encode(train_text), dtype=torch.long)
    val = torch.tensor(vocab.encode(val_text), dtype=torch.long)
    model_cfg = GPTConfig(vocab_size=vocab.size)
    model = GPT(model_cfg)
    print(f"model: {model.num_params():,} parameters | {model_cfg}")
    print(f"train: {len(train):,} chars | val: {len(val):,} chars | {cfg}")

    # Weight decay only on matrices (not on biases, LayerNorm or embeddings' scale).
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.99),
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            point = {
                "step": step,
                "train": estimate_loss(model, train, cfg, gen),
                "val": estimate_loss(model, val, cfg, gen),
                "minutes": (time.time() - started) / 60,
            }
            history.append(point)
            print(f"step {step:5}  train {point['train']:.4f}  val {point['val']:.4f}  "
                  f"lr {learning_rate(step, cfg):.2e}  {point['minutes']:.1f} min", flush=True)
        if step == cfg.steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        x, y = get_batch(train, model_cfg.block_size, cfg.batch_size, gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    checkpoint = RUN_DIR / "model.pt"
    torch.save({"model": model.state_dict(), "config": asdict(model_cfg), "vocab": list(vocab.chars)}, checkpoint)

    nll, n = full_nll_sum(model, val)
    n_bytes = len(vocab.decode(val[1 : n + 1].tolist()).encode("utf-8"))
    val_score = score(nll, n, n_bytes)
    print(f"\nfull validation: {val_score.nats_per_char:.4f} nats/char  {val_score.bits_per_byte:.4f} bits/byte  "
          f"ppl {val_score.perplexity:.2f}  ({n:,} characters)")

    samples = {}
    for start in ("Capitu", "Não consultes dicionários."):
        ids = torch.tensor([vocab.encode(start)])
        out = model.generate(ids, 500, generator=torch.Generator().manual_seed(cfg.seed))
        samples[start] = vocab.decode(out[0].tolist())
        print(f"\nsample from {start!r}:\n{samples[start]}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": asdict(model_cfg),
        "parameters": model.num_params(),
        "train": asdict(cfg),
        "history": history,
        "val_full": asdict(val_score),
        "val_characters": n,
        "checkpoint_sha256": sha256_file(checkpoint),
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {RESULTS.relative_to(PROJECT_ROOT)} | checkpoint -> {checkpoint.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
