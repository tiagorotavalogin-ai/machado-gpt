"""Train the GPT on BPE tokens (stage 5), on the GPU when there is one.

Same hand-written loop as stage 3, plus what a longer GPU run needs:

- data check: the encoded splits and the tokenizer must have the SHA-256
  recorded in results/04-bpe.json, so the run uses exactly the stage 4 data;
- mixed precision (float16 on the GPU), with a gradient scaler;
- exact loss over the whole validation split at every evaluation;
- early stopping: the best checkpoint (by validation) is kept, and training stops
  after `patience` evaluations without improvement;
- resumable: `last.pt` holds the model, optimizer, scaler and random states, so
  a dropped Colab session continues where it stopped, with the same result;
- final report in bits per byte on validation and test, comparable to stage 3.

Usage (see notebooks/05-colab.ipynb):
    python -m machado_gpt.train_bpe --size small --out <checkpoint dir>
    python -m machado_gpt.train_bpe --size small --out <checkpoint dir> --resume
    python -m machado_gpt.train_bpe --size small --steps 20   # quick smoke run
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy
import torch
from torch.nn import functional as F

from machado_gpt.bpe import BPE
from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_char import learning_rate

DATA_DIR = PROJECT_ROOT / "runs" / "04-bpe"  # stage 4 output (not versioned)
STAGE4_RESULTS = PROJECT_ROOT / "results" / "04-bpe.json"
RESULTS_DIR = PROJECT_ROOT / "results"
VOCAB_SIZE = 4096

PRESETS = {
    # ~4.3 M parameters
    "small": GPTConfig(vocab_size=VOCAB_SIZE, block_size=256, n_layer=4, n_head=4, n_embd=256,
                       dropout=0.2, tie_weights=True),
    # ~12 M parameters
    "medium": GPTConfig(vocab_size=VOCAB_SIZE, block_size=256, n_layer=6, n_head=6, n_embd=384,
                        dropout=0.2, tie_weights=True),
}
# Larger models usually need a smaller learning rate to stay stable.
LEARNING_RATES = {"small": 1e-3, "medium": 6e-4}
PROMPTS = ("Capitu", "Não consultes dicionários.")


@dataclass
class TrainConfig:
    steps: int = 5000  # maximum; early stopping usually ends the run before
    batch_size: int = 64  # 64 windows x 256 tokens = 16,384 tokens per step
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    train_eval_batches: int = 24  # random batches for the training-loss estimate
    patience: int = 10  # evaluations without improvement before stopping
    save_last_every: int = 500  # steps between resumable checkpoints
    seed: int = 1337


# ---- data -----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    """The code version that produced a result ("unknown" outside a git checkout)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def verify_stage4(data_dir: Path, results_path: Path) -> dict[str, str]:
    """Refuse to train unless tokenizer and splits match results/04-bpe.json."""
    expected = json.loads(results_path.read_text(encoding="utf-8"))
    files = {"tokenizer": (data_dir / f"bpe-{expected['vocab_size']}.json", expected["tokenizer_sha256"])}
    for name, info in expected["splits"].items():
        files[name] = (data_dir / f"{name}.bin", info["sha256"])
    checked = {}
    for name, (path, sha) in files.items():
        actual = sha256_file(path)
        if actual != sha:
            raise ValueError(f"{name}: SHA-256 differs from {results_path.name}; regenerate stage 4")
        checked[name] = actual
    return checked


def load_tokens(path: Path, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(numpy.fromfile(path, dtype=numpy.uint16).astype(numpy.int64)).to(device)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    """B random windows: x = data[i : i+T], y = data[i+1 : i+T+1] (the next token)."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    chunk = data[(starts[:, None] + torch.arange(block_size + 1)).to(data.device)]
    return chunk[:, :-1], chunk[:, 1:]


# ---- evaluation -----------------------------------------------------------------


def autocast(device: torch.device):
    """float16 on the GPU: about twice as fast and half the memory. Softmax and the
    loss are computed in float32 by autocast, where float16 would lose precision."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def nll_sum(model: GPT, data: torch.Tensor, batch_size: int = 64) -> tuple[float, int]:
    """Exact sum of -ln P over a whole split, in non-overlapping windows.
    Returns (sum in nats, number of predicted tokens): tokens data[1 : n + 1]."""
    was_training = model.training
    model.eval()
    T = model.config.block_size
    n_windows = (len(data) - 1) // T
    offsets = torch.arange(T + 1, device=data.device)
    total, count = 0.0, 0
    for start in range(0, n_windows, batch_size):
        starts = torch.arange(start, min(start + batch_size, n_windows), device=data.device) * T
        chunk = data[starts[:, None] + offsets]
        with autocast(data.device):
            logits, _ = model(chunk[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1),
                                 reduction="sum").item()
        count += chunk[:, 1:].numel()
    model.train(was_training)
    return total, count


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.train_eval_batches):
        x, y = get_batch(data, model.config.block_size, cfg.batch_size, gen)
        with autocast(data.device):
            losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def report(model: GPT, data: torch.Tensor, token_lengths: torch.Tensor) -> dict[str, float]:
    """Loss per token and bits per byte, the measure that compares tokenizers."""
    total, n = nll_sum(model, data)
    n_bytes = int(token_lengths[data[1 : n + 1]].sum())
    return {
        "nats_per_token": total / n,
        "bits_per_byte": total / math.log(2) / n_bytes,
        "perplexity_per_token": math.exp(total / n),
        "tokens": n,
        "bytes": n_bytes,
    }


# ---- checkpoints ----------------------------------------------------------------


def save(obj: dict, path: Path) -> None:
    """Write to a temporary file, then rename: a session that drops mid-write
    never leaves a truncated checkpoint behind."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_states(gen: torch.Generator) -> dict[str, torch.Tensor]:
    states = {"batches": gen.get_state(), "torch": torch.get_rng_state()}  # torch: dropout
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state()
    return states


def restore_rng(states: dict[str, torch.Tensor], gen: torch.Generator) -> None:
    gen.set_state(states["batches"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state(states["cuda"])


# ---- training -------------------------------------------------------------------


def train(model_cfg: GPTConfig, cfg: TrainConfig, train_data: torch.Tensor, val_data: torch.Tensor,
          out_dir: Path, resume: bool = False) -> dict:
    """Runs the loop; returns the history and the reason it stopped. best.pt holds
    the best model by validation loss."""
    device = train_data.device
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = GPT(model_cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]  # a tied matrix appears once
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, 0.99),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    step, history, best_val, bad_evals, elapsed = 0, [], float("inf"), 0, 0.0
    skip_eval = False
    if resume and (out_dir / "last.pt").exists():
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled():  # a disabled scaler (CPU) has no state
            scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"], gen)
        step, history, best_val = state["step"], state["history"], state["best_val"]
        bad_evals, elapsed = state["bad_evals"], state["minutes"]
        skip_eval = True  # the evaluation of this step was done before saving
        print(f"resumed at step {step} (best val {best_val:.4f})")

    started = time.time() - 60 * elapsed
    stopped = "max steps"
    while True:
        if (step % cfg.eval_every == 0 or step == cfg.steps) and not skip_eval:
            val_total, val_n = nll_sum(model, val_data)
            point = {
                "step": step,
                "train": estimate_loss(model, train_data, cfg, gen),
                "val": val_total / val_n,  # exact, whole validation split
                "minutes": (time.time() - started) / 60,
            }
            history.append(point)
            improved = point["val"] < best_val
            if improved:
                best_val, bad_evals = point["val"], 0
                save({"model": model.state_dict(), "config": asdict(model_cfg), "step": step,
                      "val": best_val}, out_dir / "best.pt")
            else:
                bad_evals += 1
            print(f"step {step:5}  train {point['train']:.4f}  val {point['val']:.4f}"
                  f"{'  *best' if improved else ''}  lr {learning_rate(step, cfg):.2e}  "
                  f"{point['minutes']:.1f} min", flush=True)
            if step == cfg.steps:
                break
            if bad_evals >= cfg.patience:
                stopped = f"early stopping: {cfg.patience} evaluations without improvement"
                break
            if step % cfg.save_last_every == 0 and step > 0:
                save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "rng": rng_states(gen), "step": step,
                      "history": history, "best_val": best_val, "bad_evals": bad_evals,
                      "minutes": (time.time() - started) / 60}, out_dir / "last.pt")
        skip_eval = False

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        x, y = get_batch(train_data, model_cfg.block_size, cfg.batch_size, gen)
        with autocast(device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()  # scaled so small float16 gradients do not round to zero
        scaler.unscale_(optimizer)  # back to true scale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

    return {"history": history, "stopped": stopped, "minutes": (time.time() - started) / 60}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(PRESETS), required=True)
    parser.add_argument("--out", type=Path, help="checkpoint directory (default: runs/05-gpt-bpe-<size>)")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    lr = LEARNING_RATES[args.size]
    cfg = TrainConfig(steps=args.steps, learning_rate=lr, min_learning_rate=lr / 10)
    model_cfg = PRESETS[args.size]
    out_dir = args.out or PROJECT_ROOT / "runs" / f"05-gpt-bpe-{args.size}"

    data_sha = verify_stage4(DATA_DIR, STAGE4_RESULTS)
    print(f"stage 4 data verified by SHA-256 | device: {device_name}")
    tokenizer = BPE.load(DATA_DIR / f"bpe-{VOCAB_SIZE}.json")
    token_lengths = torch.tensor([len(t) for t in tokenizer.token_bytes()], device=device)
    splits = {name: load_tokens(DATA_DIR / f"{name}.bin", device) for name in ("train", "val", "test")}
    params = GPT(model_cfg).num_params()
    print(f"model {args.size}: {params:,} parameters | {model_cfg}")
    print(f"train {len(splits['train']):,} tokens | {cfg}")

    run = train(model_cfg, cfg, splits["train"], splits["val"], out_dir, resume=args.resume)

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model = GPT(model_cfg).to(device)
    model.load_state_dict(best["model"])
    model.eval()
    # The model is chosen by validation only; test is reported once, for the record.
    val, test = report(model, splits["val"], token_lengths), report(model, splits["test"], token_lengths)
    print(f"\n{run['stopped']} | best step {best['step']}")
    print(f"validation: {val['bits_per_byte']:.4f} bits/byte  ({val['nats_per_token']:.4f} nats/token)")
    print(f"test:       {test['bits_per_byte']:.4f} bits/byte")
    print("reference (validation): bigram 3.33, character GPT 2.22 bits/byte")

    samples = {}
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        gen = torch.Generator(device=device).manual_seed(TrainConfig.seed)
        samples[prompt] = tokenizer.decode(model.generate(ids, 200, generator=gen)[0].tolist())
        print(f"\nsample from {prompt!r}:\n{samples[prompt]}")

    results = {
        "size": args.size,
        "model": asdict(model_cfg),
        "parameters": params,
        "train": asdict(cfg),
        "git_commit": git_commit(),
        "device": device_name,
        "torch": torch.__version__,
        "data_sha256": data_sha,
        "stopped": run["stopped"],
        "minutes": run["minutes"],
        "best_step": best["step"],
        "history": run["history"],
        "val": val,
        "test": test,
        "checkpoint": "best.pt",  # name only: the directory is machine-specific
        "checkpoint_sha256": sha256_file(out_dir / "best.pt"),
        "samples": samples,
    }
    text = json.dumps(results, ensure_ascii=False, indent=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"05-gpt-bpe-{args.size}.json").write_text(text, encoding="utf-8")
    (out_dir / f"05-gpt-bpe-{args.size}.json").write_text(text, encoding="utf-8")  # a copy beside the checkpoint
    print(f"\nresults -> results/05-gpt-bpe-{args.size}.json (and a copy in the checkpoint directory)")


if __name__ == "__main__":
    main()
