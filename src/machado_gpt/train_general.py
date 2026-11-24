"""Pretrain on the general corpus, then fine-tune on Machado (stage 6).

Only one thing changes from stage 5: the model starts pretrained. Model shape,
tokenizer, Machado validation and the measure (bits per byte) are the same.

- pretrain: one exact pass over the general training split (shuffled
  non-overlapping windows, none repeated), dropout 0. The general validation
  split picks best.pt; the Machado validation split is only recorded, as a
  curve: it has no say in anything here.
- finetune: from the pretrained best.pt (checked by its SHA-256), a fresh
  optimizer, a third of the learning rate, dropout 0.2 and early stopping on
  the Machado validation split, as in stage 5. The general validation split is
  recorded to see how much general Portuguese is forgotten. Test is reported
  once, at the end.

Usage (see notebooks/06-colab.ipynb):
    python -m machado_gpt.train_general --phase pretrain --out <dir>
    python -m machado_gpt.train_general --phase finetune --init <dir>/best.pt --out <dir2>
    python -m machado_gpt.train_general --phase pretrain --steps 20   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from machado_gpt import train_bpe
from machado_gpt.bpe import BPE
from machado_gpt.corpus.download import PROJECT_ROOT
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_bpe import (PRESETS, PROMPTS, TrainConfig, git_commit, load_tokens, report,
                                   sha256_file, train, verify_stage4)

GENERAL_DIR = PROJECT_ROOT / "runs" / "06-general"  # tokenize_general output (not versioned)
GENERAL_RESULTS = PROJECT_ROOT / "results" / "06-wikisource.json"
RESULTS_DIR = PROJECT_ROOT / "results"

PRETRAIN_LR = {"small": 1e-3, "medium": 6e-4}  # stage 5's, per size
FINETUNE_LR_FACTOR = 1 / 3
FINETUNE_STEPS = 3000
PRETRAIN_EPOCHS = 1


# ---- data -----------------------------------------------------------------------


def verify_general(general_dir: Path, results_path: Path, tokenizer: Path) -> dict[str, str]:
    """Refuse to train unless the general splits, and the tokenizer that encoded
    them, match results/06-wikisource.json."""
    expected = json.loads(results_path.read_text(encoding="utf-8"))["tokens"]
    files = {"tokenizer": (tokenizer, expected["tokenizer_sha256"])}
    for name, info in expected["splits"].items():
        files[f"general_{name}"] = (general_dir / f"{name}.bin", info["sha256"])
    checked = {}
    for name, (path, sha) in files.items():
        actual = sha256_file(path)
        if actual != sha:
            raise ValueError(f"{name}: SHA-256 differs from {results_path.name}; regenerate the general corpus")
        checked[name] = actual
    return checked


class WindowSampler:
    """One exact pass: the split is cut into non-overlapping windows of T + 1
    tokens (x and its next-token targets y), shuffled, and served B at a time;
    no window repeats until all were served, then a new shuffle starts.

    Random windows, as in stage 5, are drawn with replacement: as many draws as
    windows leave about 1/e (37%) of the text unread and repeat other parts."""

    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, seed: int):
        self.data, self.T, self.B, self.seed = data, block_size, batch_size, seed
        self.n_windows = (len(data) - 1) // block_size
        self.epoch, self.pos = 0, 0
        self._order = self._shuffle()

    def _shuffle(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n_windows, generator=gen)

    @property
    def steps_per_epoch(self) -> int:
        return self.n_windows // self.B  # the remainder of an epoch is skipped

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.B > self.steps_per_epoch * self.B:
            self.epoch, self.pos = self.epoch + 1, 0
            self._order = self._shuffle()
        starts = self._order[self.pos : self.pos + self.B].to(self.data.device) * self.T
        self.pos += self.B
        chunk = self.data[starts[:, None] + torch.arange(self.T + 1, device=self.data.device)]
        return chunk[:, :-1], chunk[:, 1:]

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state(self, state: dict[str, int]) -> None:
        self.epoch, self.pos = state["epoch"], state["pos"]
        self._order = self._shuffle()


# ---- phases ---------------------------------------------------------------------


def pretrain_config(size: str, steps: int) -> tuple[GPTConfig, TrainConfig]:
    lr = PRETRAIN_LR[size]
    model_cfg = replace(PRESETS[size], dropout=0.0)  # one pass: nothing to overfit
    cfg = TrainConfig(steps=steps, learning_rate=lr, min_learning_rate=lr / 10,
                      patience=10**9)  # no early stopping: one pass, then stop
    return model_cfg, cfg


def finetune_config(size: str, steps: int) -> tuple[GPTConfig, TrainConfig]:
    lr = PRETRAIN_LR[size] * FINETUNE_LR_FACTOR
    model_cfg = PRESETS[size]  # dropout 0.2, as in stage 5: Machado repeats
    cfg = TrainConfig(steps=steps, learning_rate=lr, min_learning_rate=lr / 10, warmup_steps=100)
    return model_cfg, cfg


def samples(model: GPT, tokenizer: BPE, device: torch.device) -> dict[str, str]:
    out = {}
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        gen = torch.Generator(device=device).manual_seed(TrainConfig.seed)
        out[prompt] = tokenizer.decode(model.generate(ids, 200, generator=gen)[0].tolist())
        print(f"\nsample from {prompt!r}:\n{out[prompt]}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pretrain", "finetune"), required=True)
    parser.add_argument("--size", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--out", type=Path, help="checkpoint directory (default: runs/06-<phase>-<size>)")
    parser.add_argument("--init", type=Path, help="finetune: the pretrained best.pt")
    parser.add_argument("--steps", type=int, help="override the number of steps (smoke runs)")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    name = f"06-{args.phase}-{args.size}"
    out_dir = args.out or PROJECT_ROOT / "runs" / name
    tokenizer_path = train_bpe.DATA_DIR / f"bpe-{train_bpe.VOCAB_SIZE}.json"

    data_sha = verify_stage4(train_bpe.DATA_DIR, train_bpe.STAGE4_RESULTS)
    data_sha |= verify_general(GENERAL_DIR, GENERAL_RESULTS, tokenizer_path)
    print(f"stage 4 and general data verified by SHA-256 | device: {device_name}")
    tokenizer = BPE.load(tokenizer_path)
    token_lengths = torch.tensor([len(t) for t in tokenizer.token_bytes()], device=device)
    machado = {s: load_tokens(train_bpe.DATA_DIR / f"{s}.bin", device) for s in ("train", "val", "test")}
    general = {s: load_tokens(GENERAL_DIR / f"{s}.bin", device) for s in ("train", "val")}

    extra = {}
    if args.phase == "pretrain":
        block = PRESETS[args.size].block_size
        sampler = WindowSampler(general["train"], block, TrainConfig.batch_size, TrainConfig.seed)
        model_cfg, cfg = pretrain_config(args.size, args.steps or PRETRAIN_EPOCHS * sampler.steps_per_epoch)
        run = train(model_cfg, cfg, general["train"], general["val"], out_dir, resume=args.resume,
                    extra_eval={"machado_val": machado["val"]}, sampler=sampler)
    else:
        init = args.init or PROJECT_ROOT / "runs" / f"06-pretrain-{args.size}" / "best.pt"
        pretrain = json.loads((RESULTS_DIR / f"06-pretrain-{args.size}.json").read_text(encoding="utf-8"))
        extra["init_sha256"] = sha256_file(init)
        if extra["init_sha256"] != pretrain["checkpoint_sha256"]:
            raise ValueError(f"{init.name}: SHA-256 differs from results/06-pretrain-{args.size}.json")
        init_state = torch.load(init, map_location=device, weights_only=True)["model"]
        model_cfg, cfg = finetune_config(args.size, args.steps or FINETUNE_STEPS)
        run = train(model_cfg, cfg, machado["train"], machado["val"], out_dir, resume=args.resume,
                    init_state=init_state, extra_eval={"general_val": general["val"]})
    params = GPT(model_cfg).num_params()

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model = GPT(model_cfg).to(device)
    model.load_state_dict(best["model"])
    model.eval()
    print(f"\n{run['stopped']} | best step {best['step']}")
    evaluation = {"machado_val": report(model, machado["val"], token_lengths),
                  "general_val": report(model, general["val"], token_lengths)}
    if args.phase == "finetune":
        # The model is chosen by Machado validation only; test is reported once, for the record.
        evaluation["machado_test"] = report(model, machado["test"], token_lengths)
    for split, r in evaluation.items():
        print(f"{split:13} {r['bits_per_byte']:.4f} bits/byte  ({r['nats_per_token']:.4f} nats/token)")
    print("reference (Machado validation): stage 5 medium 1.663 bits/byte")

    results = {
        "phase": args.phase,
        "size": args.size,
        "model": asdict(model_cfg),
        "parameters": params,
        "train": asdict(cfg),
        "git_commit": git_commit(),
        "device": device_name,
        "torch": torch.__version__,
        "data_sha256": data_sha,
        **extra,
        "stopped": run["stopped"],
        "minutes": run["minutes"],
        "best_step": best["step"],
        "history": run["history"],
        "eval": evaluation,
        "checkpoint": "best.pt",  # name only: the directory is machine-specific
        "checkpoint_sha256": sha256_file(out_dir / "best.pt"),
        "samples": samples(model, tokenizer, device),
    }
    text = json.dumps(results, ensure_ascii=False, indent=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(text, encoding="utf-8")
    (out_dir / f"{name}.json").write_text(text, encoding="utf-8")  # a copy beside the checkpoint
    print(f"\nresults -> results/{name}.json (and a copy in the checkpoint directory)")


if __name__ == "__main__":
    main()
