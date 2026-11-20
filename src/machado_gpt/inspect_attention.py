"""Look inside the trained character GPT: where does each attention head look?

For windows of the validation text, it measures for every head:
- how far back the attention goes on average (weighted distance);
- the share of weight on the previous character, on the position itself and on
  the most recent space (the start of the current word).

Usage:
    uv run python -m machado_gpt.inspect_attention
"""

from __future__ import annotations

import torch

from machado_gpt.bigram import load_split
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_char import RUN_DIR
from machado_gpt.vocab import CharVocab


def load_model() -> tuple[GPT, CharVocab]:
    ckpt = torch.load(RUN_DIR / "model.pt", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), CharVocab(tuple(ckpt["vocab"]))


@torch.no_grad()
def head_profiles(model: GPT, ids: torch.Tensor, space_id: int) -> list[list[dict[str, float]]]:
    """ids: (B, T). Returns, per layer and head, averages over positions t >= 16."""
    model(ids)
    B, T = ids.shape
    pos = torch.arange(T)
    distance = (pos.view(T, 1) - pos.view(1, T)).clamp(min=0).float()  # t - j
    # index of the most recent space at or before each position (per sequence)
    last_space = torch.full((B, T), -1)
    for b in range(B):
        current = -1
        for t in range(T):
            if ids[b, t] == space_id:
                current = t
            last_space[b, t] = current
    profiles = []
    for block in model.blocks:
        w = block.attn.last_weights  # (B, heads, T, T)
        rows = []
        for h in range(w.shape[1]):
            wh = w[:, h, 16:, :]  # skip the first positions, which have little context
            t_idx = torch.arange(16, T)
            prev = wh[:, torch.arange(len(t_idx)), t_idx - 1].mean().item()
            self_w = wh[:, torch.arange(len(t_idx)), t_idx].mean().item()
            ls = last_space[:, 16:]
            space = torch.stack([
                wh[b, torch.arange(len(t_idx)), ls[b].clamp(min=0)] * (ls[b] >= 0) for b in range(B)
            ]).mean().item()
            dist = (wh * distance[16:].unsqueeze(0)).sum(-1).mean().item()
            rows.append({"distance": dist, "previous": prev, "self": self_w, "last_space": space})
        profiles.append(rows)
    return profiles


def main() -> None:
    model, vocab = load_model()
    text = load_split("val")
    T = model.config.block_size
    data = torch.tensor(vocab.encode(text[: 64 * T + 1]))
    ids = data[: 64 * T].view(64, T)
    profiles = head_profiles(model, ids, vocab.encode(" ")[0])
    print("layer head | mean distance | previous char | itself | last space")
    for layer, rows in enumerate(profiles):
        for head, r in enumerate(rows):
            print(f"  {layer}    {head}  |     {r['distance']:5.1f}     |     {r['previous']:.2f}      |  {r['self']:.2f}  |   {r['last_space']:.2f}")


if __name__ == "__main__":
    main()
