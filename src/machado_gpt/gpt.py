"""A small GPT (decoder-only Transformer), written piece by piece.

Data flow for a batch of B sequences of T token ids:

    ids (B, T)
    -> token embedding + position embedding        (B, T, C)
    -> N x Block:
         x = x + attention(layernorm(x))           tokens look at earlier tokens
         x = x + mlp(layernorm(x))                 each token thinks on its own
    -> final layernorm -> linear head               (B, T, V) logits of the next token

Following Karpathy's "Let's build GPT" lecture, without library shortcuts:
attention is computed explicitly (Q, K, V, causal mask, softmax).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # T: maximum context length
    n_layer: int = 4  # number of blocks
    n_head: int = 4  # attention heads per block
    n_embd: int = 128  # C: width of every vector
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Every position mixes the values of earlier positions, weighted by relevance.

    For each position: query q ("what am I looking for?"), key k ("what do I
    offer?"), value v ("what do I pass on"). The weight of position j for
    position i is softmax_j(q_i . k_j / sqrt(head_size)) over j <= i only.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)  # q, k, v for all heads at once
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # mixes the heads back together
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # mask[i, j] = 1 if j <= i: position i may only see itself and the past
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.last_weights: torch.Tensor | None = None  # kept for inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size): each head works on its own slice
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, nh, T, T)
        # Dividing by sqrt(head_size) keeps the scores' variance near 1, so the
        # softmax does not saturate into a one-hot at initialization.
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # each row sums to 1; future positions get 0
        self.last_weights = weights.detach()
        out = self.attn_dropout(weights) @ v  # (B, nh, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concatenate the heads
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """Per-position feed-forward network: widen 4x, nonlinearity, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Communication (attention) followed by computation (MLP), each with
    pre-LayerNorm and a residual connection: x + f(norm(x)).

    The residual path lets gradients flow straight to early layers, which is what
    makes stacking blocks trainable; LayerNorm keeps activations at a stable scale.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)  # what the character is
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # where it is
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Small random weights: at initialization every next character is close to
        # equally likely, so the first loss should be close to ln(vocab_size).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = ids.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence of {T} tokens exceeds block_size {self.config.block_size}")
        positions = torch.arange(T, device=ids.device)
        x = self.drop(self.token_emb(ids) + self.pos_emb(positions))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int, generator: torch.Generator | None = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """Append n_new tokens, each sampled from the distribution over the next one."""
        for _ in range(n_new):
            context = ids[:, -self.config.block_size :]  # the model only sees the last T tokens
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return ids
