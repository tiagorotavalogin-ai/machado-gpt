import math

import torch

from machado_gpt.gpt import GPT, GPTConfig

CONFIG = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def model(seed=0):
    torch.manual_seed(seed)
    return GPT(CONFIG).eval()


def test_shapes():
    ids = torch.randint(0, 20, (3, 16))
    logits, loss = model()(ids, ids)
    assert logits.shape == (3, 16, 20)
    assert loss.shape == ()


def test_initial_loss_is_close_to_uniform():
    ids = torch.randint(0, 20, (8, 16))
    _, loss = model()(ids, ids)
    assert abs(loss.item() - math.log(20)) < 0.1


def test_attention_is_causal():
    """Changing a future token must not change the predictions for earlier positions."""
    m = model()
    ids = torch.randint(0, 20, (1, 16))
    changed = ids.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % 20
    a, _ = m(ids)
    b, _ = m(changed)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_attention_weights_are_distributions_over_the_past():
    m = model()
    m(torch.randint(0, 20, (2, 16)))
    w = m.blocks[0].attn.last_weights  # (B, heads, T, T)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)))
    assert torch.all(w.triu(diagonal=1) == 0)


def test_overfits_a_single_batch():
    """If the model cannot memorize one small batch, something is broken."""
    torch.manual_seed(0)
    m = GPT(CONFIG)
    ids = torch.randint(0, 20, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(ids[:, :-1], ids[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generation_respects_context_length_and_seed():
    m = model()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = m.generate(start, 40, generator=torch.Generator().manual_seed(1))
    b = m.generate(start, 40, generator=torch.Generator().manual_seed(1))
    assert a.shape == (1, 41) and torch.equal(a, b)
