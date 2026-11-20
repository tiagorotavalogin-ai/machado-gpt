import math

import torch

from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_char import TrainConfig, full_nll_sum, get_batch, learning_rate


def test_batch_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, gen=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_learning_rate_warms_up_then_decays_to_the_minimum():
    cfg = TrainConfig(steps=1000, learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=100)
    assert learning_rate(0, cfg) < learning_rate(50, cfg) < learning_rate(99, cfg)
    assert math.isclose(learning_rate(99, cfg), 1e-3)
    assert learning_rate(500, cfg) < 1e-3
    assert math.isclose(learning_rate(1000, cfg), 1e-4)


def test_full_nll_matches_mean_loss_of_the_windows():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)).eval()
    data = torch.randint(0, 10, (8 * 5 + 1,))
    total, n = full_nll_sum(model, data, batch_size=2)
    assert n == 40
    x = data[:-1].view(5, 8)
    y = data[1:].view(5, 8)
    _, loss = model(x, y)
    assert math.isclose(total / n, loss.item(), rel_tol=1e-5)
