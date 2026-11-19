import math

import torch

from machado_gpt.bigram import (
    bigram_nll_sum,
    bigram_probs,
    count_bigrams,
    sample,
    score,
    table_nll_sum,
    train_neural,
)
from machado_gpt.vocab import CharVocab


def test_counts_every_adjacent_pair():
    ids = torch.tensor([1, 2, 1, 2, 3])
    counts = count_bigrams(ids, 4)
    assert counts[1, 2] == 2 and counts[2, 1] == 1 and counts[2, 3] == 1
    assert counts.sum() == len(ids) - 1


def test_rows_are_probability_distributions():
    counts = count_bigrams(torch.tensor([1, 2, 1, 2, 3]), 4)
    probs = bigram_probs(counts, alpha=0.5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4, dtype=torch.float64))
    assert (probs > 0).all()  # smoothing: nothing has probability zero


def test_smoothing_matches_the_formula():
    counts = torch.tensor([[0, 3], [1, 1]])
    probs = bigram_probs(counts, alpha=1.0)
    assert math.isclose(float(probs[0, 0]), 1 / 5)  # (0 + 1) / (3 + 1 * 2)
    assert math.isclose(float(probs[0, 1]), 4 / 5)


def test_uniform_model_loss_is_log_vocab_size():
    probs = torch.full((5, 5), 1 / 5, dtype=torch.float64)
    ids = torch.tensor([0, 1, 2, 3, 4, 0])
    s = score(bigram_nll_sum(probs, ids), n_chars=5, n_bytes=5)
    assert math.isclose(s.nats_per_char, math.log(5))
    assert math.isclose(s.perplexity, 5)
    assert math.isclose(s.bits_per_char, math.log2(5))


def test_bits_per_byte_accounts_for_multibyte_characters():
    s = score(nll_sum_nats=10 * math.log(2), n_chars=5, n_bytes=10)  # 10 bits over 5 chars of 2 bytes
    assert math.isclose(s.bits_per_char, 2) and math.isclose(s.bits_per_byte, 1)


def test_sampling_is_reproducible_and_follows_the_table():
    vocab = CharVocab.from_text("ab")
    probs = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]], dtype=torch.float64)  # a->b, b->a
    assert sample(probs, vocab, "a", 4) == "ababa"
    assert sample(probs, vocab, "a", 4, seed=1) == sample(probs, vocab, "a", 4, seed=1)


def test_table_loss_from_counts_equals_loss_over_positions():
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 6, (500,), generator=gen)
    logits = torch.randn(6, 6, generator=gen)
    per_position = float(torch.nn.functional.cross_entropy(logits[ids[:-1]], ids[1:], reduction="sum"))
    assert math.isclose(table_nll_sum(logits, count_bigrams(ids, 6)), per_position, rel_tol=1e-6)


def test_gradient_descent_recovers_the_count_table():
    gen = torch.Generator().manual_seed(0)
    # a text where each character strongly prefers one successor
    ids = [0]
    table = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    for _ in range(20000):
        ids.append(int(torch.multinomial(table[ids[-1]], 1, generator=gen)))
    ids = torch.tensor(ids)
    counts = count_bigrams(ids, 3)
    W, run = train_neural(ids, 3, steps=400, batch_size=2048, learning_rate=5.0, eval_every=100,
                          eval_counts={"train": counts})
    counted = bigram_probs(counts, alpha=0.0)
    learned = torch.softmax(W.double(), dim=1)
    assert torch.allclose(learned, counted, atol=0.02)
    assert run.history[-1]["train"] < run.history[0]["train"]
