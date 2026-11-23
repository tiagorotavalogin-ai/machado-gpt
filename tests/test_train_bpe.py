import json
import math

import numpy
import pytest
import torch

from machado_gpt import train_bpe
from machado_gpt.bpe import BPE
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_bpe import TrainConfig, get_batch, nll_sum, train, verify_stage4

TINY = GPTConfig(vocab_size=300, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.1, tie_weights=True)
TINY_TRAIN = TrainConfig(steps=20, batch_size=4, warmup_steps=2, eval_every=5, train_eval_batches=2,
                         patience=100, save_last_every=10)


def test_presets_have_the_planned_sizes():
    assert 4.0e6 < GPT(train_bpe.PRESETS["small"]).num_params() < 4.6e6
    assert 11.5e6 < GPT(train_bpe.PRESETS["medium"]).num_params() < 12.8e6


def test_tied_weights_share_one_matrix():
    tied, untied = GPT(TINY), GPT(GPTConfig(**{**TINY.__dict__, "tie_weights": False}))
    assert tied.head.weight is tied.token_emb.weight
    assert untied.num_params() - tied.num_params() == TINY.vocab_size * TINY.n_embd


def test_batch_targets_are_inputs_shifted_by_one():
    x, y = get_batch(torch.arange(100), block_size=8, batch_size=4, gen=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_nll_sum_matches_mean_loss_of_the_windows():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    data = torch.randint(0, 10, (8 * 5 + 1,))
    total, n = nll_sum(model, data, batch_size=2)
    assert n == 40
    _, loss = model.eval()(data[:-1].view(5, 8), data[1:].view(5, 8))
    assert math.isclose(total / n, loss.item(), rel_tol=1e-5)


def write_stage4(tmp_path):
    """A fake stage 4 output: tokenizer, three splits and their SHA-256."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bpe = BPE.train("o que é que eu fiz " * 30, 300)  # stops early: few distinct pairs
    bpe.save(data_dir / f"bpe-{bpe.vocab_size}.json")
    rng = numpy.random.default_rng(0)
    splits = {}
    for name in ("train", "val", "test"):
        ids = rng.integers(0, 256, 500)  # byte tokens: valid for any tokenizer
        ids.astype(numpy.uint16).tofile(data_dir / f"{name}.bin")
        splits[name] = {"sha256": train_bpe.sha256_file(data_dir / f"{name}.bin")}
    results = tmp_path / "04-bpe.json"
    tok_sha = train_bpe.sha256_file(data_dir / f"bpe-{bpe.vocab_size}.json")
    results.write_text(json.dumps({"vocab_size": bpe.vocab_size, "splits": splits, "tokenizer_sha256": tok_sha}))
    return data_dir, results, bpe.vocab_size


def test_training_refuses_data_that_differs_from_stage_4(tmp_path):
    data_dir, results, _ = write_stage4(tmp_path)
    assert set(verify_stage4(data_dir, results)) == {"tokenizer", "train", "val", "test"}
    with (data_dir / "val.bin").open("ab") as f:
        f.write(b"\x00\x00")
    with pytest.raises(ValueError, match="val"):
        verify_stage4(data_dir, results)


def test_resumed_run_ends_exactly_like_an_uninterrupted_one(tmp_path, monkeypatch):
    """A dropped session (here: an exception at step ~15) resumed from last.pt
    (saved at step 10) must reach the same weights as a run that never stopped."""
    data = torch.randint(0, 300, (400,), generator=torch.Generator().manual_seed(1))
    val = torch.randint(0, 300, (100,), generator=torch.Generator().manual_seed(2))
    full = train(TINY, TINY_TRAIN, data, val, tmp_path / "full")

    calls = {"n": 0}
    original = train_bpe.get_batch

    def crashing(*args):
        calls["n"] += 1
        if calls["n"] == 25:  # past the save at step 10
            raise RuntimeError("session dropped")
        return original(*args)

    monkeypatch.setattr(train_bpe, "get_batch", crashing)
    with pytest.raises(RuntimeError):
        train(TINY, TINY_TRAIN, data, val, tmp_path / "resumed")
    monkeypatch.setattr(train_bpe, "get_batch", original)
    resumed = train(TINY, TINY_TRAIN, data, val, tmp_path / "resumed", resume=True)

    assert [p["step"] for p in resumed["history"]] == [0, 5, 10, 15, 20]
    assert [p["val"] for p in resumed["history"]] == pytest.approx([p["val"] for p in full["history"]])
    a = torch.load(tmp_path / "full" / "best.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "resumed" / "best.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_early_stopping_keeps_the_best_checkpoint(tmp_path):
    data = torch.randint(0, 300, (400,), generator=torch.Generator().manual_seed(1))
    cfg = TrainConfig(**{**TINY_TRAIN.__dict__, "steps": 1000, "patience": 2, "learning_rate": 0.0,
                         "min_learning_rate": 0.0})  # lr 0: validation never improves after step 0
    run = train(TINY, cfg, data, data[:100], tmp_path)
    assert run["stopped"].startswith("early stopping")
    assert len(run["history"]) == 3
    assert torch.load(tmp_path / "best.pt", weights_only=True)["step"] == 0


def test_main_writes_results_without_machine_paths(tmp_path, monkeypatch):
    data_dir, stage4, vocab_size = write_stage4(tmp_path)
    monkeypatch.setattr(train_bpe, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_bpe, "STAGE4_RESULTS", stage4)
    monkeypatch.setattr(train_bpe, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(train_bpe, "VOCAB_SIZE", vocab_size)
    monkeypatch.setattr(train_bpe, "PRESETS", {"small": GPTConfig(**{**TINY.__dict__, "vocab_size": vocab_size})})
    train_bpe.main(["--size", "small", "--steps", "10", "--out", str(tmp_path / "ckpt")])

    text = (tmp_path / "results" / "05-gpt-bpe-small.json").read_text(encoding="utf-8")
    results = json.loads(text)
    assert results["checkpoint_sha256"] == train_bpe.sha256_file(tmp_path / "ckpt" / "best.pt")
    assert results["val"]["bits_per_byte"] > 0 and results["test"]["tokens"] > 0
    assert str(tmp_path) not in text
