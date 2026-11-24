import json

import numpy
import pytest
import torch
from test_train_bpe import TINY, TINY_TRAIN, write_stage4

from machado_gpt import train_bpe, train_general
from machado_gpt.gpt import GPT, GPTConfig
from machado_gpt.train_bpe import TrainConfig, train
from machado_gpt.train_general import WindowSampler, verify_general

# ---- one exact pass -------------------------------------------------------------


def test_one_epoch_serves_every_window_exactly_once():
    data = torch.arange(8 * 40 + 1)  # 40 windows of 8 tokens (+1 for the last target)
    sampler = WindowSampler(data, block_size=8, batch_size=4, seed=0)
    assert sampler.steps_per_epoch == 10
    starts = []
    for _ in range(10):
        x, y = sampler.next_batch()
        assert torch.equal(y, x + 1)
        starts += (x[:, 0] // 8).tolist()
    assert sorted(starts) == list(range(40))
    assert starts != list(range(40))  # shuffled


def test_next_epoch_reshuffles_and_state_restores_the_order():
    data = torch.arange(8 * 40 + 1)
    a = WindowSampler(data, 8, 4, seed=0)
    first = [a.next_batch()[0] for _ in range(10)]
    second_epoch = a.next_batch()[0]
    assert a.state() == {"epoch": 1, "pos": 4}
    assert not torch.equal(second_epoch, first[0])

    b = WindowSampler(data, 8, 4, seed=0)
    b.load_state({"epoch": 0, "pos": 12})
    assert torch.equal(b.next_batch()[0], first[3])


# ---- data check -----------------------------------------------------------------


def write_general(tmp_path, tokenizer):
    general_dir = tmp_path / "general"
    general_dir.mkdir()
    rng = numpy.random.default_rng(1)
    splits = {}
    for name, n in (("train", 2000), ("val", 300)):
        rng.integers(0, 256, n).astype(numpy.uint16).tofile(general_dir / f"{name}.bin")
        splits[name] = {"sha256": train_bpe.sha256_file(general_dir / f"{name}.bin")}
    results = tmp_path / "06-wikisource.json"
    tokens = {"tokenizer_sha256": train_bpe.sha256_file(tokenizer), "splits": splits}
    results.write_text(json.dumps({"tokens": tokens}))
    return general_dir, results


def test_training_refuses_general_data_that_differs(tmp_path):
    data_dir, _, vocab = write_stage4(tmp_path)
    tokenizer = data_dir / f"bpe-{vocab}.json"
    general_dir, results = write_general(tmp_path, tokenizer)
    assert set(verify_general(general_dir, results, tokenizer)) == {"tokenizer", "general_train", "general_val"}
    with (general_dir / "train.bin").open("ab") as f:
        f.write(b"\x00\x00")
    with pytest.raises(ValueError, match="general_train"):
        verify_general(general_dir, results, tokenizer)


# ---- train() options ------------------------------------------------------------

DATA = torch.randint(0, 300, (400,), generator=torch.Generator().manual_seed(1))
VAL = torch.randint(0, 300, (100,), generator=torch.Generator().manual_seed(2))


def test_finetuning_starts_from_exactly_the_given_weights(tmp_path):
    torch.manual_seed(123)
    pretrained = GPT(TINY).state_dict()
    cfg = TrainConfig(**{**TINY_TRAIN.__dict__, "steps": 0})  # evaluate step 0, save, stop
    train(TINY, cfg, DATA, VAL, tmp_path, init_state=pretrained)
    saved = torch.load(tmp_path / "best.pt", weights_only=True)["model"]
    assert all(torch.equal(saved[k], pretrained[k]) for k in pretrained)


def test_extra_splits_are_recorded_but_do_not_change_training(tmp_path):
    plain = train(TINY, TINY_TRAIN, DATA, VAL, tmp_path / "plain")
    extra = train(TINY, TINY_TRAIN, DATA, VAL, tmp_path / "extra", extra_eval={"other_val": DATA[:100]})
    assert all("other_val" in p for p in extra["history"])
    assert [p["val"] for p in extra["history"]] == [p["val"] for p in plain["history"]]
    a = torch.load(tmp_path / "plain" / "best.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "extra" / "best.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resumed_run_with_sampler_ends_like_an_uninterrupted_one(tmp_path):
    def sampler():
        return WindowSampler(DATA, TINY.block_size, TINY_TRAIN.batch_size, seed=0)

    full = train(TINY, TINY_TRAIN, DATA, VAL, tmp_path / "full", sampler=sampler())

    crashing = sampler()
    original, calls = crashing.next_batch, {"n": 0}

    def next_batch():
        calls["n"] += 1
        if calls["n"] == 15:  # past the save at step 10
            raise RuntimeError("session dropped")
        return original()

    crashing.next_batch = next_batch
    with pytest.raises(RuntimeError):
        train(TINY, TINY_TRAIN, DATA, VAL, tmp_path / "resumed", sampler=crashing)
    resumed = train(TINY, TINY_TRAIN, DATA, VAL, tmp_path / "resumed", resume=True, sampler=sampler())

    assert [p["val"] for p in resumed["history"]] == pytest.approx([p["val"] for p in full["history"]])
    a = torch.load(tmp_path / "full" / "best.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "resumed" / "best.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---- end to end -----------------------------------------------------------------


def test_pretrain_then_finetune_end_to_end(tmp_path, monkeypatch):
    data_dir, stage4, vocab = write_stage4(tmp_path)
    general_dir, general_results = write_general(tmp_path, data_dir / f"bpe-{vocab}.json")
    tiny = GPTConfig(**{**TINY.__dict__, "vocab_size": vocab, "dropout": 0.2})
    monkeypatch.setattr(train_bpe, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_bpe, "STAGE4_RESULTS", stage4)
    monkeypatch.setattr(train_bpe, "VOCAB_SIZE", vocab)
    monkeypatch.setattr(train_general, "PRESETS", {"medium": tiny})
    monkeypatch.setattr(train_general, "GENERAL_DIR", general_dir)
    monkeypatch.setattr(train_general, "GENERAL_RESULTS", general_results)
    monkeypatch.setattr(train_general, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(train_general, "FINETUNE_STEPS", 10)

    train_general.main(["--phase", "pretrain", "--out", str(tmp_path / "pre")])
    pre = json.loads((tmp_path / "results" / "06-pretrain-medium.json").read_text(encoding="utf-8"))
    assert pre["model"]["dropout"] == 0.0
    assert pre["train"]["steps"] == (2000 - 1) // 8 // 64  # one pass: windows // batch
    assert "machado_val" in pre["history"][0] and "machado_test" not in pre["eval"]

    train_general.main(["--phase", "finetune", "--init", str(tmp_path / "pre" / "best.pt"),
                        "--out", str(tmp_path / "ft")])
    text = (tmp_path / "results" / "06-finetune-medium.json").read_text(encoding="utf-8")
    ft = json.loads(text)
    assert ft["init_sha256"] == pre["checkpoint_sha256"]
    assert ft["model"]["dropout"] == 0.2
    assert ft["train"]["learning_rate"] == pytest.approx(pre["train"]["learning_rate"] / 3)
    assert "general_val" in ft["history"][0] and ft["eval"]["machado_test"]["tokens"] > 0
    assert str(tmp_path) not in text

    # A checkpoint other than the recorded pretraining one is refused.
    torch.save({"model": GPT(tiny).state_dict()}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        train_general.main(["--phase", "finetune", "--init", str(tmp_path / "other.pt"),
                            "--out", str(tmp_path / "ft2")])
