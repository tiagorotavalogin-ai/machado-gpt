import sys

from machado_gpt import tokenize_corpus
from machado_gpt.bpe import BPE


def test_tokenizer_is_trained_on_the_training_split_only(monkeypatch, tmp_path):
    texts = {"train": "texto de treino " * 50, "val": "VALIDAÇÃO SECRETA " * 50, "test": "TESTE SECRETO " * 50}
    seen = []
    original_train = BPE.train.__func__

    def spy(cls, text, vocab_size, verbose_every=0):
        seen.append(text)
        return original_train(cls, text, min(vocab_size, 300))

    monkeypatch.setattr(tokenize_corpus, "load_split", lambda name: texts[name])
    monkeypatch.setattr(BPE, "train", classmethod(spy))
    monkeypatch.setattr(tokenize_corpus, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(tokenize_corpus, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(tokenize_corpus, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["tokenize_corpus"])

    tokenize_corpus.main()

    assert seen == [texts["train"]]
    assert (tmp_path / "runs" / "val.bin").exists()  # val and test are only encoded
