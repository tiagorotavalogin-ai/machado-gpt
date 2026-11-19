import pytest

from machado_gpt.vocab import UNKNOWN, CharVocab


def test_vocab_round_trips_training_text():
    vocab = CharVocab.from_text("Capitu, ó Capitu!")
    text = "Capitu, ó Capitu!"
    assert vocab.decode(vocab.encode(text)) == text


def test_unknown_characters_map_to_id_zero():
    vocab = CharVocab.from_text("abc")
    assert vocab.chars[0] == UNKNOWN
    assert vocab.encode("a½c") == [1, 0, 3]
    assert vocab.decode(vocab.encode("a½c")) == f"a{UNKNOWN}c"


def test_vocab_is_sorted_and_deterministic():
    assert CharVocab.from_text("cba") == CharVocab.from_text("abcabc")
    assert CharVocab.from_text("cba").chars == (UNKNOWN, "a", "b", "c")


def test_training_text_cannot_contain_the_unknown_symbol():
    with pytest.raises(ValueError):
        CharVocab.from_text(f"abc{UNKNOWN}")


def test_save_and_load(tmp_path):
    vocab = CharVocab.from_text("ação — fim")
    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert CharVocab.load(path) == vocab
