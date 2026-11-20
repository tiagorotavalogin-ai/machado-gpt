import random

import pytest

from machado_gpt.bpe import BPE, merge_pair, pre_tokenize

TEXT = (
    "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da "
    "Central um rapaz aqui do bairro, que eu conheço de vista e de chapéu.\n\n"
    "— Continue, disse eu acordando.\n— Já acabei, murmurou ele.\n"
) * 20


def test_pre_tokenization_keeps_the_leading_space_with_the_word_and_loses_nothing():
    chunks = pre_tokenize("Capitu, ó Capitu! Em 1899 — disse.\n\nFim")
    assert chunks[:4] == ["Capitu", ",", " ó", " Capitu"]
    assert " 189" in chunks and "9" in chunks  # numbers in groups of up to 3 digits
    assert "".join(chunks) == "Capitu, ó Capitu! Em 1899 — disse.\n\nFim"


def test_pre_tokenization_loses_no_character():
    """Concatenating the chunks must give back the text, for every character."""
    import string

    text = string.printable + "_ __init__ a_b ºª²½€—«»“”‘’…\u00a0\u2009🙂"
    assert "".join(pre_tokenize(text)) == text
    for code in range(0x20, 0x3000):
        assert "".join(pre_tokenize(f"x{chr(code)}y")) == f"x{chr(code)}y", hex(code)


def test_merge_pair_is_left_to_right_and_non_overlapping():
    assert merge_pair([1, 1, 1, 2, 1, 1], (1, 1), 9) == [9, 1, 2, 9]


def test_first_merge_is_the_most_frequent_pair():
    bpe = BPE.train("aaab aaab aaab", vocab_size=257)
    assert bpe.merges == [(ord("a"), ord("a"))]


def test_training_is_deterministic():
    assert BPE.train(TEXT, 300).merges == BPE.train(TEXT, 300).merges


def test_merges_never_cross_chunks():
    bpe = BPE.train(TEXT, 400)
    for token in bpe.token_bytes()[256:]:
        text = token.decode("utf-8", errors="ignore")
        assert len(pre_tokenize(text)) <= 1, token


def test_round_trip_on_training_text():
    bpe = BPE.train(TEXT, 350)
    assert bpe.decode(bpe.encode(TEXT)) == TEXT
    assert len(bpe.encode(TEXT)) < len(TEXT.encode("utf-8"))  # it compresses


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_on_arbitrary_unicode(seed):
    """Byte-level BPE must reproduce any text exactly, even characters never seen."""
    bpe = BPE.train(TEXT, 350)
    rng = random.Random(seed)
    alphabet = "aãçéõ —«»½€\n\t" + "".join(chr(rng.randrange(0x20, 0x2FFF)) for _ in range(50)) + "🙂😀𝔄"
    text = "".join(rng.choice(alphabet) for _ in range(500))
    assert bpe.decode(bpe.encode(text)) == text


def test_save_and_load(tmp_path):
    bpe = BPE.train(TEXT, 320)
    path = tmp_path / "bpe.json"
    bpe.save(path)
    loaded = BPE.load(path)
    assert loaded.merges == bpe.merges
    assert loaded.encode(TEXT) == bpe.encode(TEXT)
