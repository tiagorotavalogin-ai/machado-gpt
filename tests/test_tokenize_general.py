import numpy

from machado_gpt.bpe import BPE
from machado_gpt.build_general import DOC_SEPARATOR
from machado_gpt.tokenize_general import encode_parallel, pieces

# As in the general corpus: no document starts or ends with whitespace.
DOCS = ["Era uma vez um rei.\n\nTinha três filhas.", "Outro   documento,\ncom espaços estranhos.",
        "Poema\nem linhas", "1871 — com travessão e acentuação: ção, à, ü."]
TEXT = DOC_SEPARATOR.join(DOCS) + "\n"


def test_pieces_join_back_into_the_text():
    assert "".join(pieces(TEXT)) == TEXT


def test_encoding_by_document_equals_encoding_the_whole_text(tmp_path):
    bpe = BPE.train(TEXT * 5, 300)
    tokenizer = tmp_path / "bpe.json"
    bpe.save(tokenizer)
    ids = encode_parallel(TEXT, processes=2, tokenizer=tokenizer)
    assert ids.dtype == numpy.uint16
    assert ids.tolist() == bpe.encode(TEXT)
    assert bpe.decode(ids.tolist()) == TEXT


def test_round_trip_stays_exact_even_with_whitespace_at_document_edges(tmp_path):
    text = DOC_SEPARATOR.join(["  com espaço  ", "\n\nnovas linhas\n"]) + "\n"
    bpe = BPE.train(text * 5, 280)
    tokenizer = tmp_path / "bpe.json"
    bpe.save(tokenizer)
    assert bpe.decode(encode_parallel(text, processes=2, tokenizer=tokenizer).tolist()) == text
