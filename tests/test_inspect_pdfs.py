from collections import Counter

import pytest

from machado_gpt.corpus.inspect_pdfs import (
    body_font_size,
    has_edition_header,
    repeated_line_count,
    two_most_common_gaps,
)


@pytest.mark.parametrize(
    "text",
    [
        "Texto-fonte:",
        "Texto-Fonte:",
        "Textos-Fonte:",
        "Texto de referência:",
        "Edição referência:",
        "TEXTO-FONTE",
    ],
)
def test_edition_header_is_case_insensitive(text: str):
    assert has_edition_header(f"Título\n{text}\nObra Completa")


def test_edition_header_absent():
    assert not has_edition_header("Título\nUm dia, respondendo a Alencar")


def test_body_font_size_is_the_most_frequent_size():
    assert body_font_size(Counter({9.9: 1000, 7.4: 50, 12.0: 10})) == 9.9
    assert body_font_size(Counter()) == 0.0


def test_repeated_lines_ignore_page_numbers():
    pages = [[f"Dom Casmurro {n}", f"texto único {n * 'x'}"] for n in range(10)]
    assert repeated_line_count(pages) == 1


def test_repeated_lines_need_enough_pages():
    assert repeated_line_count([["cabeçalho"]] * 4) == 0


def test_gaps_split_line_and_paragraph_spacing():
    assert two_most_common_gaps(Counter({12: 400, 24: 100, 48: 5})) == (12, 24)
    assert two_most_common_gaps(Counter({12: 10})) == (12, 0)
    assert two_most_common_gaps(Counter()) == (0, 0)
