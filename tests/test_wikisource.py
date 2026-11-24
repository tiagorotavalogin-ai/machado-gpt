import bz2

import pytest

from machado_gpt import wikisource
from machado_gpt.wikisource import INDEX, MAIN, PAGE, RawPage, iter_pages, wikitext_to_text

# ---- wikitext to plain text -----------------------------------------------------


def test_formatting_templates_keep_their_text():
    assert wikitext_to_text("{{c|CAPÍTULO III}}") == "CAPÍTULO III"
    assert wikitext_to_text("disse {{lang|fr|mon ami}} ontem") == "disse mon ami ontem"
    assert wikitext_to_text("{{sc|{{c|Dom Casmurro}}}}") == "Dom Casmurro"


def test_navigation_and_unknown_templates_are_dropped():
    assert wikitext_to_text("{{navegar|autor=Fulano|título=X}}\nTexto da obra.") == "Texto da obra."


def test_hyphenated_word_across_pages_is_kept_once_and_whole():
    end_of_page = "o seu {{hífen|aper|aperfeiçoamento}}"
    start_of_next = "{{hífen-fim|feiçoamento|aperfeiçoamento}} moral"
    assert wikitext_to_text(end_of_page) == "o seu aperfeiçoamento"
    assert wikitext_to_text(start_of_next) == "moral"


def test_notes_page_headers_and_comments_are_dropped():
    text = '<noinclude><pagequality level="3" />Cabeçalho</noinclude>Corpo<ref>nota do editor</ref> final<!-- x -->'
    assert wikitext_to_text(text) == "Corpo final"


def test_links_keep_the_visible_text_and_drop_files_and_categories():
    assert wikitext_to_text("[[Autor:Fulano|o autor]] e [[Lisboa]]") == "o autor e Lisboa"
    assert wikitext_to_text("antes [[Ficheiro:x.jpg|thumb|legenda]] [[Categoria:Romances]]depois") == "antes depois"
    assert wikitext_to_text("[[en:Some work]]Texto") == "Texto"


def test_single_newline_joins_and_blank_line_separates_paragraphs():
    assert wikitext_to_text("uma linha\ncontinua\n\noutro parágrafo") == "uma linha continua\n\noutro parágrafo"


def test_poem_keeps_its_lines():
    assert wikitext_to_text("<poem>Primeiro verso\nSegundo verso</poem>") == "Primeiro verso\nSegundo verso"


def test_heading_becomes_its_own_paragraph():
    assert wikitext_to_text("antes\n== Capítulo I ==\ndepois") == "antes\n\nCapítulo I\n\ndepois"


# ---- reading the dump -----------------------------------------------------------

XML = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
{pages}
</mediawiki>"""
PAGE_XML = """<page><title>{title}</title><ns>{ns}</ns>{redirect}
<revision><text>{text}</text></revision></page>"""


def write_dump(path, pages):
    body = "\n".join(
        PAGE_XML.format(title=t, ns=ns, text=text, redirect='<redirect title="X" />' if red else "")
        for t, ns, text, red in pages
    )
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(XML.format(pages=body))
    return path


def test_iter_pages_skips_redirects_empty_pages_and_other_namespaces(tmp_path):
    dump = write_dump(tmp_path / "d.xml.bz2", [
        ("Obra", MAIN, "texto", False),
        ("Atalho", MAIN, "#REDIRECIONAMENTO [[Obra]]", True),
        ("Vazia", MAIN, "", False),
        ("Usuário:Alguém", "2", "conversa", False),
        ("Página:Livro.djvu/1", PAGE, "página", False),
        ("Galeria:Livro.djvu", INDEX, "índice", False),
    ])
    assert list(iter_pages(dump)) == [
        RawPage(MAIN, "Obra", "texto"),
        RawPage(PAGE, "Página:Livro.djvu/1", "página"),
        RawPage(INDEX, "Galeria:Livro.djvu", "índice"),
    ]


def test_download_refuses_a_dump_whose_sha1_differs(tmp_path):
    (tmp_path / wikisource.DUMP_NAME).write_bytes(b"not the pinned dump")
    with pytest.raises(ValueError, match="SHA-1"):
        wikisource.download(tmp_path)
