import bz2
from collections import Counter

from machado_gpt import build_general
from machado_gpt.build_general import (Document, header_author, is_val, normalize_word, page_key,
                                       read_documents, remove_machado, shingles)
from machado_gpt.wikisource import INDEX, MAIN, PAGE

MODERN = ("Ela saiu ontem do teatro com o comércio fechado e Brás ficou na rua olhando "
          "a chuva cair devagar sobre a cidade inteira")
OLD = ("Ella sahiu hontem do theatro com o commercio fechado e Braz ficou na rua olhando "
       "a chuva cahir devagar sobre a cidade inteira")
OTHER = ("O navio partiu de Lisboa ao amanhecer levando cartas, vinho e três passageiros "
         "que nunca tinham visto o mar")


def doc(*paragraphs, author="", title="Obra"):
    return Document(title, "main", author, list(paragraphs))


# ---- spelling-insensitive comparison --------------------------------------------


def test_old_and_modern_spelling_normalize_to_the_same_word():
    pairs = [("ella", "ela"), ("hontem", "ontem"), ("sahir", "sair"), ("theatro", "teatro"),
             ("commercio", "comércio"), ("Braz", "Brás"), ("pharmacia", "farmácia"), ("lyrio", "lírio")]
    for old, modern in pairs:
        assert normalize_word(old) == normalize_word(modern), (old, modern)


def test_silent_h_rule_keeps_ch_lh_nh():
    assert normalize_word("chave") == "chave"
    assert normalize_word("filho") == "filho"
    assert normalize_word("sonho") == "sonho"
    assert normalize_word("filho") != normalize_word("filo")


def test_old_spelling_edition_shares_the_8_word_sequences():
    assert shingles(OLD) == shingles(MODERN)
    assert not shingles(OTHER) & shingles(MODERN)


# ---- Machado filter (leakage rule) ----------------------------------------------


def test_documents_by_machado_are_removed_whole():
    stats = Counter()
    kept = remove_machado([doc(OTHER, author="Joaquim Maria Machado de Assis"), doc(OTHER, title="B")],
                          MODERN, stats)
    assert [d.title for d in kept] == ["B"]
    assert stats["machado_by_author_docs"] == 1


def test_machado_paragraph_in_old_spelling_is_removed_from_another_work():
    stats = Counter()
    kept = remove_machado([doc(OTHER, OLD, author="Antologia")], MODERN, stats)
    assert kept[0].paragraphs == [OTHER]
    assert stats["machado_by_text_paragraphs"] == 1


def test_machado_poem_typed_one_short_verse_per_paragraph_is_removed():
    # Regression: "O Almada" (a validation work) leaked this way, because no
    # single verse holds two 8-word sequences.
    ws = OLD.split()
    verses = [" ".join(ws[i : i + 5]) for i in range(0, len(ws), 5)]
    assert all(len(shingles(v)) == 0 for v in verses)
    kept = remove_machado([doc("I", *verses, OTHER)], MODERN, Counter())
    assert kept[0].paragraphs == ["I", OTHER]


def test_a_document_left_empty_by_the_filter_is_dropped():
    assert remove_machado([doc(OLD)], MODERN, Counter()) == []


def test_a_single_shared_sequence_is_not_enough_to_remove_a_paragraph():
    quote = " ".join(MODERN.split()[:8])  # exactly one 8-word sequence in common
    paragraph = f"{OTHER} {quote}"
    assert len(shingles(paragraph) & shingles(MODERN)) == 1
    assert remove_machado([doc(paragraph)], MODERN, Counter())[0].paragraphs == [paragraph]


# ---- reading --------------------------------------------------------------------


def test_header_author_reads_the_first_top_level_template():
    assert header_author("{{navegar|obra=X|autor=[[Autor:Machado de Assis|Machado de Assis]]}}") == "Machado de Assis"
    assert header_author("{{:MediaWiki:Proofreadpage_index_template\n|Autor=Eça de Queirós}}") == "Eça de Queirós"
    assert header_author("sem cabeçalho") == ""


def test_page_key_orders_pages_numerically():
    assert page_key("Página:Livro.djvu/23") == ("Livro.djvu", 23)
    assert page_key("Página:Livro.pdf/3") < page_key("Página:Livro.pdf/10")
    assert page_key("Página:Avulsa.jpg") == ("Avulsa.jpg", 0)


def write_dump(path, pages):
    body = "\n".join(f"<page><title>{t}</title><ns>{ns}</ns><revision><text>{text}</text></revision></page>"
                     for t, ns, text in pages)
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(f'<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">{body}</mediawiki>')
    return path


def quality(level, text):
    return f'&lt;noinclude&gt;&lt;pagequality level="{level}" /&gt;&lt;/noinclude&gt;{text}'


def test_scanned_book_joins_proofread_pages_in_order_with_the_index_author(tmp_path):
    dump = write_dump(tmp_path / "d.xml.bz2", [
        ("Página:Livro.djvu/10", PAGE, quality(3, "décima")),
        ("Página:Livro.djvu/2", PAGE, quality(4, "segunda")),
        ("Página:Livro.djvu/5", PAGE, quality(1, "OCR bruto")),
        ("Página:Livro.djvu/6", PAGE, quality(2, "problemática")),
        ("Galeria:Livro.djvu", INDEX, "{{:MediaWiki:Proofreadpage_index_template|Autor=Fulano}}"),
        ("Obra digitada", MAIN, "{{navegar|autor=Beltrano}}\nTexto digitado."),
    ])
    stats = Counter()
    docs = {d.title: d for d in read_documents(dump, stats)}
    assert docs["Livro.djvu"] == Document("Livro.djvu", "scanned", "Fulano", ["segunda", "décima"])
    assert docs["Obra digitada"] == Document("Obra digitada", "main", "Beltrano", ["Texto digitado."])
    assert stats["scanned_words_quality_1"] == 2


# ---- validation split -----------------------------------------------------------


def test_val_split_is_deterministic_and_about_one_percent():
    titles = [f"Obra {i}" for i in range(20_000)]
    first = [is_val(t) for t in titles]
    assert first == [is_val(t) for t in titles]
    assert 0.007 < sum(first) / len(titles) < 0.013


# ---- end to end -----------------------------------------------------------------


def test_main_writes_no_machado_text_to_either_split(monkeypatch, tmp_path):
    val_titles = [t for t in (f"Obra {i}" for i in range(2000)) if is_val(t)][:3]
    titles = val_titles + [f"Treino {i}" for i in range(297) if not is_val(f"Treino {i}")]
    docs = [doc(OTHER, title=t) for t in titles]
    docs += [doc(OLD, title="Antologia"), doc(OTHER, author="Machado de Assis", title="Contos")]
    splits = {"train": MODERN, "val": "nada", "test": "nada"}
    monkeypatch.setattr(build_general, "download", lambda: tmp_path / "dump.xml.bz2")
    monkeypatch.setattr(build_general, "read_documents", lambda dump, stats: list(docs))
    monkeypatch.setattr(build_general, "load_split", lambda name: splits[name])
    monkeypatch.setattr(build_general, "OUT_DIR", tmp_path / "general")
    monkeypatch.setattr(build_general, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(build_general, "PROJECT_ROOT", tmp_path)

    build_general.main()

    train = (tmp_path / "general" / "train.txt").read_text(encoding="utf-8")
    val = (tmp_path / "general" / "val.txt").read_text(encoding="utf-8")
    assert "Ella sahiu" not in train + val
    assert val.count(OTHER) == len(val_titles) == 3
    assert train.count(OTHER) == len(titles) - 3
