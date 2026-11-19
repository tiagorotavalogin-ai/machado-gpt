from collections import Counter

from machado_gpt.corpus.extract import (
    Line,
    attach_speaker_names,
    body_size,
    build_blocks,
    clean_editorial,
    regular_stanza_size,
    group_blocks,
    is_title,
    is_verse,
    join_block,
    join_prose,
    line_gap,
    normalize,
    resolve_hyphen,
    right_margin,
    split_epigraphs,
    split_index,
    split_notes_section,
    split_small_print,
    strip_note_markers,
    word_forms,
)


def line(text, top, page=1, size=9.9, bold=False, x0=98.6, x1=500.0):
    return Line(page=page, top=top, x0=x0, x1=x1, text=text, size=size, bold=bold)


def texts(blocks):
    return [[l.text for l in block] for block in blocks]


def test_normalize_composes_accents_and_expands_ligatures():
    decomposed = "coração"  # c + cedilla, a + tilde
    assert normalize(decomposed) == "coração"
    assert normalize("ﬁm") == "fim"
    assert normalize("— Continue") == "— Continue"  # dialogue dash kept


def test_body_size_weights_by_characters():
    lines = [line("x" * 100, 10), line("nota da edição", 20, size=7.4)]
    assert body_size(lines) == 9.9


def test_edition_header_and_notes_are_separated_from_the_work():
    lines = [
        line("Dom Casmurro", 70),
        line("Texto de referência:", 100, size=7.4),
        line("Nova Aguilar, 1994.", 110, size=7.4),
        line("Uma noite destas,", 300),
        line("[1] nota no fim", 600, page=9, size=6.0),
    ]
    body, header, notes = split_small_print(lines, 9.9)
    assert [l.text for l in body] == ["Dom Casmurro", "Uma noite destas,"]
    assert header == ["Texto de referência:", "Nova Aguilar, 1994."]
    assert notes == ["[1] nota no fim"]


def test_notes_section_in_body_font_runs_to_the_end():
    lines = [
        line("Verso do poema [1]", 100),
        line("[52]", 112),  # marker alone is not a note
        line("[1] Senado chamo eu ao que", 100, page=9),
        line("naquele tempo tinha o título", 112, page=9),
    ]
    body, notes = split_notes_section(lines)
    assert [l.text for l in body] == ["Verso do poema [1]", "[52]"]
    assert notes == ["[1] Senado chamo eu ao que", "naquele tempo tinha o título"]


def test_note_markers_are_stripped_and_empty_lines_dropped():
    lines = [
        line("A PALMEIRA [1]", 100),
        line("[52]", 112),
        line("CLEÓPATRA[iii]", 118),
        line("[Fragmento de III]", 120),
        line("Salve, gigante", 124),
        line("DO AUTOR [“Meu Anjo”.]", 136),
    ]
    assert [l.text for l in strip_note_markers(lines)] == [
        "A PALMEIRA",
        "CLEÓPATRA",
        "Salve, gigante",
        "DO AUTOR [“Meu Anjo”.]",
    ]


def test_titles_are_bold_or_uppercase():
    assert is_title(line("VISIO", 0, bold=True))
    assert is_title(line("O PÃO D'AÇÚCAR", 0))
    assert is_title(line("III", 0))
    assert not is_title(line("Eras pálida. E os cabelos,", 0))
    assert not is_title(line("1802-1885", 0))


def test_every_index_is_removed():
    lines = [
        line("ÍNDICE", 100, bold=True),
        line("MISS DOLLAR", 112),
        line("MISS DOLLAR", 100, page=2, bold=True),
        line("ÍNDICE", 124, page=2, bold=True),
        line("CAPÍTULO PRIMEIRO", 136, page=2),
        line("CAPÍTULO PRIMEIRO", 160, page=2, bold=True),
        line("Era conveniente ao romance", 184, page=2),
    ]
    body, index = split_index(lines)
    assert [l.text for l in body] == ["MISS DOLLAR", "CAPÍTULO PRIMEIRO", "Era conveniente ao romance"]
    assert index == ["ÍNDICE", "MISS DOLLAR", "ÍNDICE", "CAPÍTULO PRIMEIRO"]


def test_index_is_removed_up_to_the_first_listed_title():
    lines = [
        line("ÍNDICE", 100, bold=True),
        line("MUSA CONSOLATRIX", 112),
        line("VISIO", 124),
        line("MUSA CONSOLATRIX", 100, page=2, bold=True),
        line("Que a mão do tempo", 124, page=2),
    ]
    body, index = split_index(lines)
    assert [l.text for l in body] == ["MUSA CONSOLATRIX", "Que a mão do tempo"]
    assert index == ["ÍNDICE", "MUSA CONSOLATRIX", "VISIO"]


def test_attributed_epigraph_is_removed():
    lines = [
        line("RUÍNAS", 100, bold=True, x0=130),
        line("No hay pájaros en los nidos de antaño.", 124, x0=256),
        line("PROVÉRBIO ESPANHOL", 136, x0=256),
        line("Cobrem plantas sem flor crestados muros;", 160, x0=130),
        line("Rangem portas abertas;", 172, x0=130),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert [l.text for l in body] == [
        "RUÍNAS",
        "Cobrem plantas sem flor crestados muros;",
        "Rangem portas abertas;",
    ]
    assert epigraphs == [["No hay pájaros en los nidos de antaño.", "PROVÉRBIO ESPANHOL"]]


def test_indented_half_verse_after_section_number_is_kept():
    lines = [
        line("III", 100, bold=True, x0=130),
        line("Herói lhe chamam", 124, x0=253),
        line("Quantos o hão visto no fervor da guerra", 136, x0=130),
        line("E na paz o conhecem", 148, x0=130),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert len(body) == 4 and epigraphs == []


def test_right_aligned_prose_epigraph_with_short_attribution_is_removed():
    lines = [
        line("ESAÚ E JACÓ", 100, x0=273),
        line("Dico, che quando l'anima mal nata...", 124, x0=330, x1=514),
        line("Dante", 136, x0=291, x1=321),
        line("CAPÍTULO PRIMEIRO", 160, x0=247, bold=True),
        line("Era a primeira vez que as duas iam", 184, x0=98.6),
        line("ao Morro do Castelo.", 196, x0=98.6),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [["Dico, che quando l'anima mal nata...", "Dante"]]
    assert [l.text for l in body][:2] == ["ESAÚ E JACÓ", "CAPÍTULO PRIMEIRO"]


def test_centered_bold_chapter_titles_are_not_epigraphs():
    lines = [
        line("A IGREJA DO DIABO", 100, x0=250, bold=True),
        line("CAPÍTULO PRIMEIRO", 124, x0=247, bold=True),
        line("DE UMA IDÉIA MIRÍFICA", 136, x0=236, bold=True),
        line("Conta um velho manuscrito", 160, x0=98.6),
        line("beneditino que o Diabo", 172, x0=98.6),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [] and len(body) == 5


def test_uppercase_inscription_of_the_work_is_kept():
    lines = [
        line("EPITÁFIO", 100, x0=280, bold=True),
        line("AQUI JAZ", 124, x0=270, x1=330),
        line("D. EULÁLIA DAMASCENA DE BRITO", 136, x0=200, x1=400),
        line("ORAI POR ELA!", 148, x0=260, x1=340),
        line("Prosseguiu o narrador", 172, x0=98.6),
        line("e contou o resto", 184, x0=98.6),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [] and len(body) == 6


def test_speaker_names_inside_an_epigraph_do_not_duplicate_it():
    lines = [
        line("CAPÍTULO DOS CHAPÉUS", 100, x0=236, bold=True),
        line("GÉRONTE", 124, x0=466, x1=514),
        line("Dans quel chapitre, s'il vous plaît?", 136, x0=343, x1=514),
        line("SGANARELLE", 148, x0=448, x1=514),
        line("Dans le chapitre des chapeaux.", 160, x0=358, x1=514),
        line("MOLIÈRE.", 172, x0=465, x1=514),
        line("Musa, canta o despeito de Mariana,", 196, x0=98.6),
        line("esposa do bacharel Conrado", 208, x0=98.6),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert len(epigraphs) == 1 and epigraphs[0][-1] == "MOLIÈRE."
    assert [l.text for l in body][0] == "CAPÍTULO DOS CHAPÉUS"
    assert len(body) == 3


def test_dedication_and_date_stay_when_an_epigraph_follows():
    lines = [
        line("NO LIMIAR", 100, x0=130, bold=True),
        line("1862", 124, x0=256),
        line("A F. X. DE NOVAIS", 136, x0=256),
        line("Qu'aperçois-tu, mon âme? Au fond, n'est-", 148, x0=256),
        line("ce-pas Dieu?", 160, x0=256),
        line("V. DE LAPRADE", 172, x0=256),
        line("Caía a tarde. Do alto da montanha", 196, x0=130),
        line("A luz do sol se despedia", 208, x0=130),
        line("E a sombra vinha descendo", 220, x0=130),
        line("Sobre o vale adormecido", 232, x0=130),
        line("Como um véu de despedida", 244, x0=130),
        line("Que a noite estendia", 256, x0=130),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [["Qu'aperçois-tu, mon âme? Au fond, n'est-", "ce-pas Dieu?", "V. DE LAPRADE"]]
    assert [l.text for l in body][:3] == ["NO LIMIAR", "1862", "A F. X. DE NOVAIS"]


def test_dedication_pattern():
    from machado_gpt.corpus.extract import DEDICATION

    for text in ["A F. X. DE NOVAIS", "Ao meu primo o Sr. Henrique José", "AO MEU AMIGO O PADRE",
                 "O Sr. P. de Sales Guimarães", "No álbum do Sr. João", "1862",
                 "Corte, 5 de outubro", "Juiz de Fora, 1º de maio", "10 de abril", "RJ, 6 jan. 1855"]:
        assert DEDICATION.match(text), text
    for text in ["A misérrima Dido", "A bondade choremos inocente", "O teu nome é como o óleo",
                 "Os carinhos mal gozados"]:
        assert not DEDICATION.match(text), text


def test_self_quotation_is_kept():
    lines = [
        line("MEU ANJO", 100, x0=130),
        line("É sempre nos teus cantos sonorosos", 124, x0=256, x1=420),
        line("Que eu bebo inspiração.", 136, x0=256, x1=380),
        line("DO AUTOR [“Meu Anjo”.]", 148, x0=256, x1=380),
    ] + [line(f"Verso {n}", 172 + 12 * n, x0=130) for n in range(6)]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == []


def test_one_line_datelines_are_not_mottos():
    lines = []
    for n in range(4):
        lines += [
            line(f"{['I', 'II', 'III', 'IV'][n]}", 100 + 100 * n, x0=290),
            line(f"Corte, {n + 5} de outubro", 112 + 100 * n, x0=400, x1=514),
            line("Minha querida amiga, escrevo", 136 + 100 * n, x0=98.6),
            line("estas linhas à pressa.", 148 + 100 * n, x0=98.6),
        ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == []


def test_album_dedication_is_not_an_epigraph():
    lines = [
        line("MUSA", 100, x0=130),
        line("No álbum do Sr. João Dantas de", 124, x0=256, x1=420),
        line("Sousa", 136, x0=256, x1=290),
        line("Musa", 148, x0=256, x1=285),
        line("Vem, ó musa, ao meu canto", 172, x0=130),
        line("E dá-me a lira", 184, x0=130),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [] and len(body) == 6


def test_stage_direction_is_not_an_epigraph():
    lines = [
        line("UMA ODE DE ANACREONTE", 100, x0=130, bold=True),
        line("(Estão no fim de um banquete, os dois homens", 124, x0=256),
        line("Três escravos.)", 136, x0=256),
        line("Ao vinho, amigos!", 160, x0=130),
        line("Bebamos todos", 172, x0=130),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == [] and len(body) == 5


def test_long_indented_scene_is_not_an_epigraph():
    scene = [line("CÉSAR (entrando pela direita); JOÃO", 124, x0=200, x1=450)]
    for n in range(14):
        scene += [
            line(f"CÉSAR — Fala {n} do criado.", 136 + 24 * n, x0=200, x1=450),
            line("JOÃO", 148 + 24 * n, x0=200, x1=240),
        ]
    lines = [line("UM CASO", 100, x0=130)] + scene + [line(f"Verso {n}", 900, x0=130) for n in range(40)]
    body, epigraphs = split_epigraphs(lines)
    assert epigraphs == []


def test_epigraph_may_start_on_the_next_page():
    lines = [line(f"Verso {n}", 100 + 12 * n, x0=126) for n in range(6)] + [
        line("27 DE SETEMBRO DE 1887.", 721, x0=126, bold=True),
        line("Voilà ce que l'on dit de moi", 118, page=2, x0=249, x1=382),
        line("Dans la Gazette de Hollande. VOLTAIRE", 130, page=2, x0=249, x1=402),
        line("A semana que há passado...", 156, page=2, x0=126),
    ]
    body, epigraphs = split_epigraphs(lines)
    assert len(epigraphs) == 1


def test_motto_with_typing_variations_is_removed():
    variants = [
        ("Voilà ce que l'on dit de moi", "Dans la “Gazette de Hollande”."),
        ("Voilà ce que l’on dit moi", "Dans la “Gazette de Hollande”."),
        ("Voilà, ce que l'on dit de moi", "Dans la “Gazette de Hollande”"),
    ]
    lines = []
    for n, (a, b) in enumerate(variants):
        top = 100 + 200 * n
        lines += [
            line(f"{n + 1} DE NOVEMBRO DE 1886.", top, x0=126),
            line(a, top + 24, x0=249),
            line(b, top + 36, x0=249),
            line(f"Verso {n}", top + 60, x0=126),
            line("Outro verso", top + 72, x0=126),
        ]
    body, epigraphs = split_epigraphs(lines)
    assert len(epigraphs) == 3
    assert not any("Voilà" in l.text for l in body)


def test_repeated_motto_is_removed():
    lines = []
    for n in range(3):
        top = 100 + 200 * n
        lines += [
            line(f"{n + 1} DE NOVEMBRO DE 1886.", top, x0=126),
            line("Voilà ce que l'on dit de moi", top + 24, x0=249),
            line("Dans la “Gazette de Hollande”.", top + 36, x0=249),
            line(f"Verso {n}", top + 60, x0=126),
            line("Outro verso", top + 72, x0=126),
        ]
    body, epigraphs = split_epigraphs(lines)
    assert len(epigraphs) == 3
    assert not any("Voilà" in l.text for l in body)


def test_paragraphs_follow_the_vertical_gap():
    lines = [line("a", 100), line("b", 112), line("c", 136), line("d", 148)]
    assert line_gap(lines) == 12
    assert texts(group_blocks(lines, 12, 500)) == [["a", "b"], ["c", "d"]]


def test_uppercase_title_is_its_own_block_across_a_page_break():
    lines = [line("ESAÚ E JACÓ", 780, x0=273, x1=340), line("Era a primeira vez", 56, page=2)]
    assert texts(group_blocks(lines, 12, 500)) == [["ESAÚ E JACÓ"], ["Era a primeira vez"]]


def test_bold_lines_are_headings_on_their_own():
    lines = [
        line("CAPÍTULO II", 100, bold=True),
        line("DO LIVRO", 112, bold=True),
        line("Agora que expliquei", 136),
    ]
    assert texts(group_blocks(lines, 12, 500)) == [["CAPÍTULO II"], ["DO LIVRO"], ["Agora que expliquei"]]


def test_right_margin_ignores_short_lines():
    lines = [line("x", 10 * i, x1=500.0) for i in range(19)] + [line("fim.", 300, x1=150.0)]
    assert right_margin(lines) == 500.0


def test_page_break_without_punctuation_continues():
    lines = [line("passo a escrever o", 780), line("livro.", 56, page=2)]
    assert texts(group_blocks(lines, 12, 500)) == [["passo a escrever o", "livro."]]


def test_page_break_after_short_sentence_end_starts_paragraph():
    lines = [line("o livro.", 780, x1=200.0), line("Vivo só.", 56, page=2)]
    assert texts(group_blocks(lines, 12, 500)) == [["o livro."], ["Vivo só."]]


def test_page_break_after_full_line_continues_unless_dialogue():
    full = line("acabou a frase no fim da linha.", 780, x1=500.0)
    assert texts(group_blocks([full, line("E seguiu.", 56, page=2)], 12, 500)) == [
        [full.text, "E seguiu."]
    ]
    assert texts(group_blocks([full, line("— Disse ele.", 56, page=2)], 12, 500)) == [
        [full.text],
        ["— Disse ele."],
    ]


def test_verse_keeps_one_line_per_verse():
    stanza = [
        line("Chora, menina, chora,", 100, x1=250),
        line("Chora, porque não tem", 112, x1=240),
        line("Vintém,", 124, x1=150),
    ]
    assert is_verse(stanza, widest=500)
    assert join_block(stanza, 500, Counter(), {}) == "Chora, menina, chora,\nChora, porque não tem\nVintém,"


def test_prose_is_joined_with_spaces():
    lowercase_continuation = [line("Uma noite destas, vindo da", 100, x1=480), line("cidade.", 112, x1=150)]
    full_lines = [line("O assunto deste poema é rigorosamente", 100, x1=495), line("Histórico.", 112, x1=150)]
    assert not is_verse(lowercase_continuation, widest=500)
    assert not is_verse(full_lines, widest=500)
    assert join_block(lowercase_continuation, 500, Counter(), {}) == "Uma noite destas, vindo da cidade."


def test_hyphen_is_decided_by_the_corpus():
    forms = Counter({"admiti-me": 1, "cavalo": 3})
    assert resolve_hyphen("e admiti", "me com ela", forms) == "keep"
    assert resolve_hyphen("o cava", "lo correu", forms) == "join"
    assert resolve_hyphen("um xyz", "abc", forms) == "review"


def test_broken_word_is_not_evidence_for_itself():
    forms = word_forms([[line("fazê-", 0), line("lo amanhã", 12), line("e fazê-lo hoje", 24)]])
    assert forms["fazê"] == 0
    assert forms["fazê-lo"] == 1


def test_join_prose_applies_hyphen_decisions():
    forms = Counter({"admiti-me": 1, "cavalo": 1})
    log: dict[str, list[str]] = {}
    text = join_prose(["e admiti-", "me com ela; o cava-", "lo correu"], forms, log)
    assert text == "e admiti-me com ela; o cavalo correu"
    assert len(log["keep"]) == 1 and len(log["join"]) == 1


def test_dialogue_dash_at_line_end_is_not_a_hyphen():
    log: dict[str, list[str]] = {}
    assert join_prose(["disse ele -", "e saiu"], Counter(), log) == "disse ele - e saiu"
    assert log == {}


def quatrain(page, top, first):
    return [line(f"Verso {first}{n}", top + 12 * n, page=page, x0=130, x1=300) for n in range(4)]


def test_regular_stanza_size_needs_most_stanzas_equal():
    poem = [quatrain(1, 100, "a"), quatrain(1, 200, "b"), quatrain(1, 300, "c")]
    assert regular_stanza_size(poem, widest=500) == 4
    poem.append([line("Xis", 400, x0=130, x1=200), line("Ypsilon", 412, x0=130, x1=200)])
    assert regular_stanza_size(poem, widest=500) is None  # 3 of 4 = 75%


def regular_poem_then(tail):
    """A poem with three whole quatrains on page 1, followed by `tail`."""
    return (
        [line("POEMA", 50, x0=130, x1=200)]
        + quatrain(1, 100, "x") + quatrain(1, 200, "y") + quatrain(1, 300, "z")
        + tail
    )


def test_complete_stanza_at_page_end_closes_it():
    lines = regular_poem_then(quatrain(1, 700, "d") + quatrain(2, 60, "e"))
    blocks = texts(build_blocks(lines, 12, 480, 500))
    assert blocks[-2] == [f"Verso d{n}" for n in range(4)]
    assert blocks[-1] == [f"Verso e{n}" for n in range(4)]


def test_stanza_split_by_the_page_continues():
    tail = [
        line("Verso a0", 724, x0=130, x1=300),
        line("Verso a1.", 736, x0=130, x1=300),  # punctuation does not close it
        line("Verso a2", 60, page=2, x0=130, x1=300),
        line("Verso a3", 72, page=2, x0=130, x1=300),
    ]
    blocks = texts(build_blocks(regular_poem_then(tail), 12, 480, 500))
    assert blocks[-1] == ["Verso a0", "Verso a1.", "Verso a2", "Verso a3"]


def test_dialogue_lines_around_a_page_break_in_prose_are_not_a_stanza():
    lines = [
        line("Uma noite destas, vindo da cidade", 100, x1=495),
        line("para o Engenho Novo, encontrei", 112, x1=495),
        line("um rapaz do bairro.", 124, x1=250),
        line("— Já acabei, murmurou ele.", 780, x1=250),
        line("— São muito bonitos.", 56, page=2, x1=200),
    ]
    blocks = texts(build_blocks(lines, 12, 495, 495))
    assert blocks[-2:] == [["— Já acabei, murmurou ele."], ["— São muito bonitos."]]


def test_prose_page_break_rule_is_unchanged():
    lines = [line("o livro.", 780, x1=200.0), line("Vivo só.", 56, page=2)]
    assert texts(build_blocks(lines, 12, 500, 500)) == [["o livro."], ["Vivo só."]]


def test_inline_speeches_start_new_blocks():
    lines = [
        line("MARGARIDA — Mas que as mulheres sintam é natural;", 100, x1=495),
        line("os homens...", 112, x1=200),
        line("SEABRA — Os homens são homens.", 124, x1=300),
        line("D. HELENA (imitando-a) — Helena!", 136, x1=300),
    ]
    assert texts(build_blocks(lines, 12, 495, 495)) == [
        ["MARGARIDA — Mas que as mulheres sintam é natural;", "os homens..."],
        ["SEABRA — Os homens são homens."],
        ["D. HELENA (imitando-a) — Helena!"],
    ]


def test_speaker_names_are_joined_to_their_speech():
    paragraphs = [
        "PERSONAGENS", "VALENTIM", "DOUTOR",
        "Cena I",
        "VALENTIM", "Ah! És tu?",
        "DOUTOR", "Oh! Hoje é o dia das surpresas.",
        "VALENTIM", "(irônico)\nAh!",
        "DOUTOR", "Acertaste.",
        "ATO ÚNICO",
    ]
    assert attach_speaker_names(paragraphs) == [
        "PERSONAGENS", "VALENTIM", "DOUTOR",
        "Cena I",
        "VALENTIM\nAh! És tu?",
        "DOUTOR\nOh! Hoje é o dia das surpresas.",
        "VALENTIM\n(irônico)\nAh!",
        "DOUTOR\nAcertaste.",
        "ATO ÚNICO",
    ]


def test_line_gap_ignores_the_gaps_between_short_speeches():
    lines = []
    top = 100
    for n in range(10):  # many short speeches: name, gap 24, speech, gap 24
        lines += [line("VALENTIM", top, x1=150), line("Ah!", top + 24, x1=120)]
        top += 48
    lines += [line("Uma fala longa que vai até a margem direita", top, x1=500), line("e continua.", top + 12, x1=200)]
    assert line_gap(lines) == 12


def test_speaker_absorbs_a_separate_stage_direction_and_the_speech():
    paragraphs = ["JÚPITER", "(entra, pára e presta ouvido)", "Cuidei ouvir a flauta.", "JÚPITER", "Bem.", "JÚPITER", "Sim."]
    assert attach_speaker_names(paragraphs) == [
        "JÚPITER\n(entra, pára e presta ouvido)\nCuidei ouvir a flauta.",
        "JÚPITER\nBem.",
        "JÚPITER\nSim.",
    ]


def test_editorial_comments_go_and_restored_words_stay():
    assert clean_editorial("o brigadeiro, pai [ilegível] atenda") == "o brigadeiro, pai atenda"
    assert clean_editorial("não seja aprovada [pouco legível: recusa?].") == "não seja aprovada."
    assert clean_editorial("em 2 de março [a palavra março está riscada, a lápis] de 1862") == "em 2 de março de 1862"
    assert clean_editorial("semelhante veneração [?] A vingança") == "semelhante veneração A vingança"
    assert clean_editorial("tendo sido vendido [no] México") == "tendo sido vendido no México"
    assert clean_editorial("DO AUTOR [“Meu Anjo”.]") == "DO AUTOR [“Meu Anjo”.]"
