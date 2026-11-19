from machado_gpt.corpus.compare import compare_words, is_glued_hyphen


def test_glued_hyphen_is_recognized():
    assert is_glued_hyphen(["admiti-me"], ["admitime"])
    assert is_glued_hyphen(["tique-tique,"], ["tiquetique,"])
    assert is_glued_hyphen(["explicou-se-me"], ["explicou-seme"])
    assert not is_glued_hyphen(["fá-la-ia"], ["fá-la-ia"])
    assert not is_glued_hyphen(["admiti-me"], ["admitiu"])
    assert not is_glued_hyphen(["casa"], ["cas"])


def test_compare_words_classifies_differences():
    ours = "e admiti-me com ela na casa velha".split()
    baseline = "e admitime com ela casa velha agora".split()
    c = compare_words("x", ours, baseline)
    assert c.equal_words == 5
    assert c.hyphen_diffs == 1
    assert [d[0] for d in c.other_diffs] == ["delete", "insert"]
