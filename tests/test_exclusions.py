import csv
from pathlib import Path

import pytest

from machado_gpt.corpus import exclusions
from machado_gpt.corpus.exclusions import Exclusion, apply, load, locate

BLOCKS = [
    "Crisálidas",
    "LÚCIA",
    "1860\n(Alfred de Musset)",
    "Nós estávamos sós",
    "O DILÚVIO",
    "E ao som dos nossos cânticos",
    "UN VIEUX PAYS",
    "Il est un vieux pays",
    "LUZ ENTRE SOMBRAS",
]


def entry(inicio, fim="", acao="excluir", obra="poesia/x"):
    return Exclusion(obra, inicio, fim, "traducao", "Autor", "subtítulo no PDF", acao, "")


def test_range_runs_until_the_block_where_the_kept_text_resumes():
    assert locate(BLOCKS, entry("LÚCIA", "O DILÚVIO")) == range(1, 4)


def test_empty_end_runs_to_the_end_of_the_work():
    assert locate(BLOCKS, entry("LUZ ENTRE")) == range(8, 9)


def test_ambiguous_or_missing_start_fails_loudly():
    with pytest.raises(ValueError):
        locate(BLOCKS + ["LÚCIA"], entry("LÚCIA", "O DILÚVIO"))
    with pytest.raises(ValueError):
        locate(BLOCKS, entry("INEXISTENTE"))
    with pytest.raises(ValueError):
        locate(BLOCKS, entry("LÚCIA", "INEXISTENTE"))


def test_excluir_removes_and_revisar_only_reports():
    entries = [
        entry("LÚCIA", "O DILÚVIO"),
        entry("UN VIEUX PAYS", "LUZ ENTRE SOMBRAS", acao="revisar"),
        entry("LÚCIA", "O DILÚVIO", obra="poesia/outra"),  # other work: ignored
    ]
    kept, removed, review = apply("poesia/x", BLOCKS, entries)
    assert "LÚCIA" not in kept and "Nós estávamos sós" not in kept
    assert "UN VIEUX PAYS" in kept
    assert [r["inicio"] for r in removed] == ["LÚCIA"] and removed[0]["blocos"] == 3
    assert [r["inicio"] for r in review] == ["UN VIEUX PAYS"]


def test_catalog_is_well_formed():
    entries = load()
    assert entries, "catalog is empty"
    for e in entries:
        assert e.obra.count("/") == 1
        assert e.tipo in {"traducao", "imitacao", "outro_autor", "epigrafe", "nota_do_autor", "lingua_estrangeira"}


def test_unknown_action_is_rejected(tmp_path: Path):
    path = tmp_path / "x.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obra", "inicio", "fim_antes_de", "tipo", "autor_original", "fonte", "acao", "observacao"])
        w.writerow(["poesia/x", "A", "", "traducao", "", "fonte", "apagar", ""])
    with pytest.raises(ValueError):
        load(path)
