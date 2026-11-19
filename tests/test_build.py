import csv
from collections import Counter, defaultdict

import pytest

from machado_gpt.corpus.build import (
    CATALOG,
    Work,
    assign_splits,
    duplicates,
    group_works,
    order_key,
    shared_text,
)

TEXT_A = " ".join(f"palavra{i}" for i in range(300))
TEXT_B = " ".join(f"termo{i}" for i in range(300))


def test_shared_text_finds_a_republished_passage():
    works = [Work("conto/a", TEXT_A), Work("poesia/b", "outro começo " + TEXT_A[:1500]), Work("romance/c", TEXT_B)]
    pairs = shared_text(works)
    assert [(a, b) for a, b, _, _ in pairs] == [("conto/a", "poesia/b")]


def test_contained_work_is_a_duplicate_of_the_larger():
    pairs = [("crítica/mae", "crítica/revista", 950, 0.95), ("conto/x", "teatro/y", 300, 0.27)]
    words = {"crítica/mae": 1000, "crítica/revista": 30000, "conto/x": 50000, "teatro/y": 8000}
    assert duplicates(pairs, words) == {"crítica/mae": "crítica/revista"}


def test_groups_are_transitive():
    pairs = [("a/1", "a/2", 60, 0.1), ("a/2", "b/3", 60, 0.1)]
    group_of = group_works(["a/1", "a/2", "b/3", "c/4"], pairs)
    assert group_of["a/1"] == group_of["a/2"] == group_of["b/3"] == "a/1+a/2+b/3"
    assert group_of["c/4"] == "c/4"


def test_order_key_is_deterministic_and_depends_on_the_seed():
    assert order_key("romance/domCasmurro") == order_key("romance/domCasmurro")
    assert order_key("romance/domCasmurro", "s1") != order_key("romance/domCasmurro", "s2")


def make_genre(genre, sizes):
    return [Work(f"{genre}/w{i}", "x " * n) for i, n in enumerate(sizes)]


def test_split_is_by_group_and_reaches_targets_per_genre():
    works = make_genre("romance", [100] * 20) + make_genre("conto", [50] * 20)
    group_of = {w.name: w.name for w in works}
    group_of["romance/w0"] = group_of["romance/w1"] = "romance/w0+romance/w1"
    split = assign_splits(works, group_of)
    assert split["romance/w0"] == split["romance/w1"]
    for genre, total in (("romance", 2000), ("conto", 1000)):
        for name in ("val", "test"):
            words = sum(w.words for w in works if w.genre == genre and split[w.name] == name)
            assert 0 < words <= 1.25 * 0.10 * total


def test_split_does_not_depend_on_input_order():
    works = make_genre("romance", [100, 300, 50, 700, 20, 90, 400, 60, 80, 120])
    group_of = {w.name: w.name for w in works}
    assert assign_splits(works, group_of) == assign_splits(list(reversed(works)), group_of)


def test_huge_work_stays_in_training():
    works = make_genre("crônica", [2700] + [100] * 30)
    split = assign_splits(works, {w.name: w.name for w in works})
    assert split["crônica/w0"] == "train"


@pytest.mark.skipif(not CATALOG.exists(), reason="catalog not built")
def test_catalog_has_no_leakage_between_splits():
    with CATALOG.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["obra"] for r in rows]
    assert len(names) == len(set(names)), "a work appears twice in the catalog"
    splits_of_group = defaultdict(set)
    for r in rows:
        if r["situacao"] == "incluída":
            assert r["conjunto"] in {"train", "val", "test"}
            splits_of_group[r["grupo"]].add(r["conjunto"])
        else:
            assert r["conjunto"] == ""
    leaking = {g: s for g, s in splits_of_group.items() if len(s) > 1}
    assert not leaking, f"groups split across sets: {leaking}"


def test_empty_split_takes_a_moderate_group_when_all_are_large():
    works = make_genre("poesia", [150, 160, 170, 180, 200, 140, 190])  # each > 12.5% of the genre
    split = assign_splits(works, {w.name: w.name for w in works})
    counts = Counter(split.values())
    assert counts["val"] == 1 and counts["test"] == 1
