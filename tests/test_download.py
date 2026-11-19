from pathlib import Path

import pytest

from machado_gpt.corpus.download import git_blob_sha1, raw_url, safe_destination


def test_git_blob_sha1_matches_git_hash_object():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    # `git hash-object /dev/null`
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_safe_destination_keeps_files_inside_root(tmp_path: Path):
    dest = safe_destination(tmp_path, "pdf/crítica/x.pdf")
    assert dest == (tmp_path / "pdf" / "crítica" / "x.pdf").resolve()


@pytest.mark.parametrize("bad", ["../escape.pdf", "pdf/../../escape.pdf", "/etc/passwd", ""])
def test_safe_destination_rejects_paths_outside_root(tmp_path: Path, bad: str):
    with pytest.raises(ValueError):
        safe_destination(tmp_path, bad)


def test_raw_url_encodes_accents_and_pins_commit():
    url = raw_url("pdf/crítica/a b.pdf")
    assert "/a13ca4a72f268f46db64804a67930d4c0f6db8e7/" in url
    assert url.endswith("pdf/cr%C3%ADtica/a%20b.pdf")
