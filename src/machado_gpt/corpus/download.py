"""Download the luxedo PDFs and TXTs pinned to a fixed commit.

Every file is checked against the git blob SHA-1 that GitHub reports for that
commit, so the bytes on disk are exactly the bytes in the pinned revision.
A manifest with size, blob SHA-1 and SHA-256 is written to the catalog.

Usage:
    uv run python -m machado_gpt.corpus.download
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = "luxedo/dataset-machado-de-assis"
COMMIT = "a13ca4a72f268f46db64804a67930d4c0f6db8e7"
PREFIXES = ("pdf/", "txt/")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "luxedo"
MANIFEST = PROJECT_ROOT / "data" / "catalog" / "luxedo_manifest.csv"

USER_AGENT = "machado-gpt-corpus-downloader"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    blob_sha1: str


def git_blob_sha1(data: bytes) -> str:
    """Hash bytes the way `git hash-object` does."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def safe_destination(root: Path, remote_path: str) -> Path:
    """Map a repository path to a local path, refusing anything outside root."""
    parts = PurePosixPath(remote_path).parts
    if not parts or PurePosixPath(remote_path).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe path in remote tree: {remote_path!r}")
    dest = root.joinpath(*parts).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe path in remote tree: {remote_path!r}")
    return dest


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def list_remote_files() -> list[RemoteFile]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(_get(url))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated tree")
    files = [
        RemoteFile(item["path"], item["size"], item["sha"])
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(PREFIXES)
    ]
    return sorted(files, key=lambda f: f.path)


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{urllib.parse.quote(path)}"


def fetch(remote: RemoteFile, root: Path) -> tuple[Path, bytes]:
    """Return the verified bytes of a file, reusing a valid local copy."""
    dest = safe_destination(root, remote.path)
    if dest.exists():
        data = dest.read_bytes()
        if git_blob_sha1(data) == remote.blob_sha1:
            return dest, data

    data = _get(raw_url(remote.path))
    actual = git_blob_sha1(data)
    if actual != remote.blob_sha1:
        raise RuntimeError(f"hash mismatch for {remote.path}: {actual} != {remote.blob_sha1}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    time.sleep(0.1)  # be gentle with the server
    return dest, data


def write_manifest(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "size", "git_blob_sha1", "sha256", "repo", "commit"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    remotes = list_remote_files()
    print(f"{len(remotes)} files at {REPO}@{COMMIT[:7]}")
    rows = []
    for i, remote in enumerate(remotes, 1):
        _, data = fetch(remote, RAW_DIR)
        rows.append(
            {
                "path": remote.path,
                "size": len(data),
                "git_blob_sha1": remote.blob_sha1,
                "sha256": hashlib.sha256(data).hexdigest(),
                "repo": REPO,
                "commit": COMMIT,
            }
        )
        if i % 20 == 0 or i == len(remotes):
            print(f"  {i}/{len(remotes)} verified")
    write_manifest(rows, MANIFEST)
    total = sum(int(r["size"]) for r in rows)
    print(f"manifest: {MANIFEST.relative_to(PROJECT_ROOT)} ({total / 1e6:.1f} MB verified)")


if __name__ == "__main__":
    main()
