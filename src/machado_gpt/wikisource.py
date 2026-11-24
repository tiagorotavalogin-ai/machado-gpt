"""Portuguese Wikisource as the general corpus for pretraining (stage 6).

The dump is pinned to a dated snapshot and checked against the SHA-1 that
Wikimedia publishes. Two kinds of pages hold text:

- main namespace (ns 0): works typed directly;
- "Página:" namespace (ns 106): scanned books transcribed page by page. The
  work's main page only transcludes them, so each book is rebuilt by joining
  its pages in page order.

Wikitext is turned into plain text with mwparserfromhell, walking the parse
tree: templates that carry text keep it ({{c|CAPÍTULO III}} -> "CAPÍTULO III"),
hyphenation across scanned pages keeps the whole word once, and navigation,
tables, notes (<ref>), page headers (<noinclude>) and licence boxes are dropped.
"""

from __future__ import annotations

import bz2
import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mwparserfromhell as mw
from mwparserfromhell.nodes import (Argument, Comment, ExternalLink, Heading, HTMLEntity, Tag, Template,
                                    Text, Wikilink)

from machado_gpt.corpus.download import PROJECT_ROOT

DUMP_DATE = "20261101"
DUMP_NAME = f"ptwikisource-{DUMP_DATE}-pages-articles.xml.bz2"
DUMP_URL = f"https://dumps.wikimedia.org/ptwikisource/{DUMP_DATE}/{DUMP_NAME}"
DUMP_SHA1 = "a19fdb638ecb2432bea61616ebd363b898d7ae4e"  # from the dump's published sha1sums.txt
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wikisource"
USER_AGENT = "machado-gpt-corpus-downloader"

MAIN, PAGE, INDEX = "0", "106", "104"  # namespaces: works, scanned pages, book index ("Galeria")


# ---- download -------------------------------------------------------------------


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Download the pinned dump once; refuse it if the SHA-1 differs."""
    path = raw_dir / DUMP_NAME
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        request = urllib.request.Request(DUMP_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response, tmp.open("wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
        tmp.rename(path)
    if sha1_file(path) != DUMP_SHA1:
        raise ValueError(f"{DUMP_NAME}: SHA-1 differs from the published one")
    return path


# ---- reading the dump -----------------------------------------------------------


@dataclass(frozen=True)
class RawPage:
    ns: str
    title: str
    text: str


def iter_pages(path: Path, namespaces: tuple[str, ...] = (MAIN, PAGE, INDEX)) -> Iterator[RawPage]:
    """Stream the dump, skipping redirects and other namespaces."""
    def tag(e: ET.Element) -> str:
        return e.tag.rsplit("}", 1)[-1]

    with bz2.open(path) as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if tag(el) != "page":
                continue
            title = ns = text = None
            redirect = False
            for child in el:
                name = tag(child)
                if name == "title":
                    title = child.text
                elif name == "ns":
                    ns = child.text
                elif name == "redirect":
                    redirect = True
                elif name == "revision":
                    for r in child:
                        if tag(r) == "text":
                            text = r.text or ""
            el.clear()
            if ns in namespaces and not redirect and text and title:
                yield RawPage(ns, title, text)


# ---- wikitext to plain text -----------------------------------------------------

# Templates whose first positional argument is the text itself (formatting only).
KEEP_FIRST = {
    "c", "center", "centralizado", "centro", "sc", "sc2", "small caps", "smallcaps", "versalete", "smaller",
    "small", "larger", "large", "x-larger", "xx-larger", "xxx-larger", "x-smaller", "lema", "d", "t1", "t2",
    "t3", "t4", "right", "direita", "justificado", "serifa", "bloco centro", "uc", "capitular", "sup", "def",
    "pt", "nota", "kt", "sic", "abv", "nowrap", "u", "sub", "fine", "espaçado", "letter-spacing", "mono",
    "itálico", "negrito", "cor", "fonte", "font", "texto", "block center", "float right", "float left",
}
KEEP_SECOND = {"lang", "âncora", "anchor", "k"}  # {{lang|fr|text}}, {{âncora|id|text}}, {{k|size|text}}
WHOLE_WORD = {"hífen", "hifenizado", "começo de palavra hifenizada", "hws"}  # {{hífen|aper|aperfeiçoamento}}
DROP_WORD = {"hífen-fim", "término de palavra hifenizada", "hwe"}  # second half: already printed whole
LITERAL = {"s": "s", "ss": "ss", "--": "—", "-": "-", "!": "|", "=": "=", "espaço": " "}  # {{s}} is a long s
BREAK = {"gap", "nop", "dhr", "rule", "espaços", "separador", "linha horizontal", "parágrafo", "espaçamento vertical"}

DROP_TAGS = {"noinclude", "ref", "references", "pagequality", "pages", "section", "math", "gallery", "score",
             "timeline", "table", "templatestyles", "syntaxhighlight", "imagemap", "includeonly"}
BREAK_TAGS = {"br", "hr", "p", "div", "li", "dd", "dt"}
LINE = "\x1e"  # a line break that survives paragraph joining (verse, <br>, list item)
MARK = "\x1f"  # a list/indent marker: a new line in prose, nothing inside a poem (the newline is already there)
LINK_DROP_PREFIXES = ("ficheiro:", "file:", "imagem:", "image:", "categoria:", "category:", "media:")
INTERWIKI = re.compile(r"^:?[a-z]{2,3}(-[a-z]+)?:", re.I)  # [[en:...]] links to other languages


def _positional(t: Template, i: int) -> str | None:
    pos = [p for p in t.params if not p.showkey]
    return str(pos[i].value) if len(pos) > i else None


def _template(t: Template) -> str:
    name = str(t.name).strip().lower()
    if name.startswith("#") or name.startswith("subst:"):
        return ""
    if name in LITERAL:
        return LITERAL[name]
    if name in BREAK:
        return "\n"
    if name in DROP_WORD:
        return ""
    if name in WHOLE_WORD:
        value = _positional(t, 1) or _positional(t, 0)
    elif name in KEEP_SECOND:
        value = _positional(t, 1)
    elif name in KEEP_FIRST:
        value = _positional(t, 0)
    else:
        return ""  # navigation, tables, licences, headers...
    return wikitext_to_text(value) if value else ""


def _link(node: Wikilink) -> str:
    title = str(node.title).strip()
    if title.lower().lstrip(":").startswith(LINK_DROP_PREFIXES):
        return ""
    if node.text is not None:
        return _walk(node.text)
    if INTERWIKI.match(title) and not title.lower().lstrip(":").startswith(("w:", "wikipedia:", "autor:")):
        return ""  # a link to the same page in another language
    return title.lstrip(":").split(":")[-1].split("#")[0]


def _walk(code) -> str:
    out = []
    for node in code.nodes:
        if isinstance(node, Text):
            out.append(str(node))
        elif isinstance(node, Template):
            out.append(_template(node))
        elif isinstance(node, Wikilink):
            out.append(_link(node))
        elif isinstance(node, ExternalLink):
            out.append(_walk(node.title) if node.title else "")
        elif isinstance(node, Heading):
            out.append("\n\n" + _walk(node.title).strip() + "\n\n")
        elif isinstance(node, HTMLEntity):
            out.append(node.normalize())
        elif isinstance(node, Tag):
            name = str(node.tag).lower()
            if name in DROP_TAGS or node.wiki_markup == "{|":
                continue
            if node.wiki_markup in (":", "*", "#", ";"):
                out.append(MARK)  # list or indent marker at a line start; its text follows as a sibling
            elif name == "poem":
                out.append("\n\n" + _walk(node.contents).replace(MARK, "").strip("\n").replace("\n", LINE) + "\n\n")
            elif name == "br":
                out.append(LINE)
            elif name in BREAK_TAGS:
                out.append("\n\n" + (_walk(node.contents) if node.contents else "") + "\n\n")
            elif node.contents is not None:
                out.append(_walk(node.contents))
        elif isinstance(node, (Comment, Argument)):
            continue
        else:
            out.append(str(node))
    return "".join(out)


LIST_MARKERS = re.compile(r"^[ \t]*[:*#;]+[ \t]*", re.M)
MAGIC_WORDS = re.compile(r"__[A-ZÇÃÕÉÍ]+__")


def wikitext_to_text(text: str) -> str:
    """Plain text of a wikitext page, with paragraphs separated by a blank line."""
    plain = _walk(mw.parse(text))
    plain = html.unescape(MAGIC_WORDS.sub("", plain)).replace(" ", " ")
    plain = LIST_MARKERS.sub("", plain)
    # As in MediaWiki, a single newline is a space and a blank line ends a paragraph.
    paragraphs = [re.sub(r"\s*\n\s*", " ", block) for block in re.split(r"\n\s*\n", plain)]
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in "\n\n".join(paragraphs).replace(MARK, LINE).replace(LINE, "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
