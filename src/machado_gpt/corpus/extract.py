"""Extract clean text from one of the luxedo PDFs.

The PDF only stores positioned characters. This module rebuilds the text in two
layers:

1. `read_lines` turns a PDF into `Line` objects (text, position, font, size).
2. Pure functions decide what each line is and how lines join:
   - lines in a smaller font than the body are not part of the work: the
     edition header (first page) becomes metadata, everything else is a note
     and goes to a report;
   - a notes section in the body font (blocks opening with "[n] text") runs to
     the end of the book and is removed too; note markers "[n]" are stripped;
   - a table of contents ("ÍNDICE") is removed up to the first title it lists;
   - an epigraph (an indented run right after a title that ends with an
     uppercase attribution, or that repeats in several poems) is removed;
   - a vertical gap clearly larger than the normal line gap starts a block
     (paragraph or stanza);
   - at a page break there is no gap to measure. In prose a paragraph starts
     only when the previous line ends a sentence and is short or the next line
     opens a dialogue (`starts_new_block`); in verse the stanza continues unless
     the poem has a regular stanza length that is already complete
     (`split_verse_at_page_breaks`);
   - a heading (bold, or all in uppercase) stands alone, and so does a speech
     set inline in a play ("SEABRA — Queres...");
   - in plays, a character name on its own line is joined to its speech;
   - a block is verse when no line after the first starts in lowercase and its
     lines do not run to the right margin; verse keeps one line per verse, prose
     joins lines with spaces;
   - a word broken by a hyphen at the end of a prose line is resolved by the
     corpus itself (see `resolve_hyphen`);
   - finally, passages listed by hand in `data/catalog/exclusions.csv`
     (translations, poems by other authors) are removed (see `exclusions`).

Usage:
    uv run python -m machado_gpt.corpus.extract romance/domCasmurro.pdf
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pdfplumber

from machado_gpt.corpus import exclusions
from machado_gpt.corpus.download import PROJECT_ROOT, RAW_DIR

PDF_DIR = RAW_DIR / "pdf"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted"

# Points between letters above which a space is inferred. Measured on all PDFs:
# letters of a word sit below 1 pt apart (1.4-1.6 pt in letter-spaced passages),
# and the narrowest real space is 2.0 pt.
WORD_GAP = 1.8
SMALLER_FONT_MARGIN = 0.5  # points below the body size that count as "smaller"
PARAGRAPH_GAP_RATIO = 1.5  # gap / line gap above which a new block starts
SHORT_LINE_RATIO = 0.9  # a line ending before 90% of the right margin is "short"
FULL_LINE_RATIO = 0.9  # a line reaching 90% of the widest line is "full"
EPIGRAPH_INDENT = 60  # points to the right of the verse margin
EPIGRAPH_MIN_REPEATS = 3
MOTTO_SIMILARITY = 0.85
EPIGRAPH_MAX_LINES = 10  # the longest real epigraph has 8 lines; a scene has dozens
REGULAR_STANZA_SHARE = 0.8  # share of whole stanzas with the modal length
DIALOGUE_DASH = "—"
SENTENCE_END = tuple(".!?:;…\"”»)")
INDEX_TITLE = "ÍNDICE"

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
NOTE_MARKER = re.compile(r"\s*\[(\d+|[ivxlc]+)\]")  # [12] or [iii]
# An editor's comment inside a transcription: "[ilegível]", "[pouco legível: recusa?]",
# "[a palavra março está riscada ...]", "[?]". Removed entirely.
EDITORIAL_COMMENT = re.compile(r"\s*\[(?:[^\]]*(?:legível|riscad|palavra)[^\]]*|\?)\]")
# A word the editor restored: "vendido [no] México". The word stays, without brackets.
EDITORIAL_SUPPLY = re.compile(r"\[([a-zà-ÿ]+)\]")
EDITORIAL_LINE = re.compile(r"^\[[^\]]+\]$")  # "[Fragmento de III]": a modern editor's insertion
NOTE_START = re.compile(r"^\[\d+\]\s*\S")
# Dedications ("A F. X. DE NOVAIS", "Ao Sr. ...", "No álbum ...") and dates
# ("1862", "Corte, 5 de outubro") open many poems and letters; they are the
# author's text, not part of an epigraph. "A misérrima Dido" is not a
# dedication: after "A" a dedication names someone (uppercase, an initial, a title).
DEDICATION = re.compile(
    r"^(A|À|AO|AOS|ÀS|Ao|Aos|Às|O|À memória)\s+"
    r"([A-ZÀ-Ý]{2,}|[A-ZÀ-Ý]\.|Sr|Sra|St|Exm|Ilm|D\.|meu|minha|memória|seu|sua)"
    r"|^(No|Escritos no) álbum"
    r"|^\d{4}\.?$"
    r"|^([^,\d]+,\s*)?\d{1,2}º?\s+(de\s+)?[a-zç]+\.?(\s+(de\s+)?\d{4})?\.?$"
)
SELF_QUOTE = re.compile(r"^DO AUTOR\b")
# A speech set inline in a play: "SEABRA — Queres...", "D. HELENA (imitando-a) — Helena!"
SPEECH = re.compile(r"^(?:[A-ZÀ-Ý]\.\s)?[A-ZÀ-Ý]{2,}(?:\s[A-ZÀ-Ý.]+)*(?:\s\([^)]*\))?\s—\s")
ROMAN = re.compile(r"^[IVXLC]+\.?$")
SPEAKER_MIN_REPEATS = 3


@dataclass(frozen=True)
class Line:
    page: int
    top: float
    x0: float
    x1: float
    text: str
    size: float
    bold: bool


@dataclass
class Extraction:
    source: str
    paragraphs: list[str]
    edition_header: list[str]
    notes: list[str]
    index: list[str] = field(default_factory=list)
    epigraphs: list[list[str]] = field(default_factory=list)
    verse_blocks: int = 0
    hyphens: dict[str, list[str]] = field(default_factory=dict)
    excluded: list[dict[str, object]] = field(default_factory=list)
    to_review: list[dict[str, object]] = field(default_factory=list)


def normalize(text: str) -> str:
    """Unicode NFC plus typographic ligatures expanded; dashes are kept as they are."""
    for ligature, letters in LIGATURES.items():
        text = text.replace(ligature, letters)
    return unicodedata.normalize("NFC", text)


def read_lines(pdf_path: Path) -> list[Line]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            # Spaces (and no-break spaces) are inferred from the gap between
            # letters only. Some PDFs carry stray space glyphs inside words
            # ("P ERSONAGENS"), and the default tolerance (3 pt) glues words
            # whose gap is ~2.6 pt ("dadocomo").
            letters_only = page.filter(
                lambda obj: not (obj.get("object_type") == "char" and obj.get("text") in (" ", "\xa0"))
            )
            for raw in letters_only.extract_text_lines(return_chars=True, x_tolerance=WORD_GAP):
                chars = [c for c in raw["chars"] if c["text"].strip()]
                text = raw["text"].strip()
                if not chars or not text:
                    continue
                size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
                bold = all("Bold" in c["fontname"] for c in chars)
                lines.append(
                    Line(page_number, raw["top"], raw["x0"], raw["x1"], normalize(text), size, bold)
                )
    return lines


def body_size(lines: list[Line]) -> float:
    """The size used by most characters is the size of the body text."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0]


def split_small_print(lines: list[Line], body: float) -> tuple[list[Line], list[str], list[str]]:
    """Separate body lines from the edition header and from notes (smaller font).

    The edition header is the first contiguous run of small print on page 1 (the
    title above it is in the body size). Any other small print is a note.
    """
    body_lines, header, notes = [], [], []
    header_done = False
    for line in lines:
        small = line.size < body - SMALLER_FONT_MARGIN
        if not small:
            body_lines.append(line)
            header_done = header_done or bool(header)
        elif line.page == 1 and not header_done:
            header.append(line.text)
        else:
            notes.append(line.text)
    return body_lines, header, notes


def split_notes_section(lines: list[Line]) -> tuple[list[Line], list[str]]:
    """Cut the notes section in body font: from the first "[n] text" line to the end."""
    for i, line in enumerate(lines):
        if NOTE_START.match(line.text):
            return lines[:i], [l.text for l in lines[i:]]
    return lines, []


def strip_note_markers(lines: list[Line]) -> list[Line]:
    """Remove "[n]" and "[iii]" references and lines that are an editor's
    bracketed insertion; drop lines that held nothing else."""
    out = []
    for line in lines:
        text = NOTE_MARKER.sub("", line.text).strip()
        if text and not EDITORIAL_LINE.match(text):
            out.append(replace(line, text=text) if text != line.text else line)
    return out


def is_title(line: Line) -> bool:
    """Bold lines, or lines whose letters are all uppercase (e.g. 'O PÃO D'AÇÚCAR', 'III')."""
    letters = [c for c in line.text if c.isalpha()]
    return line.bold or (len(letters) >= 2 and all(c.isupper() for c in letters))


def is_attribution(line: Line, right_aligned: bool) -> bool:
    """The author line closing an epigraph.

    It carries an uppercase word ('MOLIÈRE.', 'CAMÕES — Elegias'), but is not a
    dedication ('A FRANCISCO GONÇALVES BRAGA'). In prose, where epigraphs are set
    flush right, a short name is enough ('Dante').
    """
    if DEDICATION.match(line.text):
        return False
    if any(len(w) >= 3 and w.isupper() for w in re.findall(r"[^\W\d_]+", line.text)):
        return True
    words = line.text.split()
    return right_aligned and 0 < len(words) <= 3 and line.text[:1].isupper() and len(line.text) <= 30


def similar(a: str, b: str) -> bool:
    """Same text up to small typing variations (punctuation, a missing word)."""
    key_a = re.sub(r"[^a-z]", "", a.lower())
    key_b = re.sub(r"[^a-z]", "", b.lower())
    return difflib.SequenceMatcher(None, key_a, key_b).ratio() >= MOTTO_SIMILARITY


def split_index(lines: list[Line]) -> tuple[list[Line], list[str]]:
    """Remove every 'ÍNDICE' (of the book, or of each story) and its entries, up
    to where the first listed title appears again: that second occurrence is
    the real title."""
    out: list[Line] = []
    removed: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].text.strip() == INDEX_TITLE and i + 1 < len(lines):
            first = lines[i + 1].text
            end = next((k for k in range(i + 2, len(lines)) if lines[k].text == first), None)
            if end is not None:
                removed += [l.text for l in lines[i:end]]
                i = end
                continue
        out.append(lines[i])
        i += 1
    return out, removed


def verse_margin(lines: list[Line]) -> float:
    """Left margin of the text: the most common x0."""
    return Counter(round(l.x0) for l in lines).most_common(1)[0][0]


def indented_run_after(lines: list[Line], i: int, margin: float) -> list[Line]:
    """Non-bold lines right after lines[i], indented beyond the epigraph threshold.

    Poetry sets epigraphs flush left at a deep indent; prose sets them flush
    right. Both start well to the right of the text margin. A bold line (a
    chapter heading) ends the run.
    """
    run = []
    for line in lines[i + 1 :]:
        # The run may start on the next page when the title closes a page.
        if line.bold or line.x0 <= margin + EPIGRAPH_INDENT:
            break
        run.append(line)
    return run


def split_epigraphs(lines: list[Line]) -> tuple[list[Line], list[list[str]]]:
    """Remove epigraphs: indented runs right after a title that end with an
    attribution (SHAKESPEARE, Dante) or repeat in several poems (a motto).

    Dates and dedications at the start of a run are the author's own text and
    stay; a run opening with "(" is a stage direction and stays.
    """
    margin = verse_margin(lines)
    widest = max(l.x1 for l in lines)
    runs: dict[int, list[Line]] = {}
    for i, line in enumerate(lines):
        if is_title(line):
            run = indented_run_after(lines, i, margin)
            kept = 0
            while kept < len(run) and DEDICATION.match(run[kept].text):
                kept += 1
            if run[kept:]:
                runs[i + kept] = run[kept:]
    texts = {i: " ".join(l.text for l in run) for i, run in runs.items()}

    drop: set[int] = set()
    inside_long: set[int] = set()
    epigraphs = []
    for i, run in runs.items():
        if i in drop or i in inside_long or run[0].text.startswith("("):
            continue  # inside a run already decided, or a stage direction
        if len(run) > EPIGRAPH_MAX_LINES:
            # An indented scene or letter of the work itself; speaker names
            # inside it must not start shorter runs of their own.
            inside_long.update(range(i + 1, i + 1 + len(run)))
            continue
        # A quotation has mixed-case text; a run of uppercase lines is an
        # inscription or a heading of the work itself (e.g. an epitaph).
        quoted = any(not is_title(l) for l in run[:-1])
        right_aligned = all(l.x1 >= FULL_LINE_RATIO * widest for l in run[:-1])
        attributed = len(run) >= 2 and quoted and is_attribution(run[-1], right_aligned)
        if SELF_QUOTE.match(run[-1].text):
            continue  # Machado quoting himself
        motto = len(run) >= 2 and (
            sum(similar(texts[i], other) for other in texts.values()) >= EPIGRAPH_MIN_REPEATS
        )
        if attributed or motto:
            epigraphs.append([l.text for l in run])
            drop.update(range(i + 1, i + 1 + len(run)))
    return [l for k, l in enumerate(lines) if k not in drop], epigraphs


def line_gap(lines: list[Line]) -> float:
    """Vertical distance between consecutive lines of the same paragraph.

    Measured after full lines (running to the right margin), which can only be
    followed by a continuation line. The plain most common gap fails in plays:
    short speeches separated by paragraph gaps outnumber continuation lines
    (24 pt instead of 12 pt in 9 of the 10 plays). Falls back to the most common
    gap when there are no full lines (e.g. verse).
    """
    same_page = [(a, b) for a, b in zip(lines, lines[1:]) if a.page == b.page and b.top > a.top]
    margin = right_margin(lines)
    after_full = Counter(
        round(b.top - a.top) for a, b in same_page if a.x1 >= FULL_LINE_RATIO * margin and not is_title(a)
    )
    if after_full:
        return after_full.most_common(1)[0][0]
    return Counter(round(b.top - a.top) for a, b in same_page).most_common(1)[0][0]


def right_margin(lines: list[Line]) -> float:
    """Where full lines end: the 95th percentile of the right edge of non-bold lines."""
    edges = sorted(line.x1 for line in lines if not line.bold)
    return edges[int(0.95 * (len(edges) - 1))]


def starts_new_block(prev: Line, line: Line, gap: float, margin: float) -> bool:
    if is_title(line) or is_title(prev) or SPEECH.match(line.text):
        return True
    if line.page != prev.page:
        # No gap to measure across pages. Measured on Dom Casmurro, a line that
        # ends a sentence and is short always closed a paragraph; a full line
        # did so only when the next line opened a dialogue.
        if not prev.text.endswith(SENTENCE_END):
            return False
        short = prev.x1 < SHORT_LINE_RATIO * margin
        return short or line.text.startswith(DIALOGUE_DASH)
    return (line.top - prev.top) > PARAGRAPH_GAP_RATIO * gap


def group_blocks(lines: list[Line], gap: float, margin: float) -> list[list[Line]]:
    blocks: list[list[Line]] = []
    for prev, line in zip([None, *lines], lines):
        if prev is None or starts_new_block(prev, line, gap, margin):
            blocks.append([])
        blocks[-1].append(line)
    return blocks


def group_segments(lines: list[Line], gap: float) -> list[list[Line]]:
    """Split only where the split is certain: titles and vertical gaps.

    Page breaks stay inside segments; `build_blocks` decides them, because the
    right signal differs between prose and verse.
    """
    segments: list[list[Line]] = []
    for prev, line in zip([None, *lines], lines):
        certain = (
            prev is None
            or is_title(line)
            or is_title(prev)
            or bool(SPEECH.match(line.text))
            or (line.page == prev.page and (line.top - prev.top) > PARAGRAPH_GAP_RATIO * gap)
        )
        if certain:
            segments.append([])
        segments[-1].append(line)
    return segments


def has_page_break(segment: list[Line]) -> bool:
    return any(a.page != b.page for a, b in zip(segment, segment[1:]))


def regular_stanza_size(segments: list[list[Line]], widest: float) -> int | None:
    """Most common stanza length in a poem, if at least 80% of its whole
    stanzas (those not crossing a page) have it; otherwise None."""
    whole = [len(s) for s in segments if not has_page_break(s) and is_verse(s, widest)]
    if len(whole) < 2:
        return None
    size, count = Counter(whole).most_common(1)[0]
    return size if count / len(whole) >= REGULAR_STANZA_SHARE else None


def split_verse_at_page_breaks(segment: list[Line], size: int | None) -> list[list[Line]]:
    """Decide stanza breaks at page breaks inside a verse segment.

    Measured on the poetry books, punctuation is a weak signal in verse (only 42%
    of verses ending in punctuation close a stanza), so the stanza continues
    unless a regular poem says the stanza is already complete.
    """
    blocks = [[segment[0]]]
    for prev, line in zip(segment, segment[1:]):
        if line.page != prev.page and size and len(blocks[-1]) % size == 0:
            blocks.append([])
        blocks[-1].append(line)
    return blocks


def is_verse_book(segments: list[list[Line]], widest: float) -> bool:
    """Most lines of the book are verse, judged on segments within one page.

    In a novel, two short dialogue lines around a page break look like verse;
    the verse rule for page breaks only applies to books that are mostly verse.
    """
    whole = [s for s in segments if not has_page_break(s) and not is_title(s[0])]
    total = sum(len(s) for s in whole)
    verse = sum(len(s) for s in whole if is_verse(s, widest))
    return total > 0 and verse / total >= 0.5


def build_blocks(lines: list[Line], gap: float, margin: float, widest: float) -> list[list[Line]]:
    """Group lines into paragraphs and stanzas, deciding page breaks per kind."""
    segments = group_segments(lines, gap)
    poems: list[list[list[Line]]] = [[]]
    for segment in segments:
        if is_title(segment[0]):
            poems.append([])
        else:
            poems[-1].append(segment)
    size_of = {id(seg): regular_stanza_size(poem, widest) for poem in poems for seg in poem}
    verse_book = is_verse_book(segments, widest)

    blocks: list[list[Line]] = []
    for segment in segments:
        if verse_book and has_page_break(segment) and is_verse(segment, widest):
            blocks += split_verse_at_page_breaks(segment, size_of.get(id(segment)))
        else:
            blocks += group_blocks(segment, gap, margin)
    return blocks


def is_verse(block: list[Line], widest: float) -> bool:
    """Verse: every line after the first starts without lowercase, and the lines
    do not all run to the right margin (measured on novels and poetry books)."""
    if len(block) < 2:
        return False
    continues_lowercase = any(l.text[:1].islower() for l in block[1:])
    all_full = all(l.x1 >= FULL_LINE_RATIO * widest for l in block[:-1])
    return not continues_lowercase and not all_full


def word_forms(blocks: list[list[Line]]) -> Counter[str]:
    """Words seen inside lines (never across a line break), lowercased."""
    forms: Counter[str] = Counter()
    for block in blocks:
        for line in block:
            words = WORD.findall(line.text.lower())
            if line.text.endswith("-") and words:
                words = words[:-1]  # the broken word is not evidence
            forms.update(words)
    return forms


def resolve_hyphen(left: str, right: str, forms: Counter[str]) -> str:
    """Decide how 'left-' at a line end joins with 'right...' on the next line.

    Returns 'keep' (real hyphen, e.g. admiti-me), 'join' (word split by the line
    break, e.g. cava-lo -> cavalo) or 'review' when the corpus has no evidence.
    """
    head = WORD.findall(left.lower())
    tail = WORD.findall(right.lower())
    if not head or not tail:
        return "review"
    hyphenated = f"{head[-1]}-{tail[0]}"
    joined = f"{head[-1]}{tail[0]}"
    if forms[hyphenated] > 0:
        return "keep"
    if forms[joined] > 0:
        return "join"
    return "review"


def join_prose(texts: list[str], forms: Counter[str], log: dict[str, list[str]]) -> str:
    text = texts[0]
    for line in texts[1:]:
        if text.endswith("-") and not text.endswith(" -"):
            decision = resolve_hyphen(text[:-1], line, forms)
            sample = f"{text[-20:]}|{line[:20]}"
            log.setdefault(decision, []).append(sample)
            # Unresolved cases keep the hyphen: these PDFs come from HTML, which
            # does not hyphenate words automatically.
            text = text[:-1] + line if decision == "join" else text + line
        else:
            text = f"{text} {line}"
    return text


def join_block(block: list[Line], widest: float, forms: Counter[str], log: dict[str, list[str]]) -> str:
    texts = [l.text for l in block]
    if is_verse(block, widest):
        return "\n".join(texts)
    return join_prose(texts, forms, log)


def clean_editorial(paragraph: str) -> str:
    """Drop an editor's comments and unbracket the words an editor restored."""
    return EDITORIAL_SUPPLY.sub(r"\1", EDITORIAL_COMMENT.sub("", paragraph))


def is_speaker_name(block: str) -> bool:
    """A short uppercase line that could name a character ('VALENTIM', 'D. LEONOR')."""
    letters = [c for c in block if c.isalpha()]
    return (
        "\n" not in block
        and len(block.split()) <= 4
        and len(letters) >= 2
        and all(c.isupper() for c in letters)
        and not ROMAN.match(block)
    )


def attach_speaker_names(paragraphs: list[str]) -> list[str]:
    """In plays set with the name on its own line, join the name to its speech
    with a line break ('VALENTIM\nAh! És tu?').

    A name counts as a speaker when it stands alone at least 3 times in the play;
    headings ('ATO ÚNICO', 'PERSONAGENS') appear once and stay alone.
    """
    counts = Counter(p for p in paragraphs if is_speaker_name(p))
    speakers = {name for name, n in counts.items() if n >= SPEAKER_MIN_REPEATS}
    def joinable(block: str) -> bool:
        return not is_speaker_name(block) and not block.lower().startswith(("cena ", "ato "))

    def is_direction(block: str) -> bool:
        return block.startswith("(") and block.endswith(")")

    out: list[str] = []
    i = 0
    while i < len(paragraphs):
        current = paragraphs[i]
        i += 1
        if current in speakers:
            # Absorb stage directions set apart, then the speech itself.
            while i < len(paragraphs) and joinable(paragraphs[i]):
                current = f"{current}\n{paragraphs[i]}"
                i += 1
                if not is_direction(current.split("\n")[-1]):
                    break
        out.append(current)
    return out


def extract(pdf_path: Path) -> Extraction:
    lines = read_lines(pdf_path)
    body, header, notes = split_small_print(lines, body_size(lines))
    body, body_notes = split_notes_section(body)
    body = strip_note_markers(body)
    body, index = split_index(body)
    body, epigraphs = split_epigraphs(body)

    widest = max(l.x1 for l in body)
    blocks = build_blocks(body, line_gap(body), right_margin(body), widest)
    forms = word_forms(blocks)
    log: dict[str, list[str]] = {}
    paragraphs = [clean_editorial(join_block(block, widest, forms, log)) for block in blocks]
    source = str(pdf_path.relative_to(PDF_DIR))
    if source.startswith("teatro/"):
        paragraphs = attach_speaker_names(paragraphs)
    paragraphs, excluded, to_review = exclusions.apply(
        str(Path(source).with_suffix("")), paragraphs, exclusions.load()
    )
    return Extraction(
        source=source,
        paragraphs=paragraphs,
        edition_header=header,
        notes=notes + body_notes,
        index=index,
        epigraphs=epigraphs,
        verse_blocks=sum(is_verse(b, widest) for b in blocks),
        hyphens=log,
        excluded=excluded,
        to_review=to_review,
    )


def save(result: Extraction) -> Path:
    stem = OUT_DIR / Path(result.source).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    text_path = stem.with_suffix(".txt")
    text_path.write_text("\n\n".join(result.paragraphs) + "\n", encoding="utf-8")
    meta = asdict(result)
    meta.pop("paragraphs")
    stem.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path


def main(argv: list[str]) -> None:
    for name in argv:
        result = extract(PDF_DIR / name)
        path = save(result)
        counts = {k: len(v) for k, v in result.hyphens.items()}
        print(
            f"{result.source}: {len(result.paragraphs)} blocks ({result.verse_blocks} verse), "
            f"header {len(result.edition_header)}, notes {len(result.notes)}, "
            f"index {len(result.index)}, epigraphs {len(result.epigraphs)}, "
            f"excluded {len(result.excluded)}, to review {len(result.to_review)}, "
            f"hyphens {counts} -> {path.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
