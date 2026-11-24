"""Convert a FactSet CallStreet earnings-call transcript PDF to Markdown.

CLI (any transcript PDF in the same layout)::

    python -m factcheck.module.utils.tools <input.pdf> <output.md>

Page furniture is removed statistically, not by matching its wording:
  * running header/footer: text that recurs (digits ignored) in the top/bottom margin
    of at least half of the pages marks a header slot; every margin line in such a slot
    is dropped, so page numbers and the cover's "Total Pages" go too;
  * trailing legal text: after the last line set in the document's dominant font size,
    everything is dropped;
  * a page with almost no text left (the cover) becomes the title.

Structure assumptions (CallStreet layout): speaker blocks are separated by a line of
dots and look like ``Name / Title / [Q|A] / body``.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import fitz  # PyMuPDF

_SEPARATOR = re.compile(r"^\s*\.{10,}\s*$")
_SECTION = re.compile(r"^[A-Z][A-Z &/-]{5,}$")
_QA = {"Q", "A"}
_PARTICIPANT_SECTIONS = {"CORPORATE PARTICIPANTS", "OTHER PARTICIPANTS"}

_MARGIN = 0.12  # top/bottom fraction of the page height treated as margin
_RECURRING = 0.5  # share of pages a margin line must appear on to count as furniture
_SLOT = 3.0  # vertical resolution (points) used to compare margin positions
_COVER_RATIO = 0.1  # a page with less than this share of the median text is a cover


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    y: float
    x: float
    margin: bool
    bold: bool = False
    italic: bool = False


def _read_pages(doc: fitz.Document) -> list[list[_Line]]:
    pages: list[list[_Line]] = []
    for page in doc:
        height = page.rect.height
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                if not spans:
                    continue
                y = line["bbox"][1]
                font = spans[0]["font"].lower()
                flags = spans[0]["flags"]
                lines.append(
                    _Line(
                        text="".join(s["text"] for s in spans).rstrip(),
                        size=round(spans[0]["size"] * 2) / 2,
                        y=y,
                        x=line["bbox"][0],
                        bold="bold" in font or bool(flags & 16),
                        italic="italic" in font or bool(flags & 2),
                        margin=y < height * _MARGIN or y > height * (1 - _MARGIN),
                    )
                )
        pages.append(lines)
    return pages


def _slot(line: _Line) -> int:
    return round(line.y / _SLOT)


def _furniture_slots(pages: list[list[_Line]]) -> set[int]:
    """Margin positions that hold the same (digit-normalised) text on enough pages."""
    seen: Counter[tuple[str, int]] = Counter()
    for lines in pages:
        keys = {
            (re.sub(r"\d+", "#", ln.text.strip()), _slot(ln))
            for ln in lines
            if ln.margin and ln.text.strip()
        }
        seen.update(keys)
    need = max(2, _RECURRING * len(pages))
    return {slot for (_, slot), count in seen.items() if count >= need}


def _body_size(pages: list[list[_Line]]) -> float:
    chars: Counter[float] = Counter()
    for lines in pages:
        for ln in lines:
            chars[ln.size] += len(ln.text.strip())
    return chars.most_common(1)[0][0]


def _body_x(lines: list[_Line]) -> float:
    """Dominant left edge (by characters), rounded to 2 points."""
    chars: Counter[float] = Counter()
    for ln in lines:
        chars[round(ln.x / 2) * 2] += len(ln.text.strip())
    return chars.most_common(1)[0][0]


def _page_lines(doc: fitz.Document) -> tuple[list[_Line], list[_Line]]:
    """Return (cover lines, body lines) with headers, footers and trailer removed."""
    pages = _read_pages(doc)
    furniture = _furniture_slots(pages)
    pages = [[ln for ln in lines if not (ln.margin and _slot(ln) in furniture)] for lines in pages]

    weight = [sum(len(ln.text.strip()) for ln in lines) for lines in pages]
    typical = statistics.median(weight)
    cover: list[_Line] = []
    kept: list[_Line] = []
    for lines, size in zip(pages, weight, strict=True):
        if size < typical * _COVER_RATIO:
            cover.extend(ln for ln in lines if ln.text.strip())
        else:
            kept.extend(lines)

    body = _body_size(pages)
    left = _body_x(kept)

    def is_body(ln: _Line) -> bool:
        return bool(ln.text.strip()) and ln.size >= body and abs(ln.x - left) <= 3

    last = max((i for i, ln in enumerate(kept) if is_body(ln)), default=-1)
    return cover, kept[: last + 1]


def _blocks(lines: list[str]) -> list[list[str]]:
    """Split on dot-separator lines; drop blocks that are only whitespace."""
    blocks: list[list[str]] = [[]]
    for line in lines:
        if _SEPARATOR.match(line):
            blocks.append([])
        else:
            blocks[-1].append(line.rstrip())
    return [_strip_blank(b) for b in blocks if any(x.strip() for x in b)]


def _paragraphs(lines: list[str]) -> list[str]:
    """Join hard-wrapped lines into paragraphs; a blank line ends a paragraph."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        text = line.strip()
        if text:
            current.append(text)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _strip_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _participants(lines: list[str]) -> list[tuple[str, str]]:
    """Name/title line pairs inside a participants section."""
    texts = [line.strip() for line in lines if line.strip()]
    return list(zip(texts[::2], texts[1::2], strict=False))


def _speaker(lines: list[str], known: set[str]) -> tuple[str, str, str, list[str]] | None:
    """Parse ``Name / [Title] / [Q|A] / body``. Return (name, title, qa, body lines)."""
    lines = _strip_blank(lines)
    if len(lines) < 2:
        return None
    name = lines[0].strip()
    head = [x.strip() for x in lines[1:4]]
    qa_at = next((i for i, x in enumerate(head) if x in _QA), None)
    if name not in known and qa_at is None and lines[1].strip():
        return None
    if qa_at is not None:
        title = " ".join(x for x in head[:qa_at] if x)
        return name, title, head[qa_at], lines[qa_at + 2 :]
    if lines[1].strip():  # known name, title on the next line
        return name, lines[1].strip(), "", lines[2:]
    return name, "", "", lines[2:]  # e.g. "Unverified Participant" followed by a blank line


def _styled_markdown(cover: list[_Line], lines: list[_Line]) -> str:
    """Layout without separators (web page print): structure comes from font style.

    * the lines before the first body-sized line are the cover; its largest text is the title;
    * a bold line followed by an italic line is a speaker (name, title);
    * any other bold line is a section heading; plain lines straight after a heading
      (before any speaker) are list items;
    * a vertical gap wider than the usual line pitch starts a new paragraph.
    """
    body = _body_size([lines])
    first = next((i for i, ln in enumerate(lines) if ln.size == body), 0)
    cover = [*cover, *(ln for ln in lines[:first] if ln.text.strip())]
    lines = [ln for ln in lines[first:] if ln.text.strip()]

    pitches = [
        b.y - a.y
        for a, b in pairwise(lines)
        if not a.bold and not b.bold and 0 < b.y - a.y < 2 * a.size
    ]
    pitch = statistics.median(pitches) if pitches else 0.0

    out: list[str] = []
    if cover:
        top = max(ln.size for ln in cover)
        out.append("# " + " ".join(ln.text.strip() for ln in cover if ln.size == top))

    paragraph: list[str] = []
    in_list = False
    last_y: float | None = None

    def flush() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    i = 0
    while i < len(lines):
        ln = lines[i]
        text = ln.text.strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if ln.bold:
            flush()
            if nxt is not None and nxt.italic:
                out.append(f"### **{text}** · {nxt.text.strip()}")
                in_list = False
                i += 2
            else:
                out.append(f"## {text}")
                in_list = True
                i += 1
            last_y = None
            continue
        if in_list:
            out.append(f"- {text}")
        else:
            if paragraph and last_y is not None and ln.y - last_y > pitch * 1.3 > 0:
                flush()
            paragraph.append(text)
        last_y = ln.y
        i += 1
    flush()
    return "\n\n".join(_join_list_items(out)) + "\n"


def _join_list_items(blocks: list[str]) -> list[str]:
    """Merge consecutive ``- item`` blocks into one list block."""
    merged: list[str] = []
    for block in blocks:
        if block.startswith("- ") and merged and merged[-1].startswith("- "):
            merged[-1] += "\n" + block
        else:
            merged.append(block)
    return merged


def transcript_to_markdown(pdf_path: str | Path) -> str:
    """Extract one transcript PDF as Markdown, keeping the wording verbatim."""
    with fitz.open(pdf_path) as doc:
        cover, lines = _page_lines(doc)
    if not any(_SEPARATOR.match(ln.text) for ln in lines):
        return _styled_markdown(cover, lines)
    body = [ln.text for ln in lines]
    blocks = _blocks(body)

    known: set[str] = set()
    for block in blocks:
        first = block[0].strip() if block else ""
        if first in _PARTICIPANT_SECTIONS:
            known.update(name for name, _ in _participants(block[1:]))
    if not known:
        for block in blocks:
            known.update(name for name, _ in _participants(block))

    out: list[str] = []
    if cover:
        biggest = max(cover, key=lambda ln: ln.size)
        out.append(f"# {biggest.text.strip()}")
        out.append(" · ".join(ln.text.strip() for ln in cover if ln is not biggest))

    for block in blocks:
        lines = _strip_blank(block)
        head = lines[0].strip()
        if head in _PARTICIPANT_SECTIONS:
            out.append(f"## {head.title()}")
            out.append("\n".join(f"- {n} — {t}" for n, t in _participants(lines[1:])))
            continue
        if _SECTION.match(head):
            out.append(f"## {head.title()}")
            lines = _strip_blank(lines[1:])
            if not lines:
                continue
        parsed = _speaker(lines, known)
        if parsed is None:
            out.extend(_paragraphs(lines))  # e.g. "Operator: ..." lines
            continue
        name, title, qa, text = parsed
        label = " · ".join(x for x in (f"**{name}**", title, qa) if x)
        out.append(f"### {label}")
        out.extend(_paragraphs(text))

    return "\n\n".join(out) + "\n"


def convert(pdf_path: str | Path, out_path: str | Path) -> Path:
    """Write the transcript Markdown to ``out_path`` (UTF-8) and return the path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript_to_markdown(pdf_path), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdf", type=Path, help="CallStreet transcript PDF")
    parser.add_argument("out", type=Path, help="output Markdown path, e.g. refs/R3.md")
    args = parser.parse_args(argv)
    print(f"wrote {convert(args.pdf, args.out)}")


if __name__ == "__main__":
    main()
