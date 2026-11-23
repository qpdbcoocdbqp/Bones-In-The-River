from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from functools import lru_cache

from factcheck.module.models import Block, Sentence

MIN_QUOTE_CHARS = 12  # a normalized quote shorter than this is too short to prove anything

_LINK_TARGET = re.compile(r"\]\([^)\n]*\)")  # the (url) part of a Markdown link
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)\n]*\)")
_TAG = re.compile(r"\[[^\]\n]{1,30}\]")  # tags in captions such as [music], [applause]
_HR = re.compile(r"-{3,}|\*{3,}|_{3,}")
_HEADING = re.compile(r"\s{0,3}#{1,6}\s+\S")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")
# sentence = non-greedy text + terminator (CJK/Latin punctuation plus closing quotes/brackets, or
# "period + space + capital", or end of string)
_SENTENCE = re.compile(
    r".+?(?:[。！？!?]+[」』”’）)】\"']*|\.(?=\s+[A-Z])|$)",
    re.DOTALL,
)


def plain(text: str) -> str:
    """Remove Markdown link syntax, keeping the link text; collapse whitespace."""
    return " ".join(_MD_LINK.sub(r"\1", text).split())


def split_sentences(paragraph: str) -> list[str]:
    """Rule-based sentence splitting. Link (url) parts are masked first so a ? inside a URL is not
    taken as a sentence end.

    Known limitation: a Latin abbreviation followed by a capital letter (e.g. "e.g. Apple") is split
    wrongly.
    """
    stash: list[str] = []

    def hide(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\x00{len(stash) - 1}\x00"

    masked = _LINK_TARGET.sub(hide, paragraph)
    parts = (m.group(0).strip() for m in _SENTENCE.finditer(masked))
    return [_PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], p) for p in parts if p]


def parse_body(body: str) -> list[Block]:
    """Body -> blocks. Headings and rules are kept as-is (verbatim); the rest are paragraphs split
    into sentences."""
    blocks: list[Block] = []
    paragraph_no = 0
    for kind, text in _blocks(body):
        if kind == "verbatim":
            blocks.append(Block(kind="verbatim", text=text))
            continue
        paragraph_no += 1
        sentences = [
            Sentence(id=f"p{paragraph_no}s{n}", text=s, claim=plain(s))
            for n, s in enumerate(split_sentences(text), start=1)
        ]
        blocks.append(Block(kind="paragraph", text=text, sentences=sentences))
    return blocks


def _blocks(body: str) -> Iterator[tuple[str, str]]:
    """Split into blocks on blank lines, then extract heading lines (`# Title`, space required
    after #) and rules as separate verbatim blocks.

    If body text follows a heading with no blank line, that text is still a paragraph and is not
    swallowed by the heading; `#hashtag` is not a heading.
    """
    for raw in re.split(r"\n\s*\n", body.strip()):
        paragraph: list[str] = []
        for line in raw.splitlines():
            if _HEADING.match(line) or _HR.fullmatch(line.strip()):
                if paragraph:
                    yield "paragraph", "\n".join(paragraph).strip()
                    paragraph = []
                yield "verbatim", line.strip()
            else:
                paragraph.append(line)
        if "".join(paragraph).strip():
            yield "paragraph", "\n".join(paragraph).strip()


_NOTHING = {"", "none", "nothing", "n/a", "na", "null", "無", "没有", "沒有", "無缺漏", "沒有缺漏"}


def means_nothing(text: str) -> bool:
    """Field values with which the model says "nothing", e.g. "none" or "無"."""
    return text.strip().strip("。.!！\"'「」 ").casefold() in _NOTHING


def chunk_spans(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Split long text into (start, end) ranges; a single range if within max_chars. Prefers
    breaking at whitespace."""
    if overlap >= max_chars // 2:
        raise ValueError("overlap must be less than half of max_chars")
    if len(text) <= max_chars:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + max_chars // 2, end)
            if cut > start:
                end = cut
        spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return spans


@lru_cache(maxsize=8)
def _fold(text: str) -> tuple[str, tuple[int, ...]]:
    """Normalize: NFKC, casefold, keep only alphanumerics, collapse whitespace to one space, skip
    [tag]s.

    Returns (normalized string, original-text index of each normalized character), so the quote
    range can be restored.
    """
    skipped = {i for m in _TAG.finditer(text) for i in range(m.start(), m.end())}
    chars: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, original in enumerate(text):
        if i in skipped:
            pending_space = True
            continue
        for c in unicodedata.normalize("NFKC", original).casefold():
            if c in "'’":  # contractions (don't / dont) count as one word, not a separator
                continue
            if c.isalnum():
                if pending_space and chars:
                    chars.append(" ")
                    index.append(i)
                chars.append(c)
                index.append(i)
                pending_space = False
            else:
                pending_space = True
    return "".join(chars), tuple(index)


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Find the quote in text (ignoring case, punctuation, whitespace differences) and return the
    character range in the original.

    Returns None if not found, too short, or found multiple times in text (cannot tell which one).
    """
    folded_quote, _ = _fold(quote)
    if len(folded_quote) < MIN_QUOTE_CHARS:
        return None
    folded_text, index = _fold(text)
    pos = folded_text.find(folded_quote)
    if pos < 0 or folded_text.find(folded_quote, pos + 1) >= 0:
        return None
    return index[pos], index[pos + len(folded_quote) - 1] + 1
