"""Turn a clipped web article (Obsidian Web Clipper Markdown) into the inputs of `mark`:
body.md plus one reference file per link in the article's "reference materials" list."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from pydantic import BaseModel

from .fetch import FetchError, fetch_text, youtube_id
from .io import MIN_REF_CHARS, ref_problem, write_manual_list
from .models import ManualRef, RefResult

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---[ \t]*\n", re.DOTALL)
_REF_HEADER = re.compile(
    r"參考資料|参考资料|references?\b|sources?\b|further reading", re.IGNORECASE
)
_HEADER_MAX_CHARS = 40  # a real header is short; a prose sentence that mentions "sources" is not
_BULLET = re.compile(r"\s*(?:[-*+]|\d{1,3}[.)])\s+(.*)")
# [text](url) where text may hold an image and url one level of parentheses; or a bare / <auto> URL
_MD_LINK = re.compile(
    r"(?<!!)\[((?:[^\[\]]|!\[[^\]]*\]\([^)]*\))*)\]"
    r"\((https?://(?:[^()\s]|\([^()\s]*\))+)(?:\s+\"[^\"]*\")?\)"
)
_BARE_URL = re.compile(r"https?://[^\s<>\])]+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HEADING = re.compile(r"\s{0,3}#{1,6}\s+\S")
_DATE_LINE = re.compile(r"(發布日期|发布日期|published|date)\s*[｜|:：]", re.IGNORECASE)
_HR = re.compile(r"\s*-{3,}\s*")
_TITLE_FIELD = re.compile(r'^title:\s*"?(.*?)"?\s*$', re.MULTILINE)
_TRACKING_PARAM = re.compile(r"utm_|fbclid$|gclid$|igshid$|si$", re.IGNORECASE)
_PREAMBLE_MAX_CHARS = 600  # text before the list is page furniture only if it is this short

Fetcher = Callable[[str], str]


class Link(BaseModel):
    title: str
    url: str


class Clipping(BaseModel):
    body: str
    refs: list[Link]


def _is_header(line: str) -> bool:
    if _BULLET.match(line):
        return False
    clean = line.strip().lstrip("#").strip().strip("*_ :：")
    return 0 < len(clean) <= _HEADER_MAX_CHARS and bool(_REF_HEADER.search(clean))


def _links_in(item: str) -> list[Link]:
    """Links of one list item: markdown links, then any URL left over (bare or <autolink>)."""
    links = [Link(title=_IMAGE.sub("", t).strip(), url=u) for t, u in _MD_LINK.findall(item)]
    rest = _IMAGE.sub("", _MD_LINK.sub("", item))
    links.extend(Link(title="", url=u.rstrip(".,;:!?'\"")) for u in _BARE_URL.findall(rest))
    return links


def _split_references(lines: list[str]) -> tuple[list[Link], int, int] | None:
    """Find the line that introduces the reference list and read the list items under it. Returns
    the links, the index of the header and the index of the first line after the list, or None if
    there is no such list."""
    for i, line in enumerate(lines):
        if not _is_header(line):
            continue
        links: list[Link] = []
        end = i + 1
        for j in range(i + 1, len(lines)):
            if not lines[j].strip():
                continue  # blank lines may sit between items
            item = _BULLET.match(lines[j])
            if not item:
                break
            links.extend(_links_in(item.group(1)))
            end = j + 1
        if links:
            return links, i, end
    return None


def _strip_furniture(lines: list[str], title: str) -> list[str]:
    """Drop leading blank lines, the title heading, the date line and horizontal rules."""
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and title and _HEADING.match(lines[start]) and title in lines[start]:
        start += 1
    while start < len(lines) and (
        not lines[start].strip() or _DATE_LINE.match(lines[start]) or _HR.fullmatch(lines[start])
    ):
        start += 1
    return lines[start:]


def parse_clipping(text: str) -> Clipping:
    """Split a clipped article into its body and its reference links.

    The reference list is removed from the body. Text before it is dropped as page furniture
    (title, date, intro) when it is short, i.e. the list sits at the top; when the list is at the
    end or in the middle of a long article, the text on both sides is kept. Without a reference
    list only the frontmatter, the title heading and the date line are dropped."""
    text = text.replace("\r\n", "\n")
    front = _FRONTMATTER.match(text)
    title_match = _TITLE_FIELD.search(front.group(0)) if front else None
    title = title_match.group(1).split(" | ")[0].strip() if title_match else ""
    lines = text[front.end() :].split("\n") if front else text.split("\n")

    found = _split_references(lines)
    if found:
        refs, header, end = found
        before = _strip_furniture(lines[:header], title)
        if len("\n".join(before).strip()) < _PREAMBLE_MAX_CHARS:
            before = []
        kept = before + _strip_furniture(lines[end:], "")
    else:
        refs, kept = [], _strip_furniture(lines, title)
    body = "\n".join(kept).strip()
    return Clipping(body=body + "\n", refs=_dedupe(refs))


def _url_key(url: str) -> str:
    """Identity of a link for de-duplication: same video / same page without tracking noise."""
    video = youtube_id(url)
    if video:
        return f"youtube:{video}"
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAM.match(k)]
    host = (parts.hostname or "").lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}?{urlencode(query)}"


def _dedupe(links: list[Link]) -> list[Link]:
    seen: dict[str, Link] = {}
    for link in links:
        seen.setdefault(_url_key(link.url), link)
    return list(seen.values())


def ref_id(index: int) -> str:
    """Short, file-name-safe id (R1, R2, ...); the url and title live in sources.json."""
    return f"R{index}"


def prep(
    clipping_path: Path,
    out_dir: Path,
    fetcher: Fetcher = fetch_text,
    *,
    download: bool = True,
    min_chars: int = MIN_REF_CHARS,
    on_progress: Callable[[str], None] = lambda _msg: None,
) -> tuple[Clipping, list[RefResult]]:
    """Write out_dir/body.md, out_dir/refs/<id>.md for each downloaded reference, and
    out_dir/sources.json (id → title, url, status). A download that failed or returned an unusable
    page (error / login wall, see `ref_problem`) is not saved; it is listed in
    out_dir/manual_refs.md instead and does not stop the others. A download run first deletes the
    old refs/R*.md, so run it before saving hand-fetched references."""
    clipping = parse_clipping(clipping_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "body.md").write_text(clipping.body, encoding="utf-8")
    refs_dir = out_dir / "refs"
    if download and refs_dir.is_dir():
        # ids are positional: files of an earlier run would sit under the wrong id (or a failed one)
        for stale in refs_dir.glob("R*.md"):
            stale.unlink()
    results: list[RefResult] = []
    for n, link in enumerate(clipping.refs, start=1):
        rid = ref_id(n)
        if not download:
            results.append(RefResult(id=rid, title=link.title, url=link.url, file=None))
            continue
        on_progress(f"[{n}/{len(clipping.refs)}] {link.url}")
        try:
            text = fetcher(link.url)
        except Exception as exc:  # one bad link must not lose the other downloads
            error = str(exc) if isinstance(exc, FetchError) else f"{type(exc).__name__}: {exc}"
            results.append(
                RefResult(id=rid, title=link.title, url=link.url, file=None, error=error)
            )
            continue
        problem = ref_problem(text, min_chars)
        if problem:  # an error / login page saved as a reference would silently mean "no source"
            results.append(
                RefResult(id=rid, title=link.title, url=link.url, file=None, error=problem)
            )
            continue
        refs_dir.mkdir(exist_ok=True)
        (refs_dir / f"{rid}.md").write_text(text.strip() + "\n", encoding="utf-8")
        results.append(RefResult(id=rid, title=link.title, url=link.url, file=f"refs/{rid}.md"))
    if download:
        write_manual_list(
            [
                ManualRef(id=r.id, title=r.title, url=r.url, reason=r.error or "not downloaded")
                for r in results
                if r.file is None
            ],
            out_dir / "manual_refs.md",
        )
    (out_dir / "sources.json").write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return clipping, results
