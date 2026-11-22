from __future__ import annotations

import contextlib
import re
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Playwright

_DROP_TAGS = ("script", "style", "noscript", "template", "nav", "footer", "svg")
# <header> / <aside> are page furniture, but inside an <article> they hold lead text and
# pull-quotes; <form> is never dropped: some CMSs (ASP.NET) wrap the whole page in one
_DROP_OUTSIDE_ARTICLE = ("header", "aside")
_BLOCK_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre",
    "div", "section", "td", "th", "dd", "dt", "figcaption",
]  # fmt: skip
_TRANSCRIPT_LANGS = ["en", "zh-Hant", "zh-TW", "zh-Hans", "zh"]
_YOUTUBE_HOSTS = ("youtube.com", "music.youtube.com", "youtube-nocookie.com")
_SNIPPETS_PER_PARAGRAPH = 20
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
}
_TEXT_TYPES = ("text/", "application/xhtml", "application/xml")


class FetchError(Exception):
    """A reference could not be downloaded or converted to text."""


def youtube_id(url: str) -> str | None:
    """Video id of a youtube.com / youtu.be link, or None for any other URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return parsed.path.strip("/") or None
    if host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        match = re.match(r"/(?:embed|shorts|live)/([\w-]+)", parsed.path)
        return match.group(1) if match else None
    return None


def html_to_text(html: str | bytes, encoding: str | None = None) -> str:
    """Main text of a web page: the largest <article> / <main> element if present, else <body>;
    one paragraph per innermost block (heading, paragraph, list item, table cell, <br>-separated
    line). Bytes are decoded by the charset in `encoding` or in the page's own <meta> tag."""
    soup = BeautifulSoup(
        html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None
    )
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for tag in soup(_DROP_OUTSIDE_ARTICLE):
        if not tag.find_parent(["article", "main"]):
            tag.decompose()
    for br in soup("br"):
        br.replace_with("\n")
    # a page can hold several <article> (teaser cards) next to the real one: take the largest
    candidates = [c for c in soup.find_all(["article", "main"]) if isinstance(c, Tag)]
    root = max(candidates, key=lambda c: len(c.get_text()), default=soup.body or soup)
    paragraphs: list[str] = []
    for el in root.find_all(_BLOCK_TAGS):
        if el.find(_BLOCK_TAGS):
            continue  # only innermost blocks: a <div> around <p>s would repeat their text
        paragraphs.extend(" ".join(line.split()) for line in el.get_text(" ").split("\n"))
    # nested duplicates cannot occur, but identical boilerplate lines may
    text = "\n\n".join(dict.fromkeys(p for p in paragraphs if p))
    return text or "\n\n".join(line for line in root.get_text("\n").splitlines() if line.strip())


def fetch_page(url: str, timeout: float = 30.0) -> str:
    try:
        response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise FetchError(f"{url}: {exc}") from exc
    kind = response.headers.get("content-type", "text/html").split(";")[0].strip().lower()
    if not kind.startswith(_TEXT_TYPES):
        raise FetchError(f"{url}: not a web page ({kind}); download it by hand")
    if kind == "text/plain":
        return response.text
    # the bytes go to the parser so a charset declared only in <meta> (Big5, GBK, ...) is honoured
    return html_to_text(response.content, response.charset_encoding)


def _paragraphs(lines: list[str]) -> str:
    step = _SNIPPETS_PER_PARAGRAPH
    return "\n\n".join(" ".join(lines[i : i + step]) for i in range(0, len(lines), step))


def fetch_transcript(video_id: str) -> str:
    """Captions of a YouTube video (manual or auto-generated) as paragraphs of plain text."""
    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

    try:
        available = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = available.find_transcript(_TRANSCRIPT_LANGS)
        except NoTranscriptFound:  # captions exist, but only in another language: better than none
            fallback = next(iter(available), None)
            if fallback is None:
                raise
            transcript = fallback
        snippets = transcript.fetch()
    except Exception as exc:  # the library raises one exception class per failure cause
        raise FetchError(f"youtube {video_id}: {type(exc).__name__}: {exc}") from exc
    lines = [" ".join(s.text.split()) for s in snippets if s.text.strip()]
    return _paragraphs(lines)


def fetch_text(url: str) -> str:
    """Text of a reference URL: captions for YouTube links, main page text for anything else."""
    video = youtube_id(url)
    text = fetch_transcript(video) if video else fetch_page(url)
    if not text.strip():
        raise FetchError(f"{url}: no text extracted")
    return text


_NAV_TIMEOUT_MS = 45_000
_SETTLE_TIMEOUT_MS = 10_000
_CLICK_TIMEOUT_MS = 3_000  # optional UI elements: a missing one must not stall for the 45 s default
_PANEL_TIMEOUT_MS = 8_000
_CONSENT_BUTTON = re.compile(
    r"accept all|reject all|全部接受|全部拒絕|接受全部|拒絕全部", re.IGNORECASE
)
_TRANSCRIPT_BUTTON = re.compile(r"show transcript|顯示轉錄|顯示文字記錄|轉錄稿", re.IGNORECASE)


class BrowserFetcher:
    """Downloads references by driving a real Chromium through Playwright, so pages that need
    JavaScript, cookies or a browser-like client work. One browser is shared by all downloads:
    use as a context manager.

    YouTube links: the transcript panel of the watch page is read first (the caption API is often
    blocked by IP); if that fails, the caption API is tried."""

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    def __enter__(self) -> Self:
        from playwright.sync_api import Error, sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self._headless)
        except Error as exc:
            self._playwright.stop()
            self._playwright = None
            raise FetchError(
                f"cannot start Chromium (run `playwright install chromium`): {exc}"
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._browser = self._playwright = None

    def __call__(self, url: str) -> str:
        video = youtube_id(url)
        if video:
            try:
                return self._checked(url, self._youtube_panel(url))
            except FetchError:
                return self._checked(url, fetch_transcript(video))
        return self._checked(url, self._page_text(url))

    @staticmethod
    def _checked(url: str, text: str) -> str:
        if not text.strip():
            raise FetchError(f"{url}: no text extracted")
        return text

    def _new_page(self) -> Any:
        if self._browser is None:
            raise FetchError("BrowserFetcher must be used as a context manager")
        context = self._browser.new_context(locale="en-US", user_agent=_HEADERS["User-Agent"])
        context.set_default_timeout(_NAV_TIMEOUT_MS)
        return context.new_page()

    def _page_text(self, url: str) -> str:
        from playwright.sync_api import Error
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        page = self._new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded")
            if response is not None and response.status >= 400:
                raise FetchError(f"{url}: HTTP {response.status} (blocked or missing)")
            # background requests may never go idle; the content is usually there anyway
            with contextlib.suppress(PlaywrightTimeout):
                page.wait_for_load_state("networkidle", timeout=_SETTLE_TIMEOUT_MS)
            return html_to_text(page.content())
        except Error as exc:
            raise FetchError(f"{url}: {exc}") from exc
        finally:
            page.context.close()

    def _youtube_panel(self, url: str) -> str:
        from playwright.sync_api import Error
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        page = self._new_page()
        try:
            page.goto(url, wait_until="domcontentloaded")
            consent = page.get_by_role("button", name=_CONSENT_BUTTON)
            if consent.count():  # cookie consent, if shown
                consent.first.click(timeout=_CLICK_TIMEOUT_MS)
            with contextlib.suppress(PlaywrightTimeout):  # short descriptions have no "...more"
                page.locator("#expand").first.click(timeout=_CLICK_TIMEOUT_MS)
            page.get_by_role("button", name=_TRANSCRIPT_BUTTON).first.click(
                timeout=_PANEL_TIMEOUT_MS
            )
            segments = page.locator("transcript-segment-view-model span[role=text]")
            segments.first.wait_for(timeout=_PANEL_TIMEOUT_MS)
            lines = [" ".join(t.split()) for t in segments.all_inner_texts()]
        except Error as exc:
            raise FetchError(f"{url}: transcript panel: {exc}") from exc
        finally:
            page.context.close()
        return _paragraphs(lines)
