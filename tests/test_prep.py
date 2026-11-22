from __future__ import annotations

from pathlib import Path

import pytest

from factcheck.fetch import FetchError, html_to_text, youtube_id
from factcheck.prep import parse_clipping, prep, ref_id

CLIP = """---
title: "AI 購物 | 曼報 Pro"
source: "https://example.com/post"
tags:
  - "clippings"
---
## AI 購物

發布日期｜2026.11.08

歡迎閱讀。

---

本篇文章的重點參考資料：

- [訪談](https://www.youtube.com/watch?v=abc123&t=5s)
- [財報](https://www.shopify.com/news/q2)
- [訪談重複](https://www.youtube.com/watch?v=abc123&t=5s)

## 第一節

內文 [連結](https://example.com/x) 在此。
"""


def test_parse_drops_furniture_and_collects_refs() -> None:
    clip = parse_clipping(CLIP)
    assert clip.body == "## 第一節\n\n內文 [連結](https://example.com/x) 在此。\n"
    assert [r.url for r in clip.refs] == [
        "https://www.youtube.com/watch?v=abc123&t=5s",
        "https://www.shopify.com/news/q2",
    ]


def test_parse_without_reference_list_keeps_body() -> None:
    text = '---\ntitle: "T | S"\n---\n## T\n\n發布日期｜2026.01.01\n\n正文第一段。\n'
    clip = parse_clipping(text)
    assert clip.body == "正文第一段。\n"
    assert clip.refs == []


def test_youtube_id_variants() -> None:
    assert youtube_id("https://www.youtube.com/watch?v=abc&t=5s") == "abc"
    assert youtube_id("https://youtu.be/abc?si=1") == "abc"
    assert youtube_id("https://www.youtube.com/shorts/abc") == "abc"
    assert youtube_id("https://www.shopify.com/news") is None


def test_ref_id_is_short() -> None:
    assert ref_id(1) == "R1"
    assert ref_id(4) == "R4"


def test_html_to_text_extracts_article_blocks() -> None:
    html = (
        "<html><body><nav><p>menu</p></nav><article><h1>Title</h1>"
        "<p>First  para.</p><ul><li>one</li></ul></article><footer><p>f</p></footer></body></html>"
    )
    assert html_to_text(html) == "Title\n\nFirst para.\n\none"


def test_prep_writes_files_and_survives_a_failed_download(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.md"
    clip_path.write_text(CLIP, encoding="utf-8")

    def fetcher(url: str) -> str:
        if "shopify" in url:
            raise FetchError("boom")
        return "transcript text"

    out = tmp_path / "out"
    _, results = prep(clip_path, out, fetcher, min_chars=0)
    assert (out / "body.md").read_text(encoding="utf-8").startswith("## 第一節")
    assert (out / "refs" / "R1.md").read_text(encoding="utf-8") == ("transcript text\n")
    assert [r.file for r in results] == ["refs/R1.md", None]
    assert results[1].error == "boom"
    assert "R2" in (out / "sources.json").read_text(encoding="utf-8")


def test_prep_no_download_does_not_call_fetcher(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.md"
    clip_path.write_text(CLIP, encoding="utf-8")

    def fetcher(url: str) -> str:
        pytest.fail("fetcher must not be called")

    prep(clip_path, tmp_path / "out", fetcher, download=False)
    assert not (tmp_path / "out" / "refs").exists()


def test_html_to_text_prefers_the_largest_article_over_teaser_cards() -> None:
    html = (
        "<body><article><p>Teaser</p></article>"
        "<main><article><p>The real article text is much longer than the teaser.</p></article>"
        "</main></body>"
    )
    assert html_to_text(html) == "The real article text is much longer than the teaser."


def test_prep_treats_block_pages_as_failed_and_lists_them(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.md"
    clip_path.write_text(CLIP, encoding="utf-8")

    def fetcher(url: str) -> str:
        return "Failed to verify your browser\n\nCode 21" if "shopify" in url else "ok " * 400

    out = tmp_path / "out"
    _, results = prep(clip_path, out, fetcher)
    assert results[0].file is not None
    assert results[1].file is None
    assert results[1].error is not None
    assert "block / error page" in results[1].error
    assert not (out / "refs" / "R2.md").exists()
    manual = (out / "manual_refs.md").read_text(encoding="utf-8")
    assert "R2" in manual
    assert "https://www.shopify.com/news/q2" in manual
    assert "R1" not in manual


def test_parse_keeps_body_when_reference_list_is_at_the_end() -> None:
    article = "## 標題\n\n" + "正文句子。" * 200 + "\n\n## 參考資料\n\n- [A](https://a.test/x)\n"
    clip = parse_clipping(article)
    assert [r.url for r in clip.refs] == ["https://a.test/x"]
    assert "正文句子。" in clip.body
    assert "參考資料" not in clip.body


def test_parse_ignores_prose_that_mentions_sources() -> None:
    text = (
        "According to sources familiar with the matter, the deal is done and will close soon.\n\n"
        "- [Related](https://a.test/related)\n"
    )
    clip = parse_clipping(text)
    assert clip.refs == []
    assert "According to sources" in clip.body


def test_parse_link_forms() -> None:
    text = (
        "References:\n\n"
        "1. [Foo](https://en.wikipedia.org/wiki/Foo_(bar))\n"
        "2. [![logo](https://a.test/logo.png)](https://b.test/page)\n"
        "- <https://c.test/auto>\n"
        "- see https://d.test/bare.\n"
    )
    assert [r.url for r in parse_clipping(text).refs] == [
        "https://en.wikipedia.org/wiki/Foo_(bar)",
        "https://b.test/page",
        "https://c.test/auto",
        "https://d.test/bare",
    ]


def test_dedupe_ignores_slash_fragment_and_tracking() -> None:
    text = (
        "參考資料：\n\n- [a](https://a.test/x)\n- [b](https://a.test/x/?utm_source=n#top)\n"
        "- [c](https://youtu.be/abc)\n- [d](https://www.youtube.com/watch?v=abc&t=9)\n"
    )
    assert len(parse_clipping(text).refs) == 2


def test_prep_survives_unexpected_errors_and_removes_stale_refs(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.md"
    clip_path.write_text(CLIP, encoding="utf-8")
    stale = tmp_path / "out" / "refs" / "R9.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    def fetcher(url: str) -> str:
        if "youtube" in url:
            raise ValueError("weird")
        return "ok " * 400

    _, results = prep(clip_path, tmp_path / "out", fetcher)
    assert results[0].error == "ValueError: weird"
    assert results[1].file == "refs/R2.md"
    assert (tmp_path / "out" / "sources.json").exists()
    assert not stale.exists()


def test_html_to_text_keeps_form_wrapped_and_div_content() -> None:
    html = (
        "<form id='aspnetForm'><div>First line<br>Second line</div>"
        "<table><tr><td>cell text</td></tr></table>"
        "<div><p>para</p></div><ul><li>one</li></ul></form>"
    )
    assert html_to_text(html) == "First line\n\nSecond line\n\ncell text\n\npara\n\none"


def test_html_to_text_keeps_header_inside_article_only() -> None:
    html = (
        "<header><p>site menu</p></header>"
        "<article><header><p>Lead paragraph</p></header><p>Body</p></article>"
    )
    assert html_to_text(html) == "Lead paragraph\n\nBody"


def test_html_to_text_decodes_meta_charset_from_bytes() -> None:
    html = '<meta charset="big5"><p>繁體中文內容</p>'.encode("big5")
    assert html_to_text(html) == "繁體中文內容"


def test_youtube_id_other_hosts() -> None:
    assert youtube_id("https://www.youtube-nocookie.com/embed/abc") == "abc"
    assert youtube_id("https://music.youtube.com/watch?v=abc") == "abc"
