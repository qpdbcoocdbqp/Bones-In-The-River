from itertools import pairwise

import pytest

from factcheck.text import (
    chunk_spans,
    locate_quote,
    parse_body,
    plain,
    split_sentences,
)


def test_split_sentences_chinese_keeps_decimals_and_percent():
    text = "Shopify 向商家收取約 3.5% 的手續費。這是真的嗎？是的！"
    assert split_sentences(text) == [
        "Shopify 向商家收取約 3.5% 的手續費。",
        "這是真的嗎？",
        "是的！",
    ]


def test_split_sentences_keeps_closing_quote_with_sentence():
    assert split_sentences("他說：「好。」然後離開。") == ["他說：「好。」", "然後離開。"]


def test_split_sentences_does_not_split_inside_link_url():
    text = "請看[影片](https://youtu.be/watch?v=abc)。下一句。"
    assert split_sentences(text) == ["請看[影片](https://youtu.be/watch?v=abc)。", "下一句。"]


def test_split_sentences_english_period_before_capital():
    assert split_sentences("It rose 3.5 percent. Then it fell.") == [
        "It rose 3.5 percent.",
        "Then it fell.",
    ]


def test_split_sentences_trailing_text_without_terminator():
    assert split_sentences("第一句。沒有句號的尾巴") == ["第一句。", "沒有句號的尾巴"]


def test_plain_strips_markdown_link_syntax():
    assert plain("見[之前的分析](https://x.test/a)了嗎") == "見之前的分析了嗎"


def test_parse_body_blocks_and_ids():
    body = "## 標題\n\n第一段第一句。第一段第二句。\n\n---\n\n第二段。"
    blocks = parse_body(body)
    assert [b.kind for b in blocks] == ["verbatim", "paragraph", "verbatim", "paragraph"]
    assert [s.id for b in blocks for s in b.sentences] == ["p1s1", "p1s2", "p2s1"]


def test_chunk_spans_short_text_is_single_chunk():
    assert chunk_spans("abc", 10, 2) == [(0, 3)]


def test_chunk_spans_covers_text_with_overlap_and_prefers_spaces():
    text = " ".join(["word"] * 40)  # 199 字元
    spans = chunk_spans(text, 60, 10)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (_, prev_end), (next_start, _) in pairwise(spans):
        assert next_start < prev_end  # 有重疊
    assert all(end - start <= 60 for start, end in spans)
    assert all(text[end] == " " for _, end in spans[:-1])  # 在空白處斷開


def test_chunk_spans_rejects_large_overlap():
    with pytest.raises(ValueError):
        chunk_spans("x" * 100, 20, 10)


def test_locate_quote_ignores_case_punctuation_and_tags():
    source = "So, uh [music] the thing is: we don't control the entry point. It's fine."
    span = locate_quote("The thing is we dont control the entry point", source)
    assert span is not None
    assert source[span[0] : span[1]] == "the thing is: we don't control the entry point"


def test_locate_quote_not_found_and_too_short():
    assert (
        locate_quote("this sentence is not in the source at all", "something else entirely") is None
    )
    assert locate_quote("short", "short text here") is None


def test_means_nothing():
    from factcheck.text import means_nothing

    assert all(means_nothing(v) for v in ("", " none ", "None.", "N/A", "無", "沒有。"))
    assert not means_nothing("沒有提到品牌")


def test_heading_followed_directly_by_body_text_does_not_swallow_the_body():
    """審查發現：標題後面沒有空行就接正文時，整塊被當成標題而不驗證。"""
    blocks = parse_body("## 營收\n本季營收成長 20%。\n\n另一段。")
    assert [(b.kind, b.text) for b in blocks] == [
        ("verbatim", "## 營收"),
        ("paragraph", "本季營收成長 20%。"),
        ("paragraph", "另一段。"),
    ]
    assert [s.id for b in blocks for s in b.sentences] == ["p1s1", "p2s1"]


def test_hashtag_without_space_is_not_a_heading():
    blocks = parse_body("#hashtag 開頭的句子。")
    assert blocks[0].kind == "paragraph"


def test_paragraph_text_before_a_heading_is_kept_as_paragraph():
    blocks = parse_body("前文一句。\n# 標題\n後文一句。")
    assert [b.kind for b in blocks] == ["paragraph", "verbatim", "paragraph"]


def test_locate_quote_rejects_ambiguous_match():
    source = "the quick brown fox jumps. other words here. the quick brown fox jumps."
    assert locate_quote("the quick brown fox jumps", source) is None
