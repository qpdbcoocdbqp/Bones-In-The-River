import json
from pathlib import Path

import pytest

from factcheck.module.app.render import NO_SOURCE, to_markdown
from factcheck.module.cite.annotate import annotate
from factcheck.module.cite.verifier import LLMVerifier
from factcheck.module.models import Label, Reference
from factcheck.module.refer.io import load_references, write_outputs
from tests.fakes import FakeClient

BODY = "## 標題\n\n商家保有自己的顧客。入口由別人控制。\n\n與來源無關的推論。"
REFS = [
    Reference(id="r1", text="Merchants keep their own customers and brand. Nothing else here."),
    Reference(id="r2", text="Entry points are controlled by the AI platforms, not merchants."),
]


def _respond(prompt: str) -> str:
    if "STATEMENT (Chinese)" in prompt:  # 第二階段：r1 完整支持，r2 只有部分
        if "Merchants keep their own customers" in prompt:
            return json.dumps({"verdict": "full", "missing": "無"})
        return json.dumps({"verdict": "partial", "missing": "沒有說明由誰控制"})
    if "Merchants keep their own customers" in prompt:
        item = {
            "id": "p1s1",
            "label": "supported",
            "quote": "Merchants keep their own customers",
            "note": "ok",
        }
    elif "controlled by the AI platforms" in prompt:
        item = {
            "id": "p1s2",
            "label": "supported",
            "quote": "Entry points are controlled by the AI platforms",
            "note": "ok",
        }
    else:
        return json.dumps({"results": []})
    return json.dumps({"results": [item]})


@pytest.fixture
def annotation():
    return annotate(BODY, REFS, LLMVerifier(FakeClient(_respond)))


def test_sources_are_attached_per_paragraph(annotation):
    heading, first, second = annotation.blocks
    assert heading.block.kind == "verbatim"
    assert first.ref_ids(Label.SUPPORTED) == ["r1"]
    assert first.ref_ids(Label.PARTIAL) == ["r2"]
    assert second.ref_ids(Label.SUPPORTED) == []
    assert annotation.meta.ref_ids == ["r1", "r2"]
    assert annotation.meta.calls == 4  # 每份參考資料：1 次找候選 + 1 次嚴格核對
    assert annotation.meta.calibrated is False


def test_markdown_keeps_body_text_and_marks_sources(annotation):
    markdown = to_markdown(annotation)
    assert "## 標題" in markdown
    assert "商家保有自己的顧客。入口由別人控制。  [Source: r1][Partial: r2]" in markdown
    assert "◐ Partially related **r2**" in markdown
    assert "Missing: 沒有說明由誰控制" in markdown
    assert f"與來源無關的推論。  {NO_SOURCE}" in markdown
    assert 'Sentence 1 ← **r1**: "Merchants keep their own customers"' in markdown
    assert "not calibrated" in markdown


def test_write_outputs_roundtrip(tmp_path: Path, annotation):
    md_path, json_path = write_outputs(annotation, to_markdown(annotation), tmp_path / "out")
    assert md_path.read_text(encoding="utf-8").startswith("> ⚠")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    judgement = data["blocks"][1]["results"][0]["judgements"][0]
    assert judgement["ref_id"] == "r1"
    assert judgement["label"] == "supported"
    assert judgement["span"] is not None


def test_load_references_uses_stem_as_id_and_rejects_empty_dir(tmp_path: Path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    assert [r.id for r in load_references(tmp_path)] == ["a", "b"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_references(empty)


def test_duplicate_reference_ids_are_rejected():
    """審查發現：earnings.txt 與 earnings.md 的 id 相同，結果會互相覆蓋。"""
    refs = [Reference(id="earnings", text="a"), Reference(id="earnings", text="b")]
    with pytest.raises(ValueError, match="Duplicate reference id"):
        annotate(BODY, refs, LLMVerifier(FakeClient(_respond)))


def test_markdown_header_warns_when_some_calls_failed():
    bad = annotate(BODY, REFS, LLMVerifier(FakeClient(["garbage"] * 20), retries=0))
    assert bad.meta.errors
    assert "result is incomplete" in to_markdown(bad)
    assert "result is incomplete" not in to_markdown(
        annotate(BODY, REFS, LLMVerifier(FakeClient(_respond)))
    )


def test_screen_references_skips_failed_downloads_and_reports_their_url(tmp_path: Path):
    import json

    from factcheck.module.refer.io import MIN_REF_CHARS, screen_references, write_manual_list

    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    good = Reference(id="r1", text="word " * MIN_REF_CHARS)
    bad = Reference(id="r2", text="Failed to verify your browser")
    entry = {"id": "r2", "title": "Talk", "url": "https://x.test/talk", "file": "refs/r2.md"}
    (tmp_path / "sources.json").write_text(json.dumps([entry]), encoding="utf-8")

    usable, skipped = screen_references([good, bad], refs_dir)
    assert [r.id for r in usable] == ["r1"]
    assert [(s.id, s.url) for s in skipped] == [("r2", "https://x.test/talk")]

    write_manual_list(skipped, tmp_path / "manual_refs.md")
    assert "https://x.test/talk" in (tmp_path / "manual_refs.md").read_text(encoding="utf-8")
    write_manual_list([], tmp_path / "manual_refs.md")
    assert not (tmp_path / "manual_refs.md").exists()
