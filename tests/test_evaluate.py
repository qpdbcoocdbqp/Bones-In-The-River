import csv
from pathlib import Path

import pytest

from factcheck.module.app.sheet import sample_sentences, sheet_rows
from factcheck.module.cite.evaluate import (
    ANY_RELEVANT,
    FULL,
    GOLD_ANY,
    GOLD_FULL,
    cohen_kappa,
    score_paragraphs,
    score_sentences,
    weights_from_meta,
)
from factcheck.module.models import (
    Annotation,
    Block,
    BlockResult,
    GoldRow,
    Judgement,
    Label,
    RunMeta,
    SampleMeta,
    Sentence,
    SentenceResult,
)
from factcheck.module.refer.io import read_gold, write_label_sheet


def _result(sid: str, *judgements: tuple[str, Label]) -> SentenceResult:
    sentence = Sentence(id=sid, text=sid, claim=sid)
    return SentenceResult(
        sentence=sentence,
        judgements=[
            Judgement(sentence_id=sid, ref_id=ref, label=label, quote="q")
            for ref, label in judgements
        ],
    )


def _annotation(blocks: list[list[SentenceResult]]) -> Annotation:
    meta = RunMeta(model="m", prompt_version="v", ref_ids=["r1", "r2"])
    return Annotation(
        meta=meta,
        blocks=[
            BlockResult(block=Block(kind="paragraph", text=f"para{i}"), results=results)
            for i, results in enumerate(blocks)
        ],
    )


def _gold(**rows: tuple[str, set[str]]) -> dict[str, GoldRow]:
    return {
        sid: GoldRow.model_validate({"sentence_id": sid, "level": level, "refs": refs})
        for sid, (level, refs) in rows.items()
    }


ANNOTATION = _annotation(
    [
        [
            _result("p1s1", ("r1", Label.SUPPORTED)),  # 對：gold full r1
            _result("p1s2", ("r2", Label.SUPPORTED)),  # 來源錯：gold full r1
            _result("p1s3", ("r1", Label.PARTIAL)),  # gold none → 在「任何相關」是 FP
        ],
        [
            _result("p2s1"),  # 系統沒找到，gold full r1 → FN
            _result("p2s2", ("r1", Label.SUPPORTED)),  # 未標註 → 不計
        ],
    ]
)
GOLD = _gold(
    p1s1=("full", {"r1"}),
    p1s2=("full", {"r1"}),
    p1s3=("none", set()),
    p2s1=("full", {"r1"}),
)


def test_sentence_scores_require_matching_ref_and_ignore_unlabelled():
    score = score_sentences(ANNOTATION, GOLD, FULL, GOLD_FULL)
    assert (score.tp, score.system, score.gold) == (1, 2, 3)
    assert score.precision == pytest.approx(1 / 2)
    assert score.recall == pytest.approx(1 / 3)


def test_any_relevant_counts_partial_on_both_sides():
    score = score_sentences(ANNOTATION, GOLD, ANY_RELEVANT, GOLD_ANY)
    assert (score.tp, score.system, score.gold) == (1, 3, 3)


def test_paragraph_scores_compare_ref_sets_over_labelled_sentences_only():
    score = score_paragraphs(ANNOTATION, GOLD, FULL, GOLD_FULL)
    # 段落 1：系統 {r1, r2}、人工 {r1} → TP 1；段落 2：系統 {}（p2s2 未標註）、人工 {r1}
    assert (score.tp, score.system, score.gold) == (1, 2, 2)


def test_score_without_positives_is_undefined_not_zero():
    annotation = _annotation([[_result("p1s1")]])
    score = score_sentences(annotation, _gold(p1s1=("none", set())), FULL, GOLD_FULL)
    assert score.precision is None
    assert score.recall is None


def test_cohen_kappa():
    a = _gold(s1=("full", {"r1"}), s2=("none", set()), s3=("partial", {"r1"}), s4=("none", set()))
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    b = _gold(s1=("full", {"r1"}), s2=("none", set()), s3=("none", set()), s4=("none", set()))
    assert 0 < (cohen_kappa(a, b) or 0) < 1
    assert cohen_kappa(a, {}) is None
    same = _gold(s1=("none", set()), s2=("none", set()))
    assert cohen_kappa(same, same) is None  # 只用單一標籤，kappa 無定義


def test_read_gold_skips_blank_rows_and_validates(tmp_path: Path):
    path = tmp_path / "g.csv"
    base = {"sentence_id": "", "gold": "", "gold_refs": ""}
    write_label_sheet(
        [
            {**base, "sentence_id": "a", "gold": "FULL", "gold_refs": "r1; r2"},
            {**base, "sentence_id": "b", "gold": "none"},
            {**base, "sentence_id": "c"},  # 未標註
        ],
        path,
    )
    gold = read_gold(path)
    assert set(gold) == {"a", "b"}
    assert gold["a"].refs == frozenset({"r1", "r2"})

    write_label_sheet([{**base, "sentence_id": "a", "gold": "full"}], path)  # 缺 gold_refs
    with pytest.raises(ValueError, match="gold_refs"):
        read_gold(path)
    write_label_sheet([{**base, "sentence_id": "a", "gold": "maybe"}], path)
    with pytest.raises(ValueError, match="gold must be one of"):
        read_gold(path)
    write_label_sheet(
        [
            {**base, "sentence_id": "a", "gold": "none"},
            {**base, "sentence_id": "a", "gold": "none"},
        ],
        path,
    )
    with pytest.raises(ValueError, match="duplicate sentence_id"):
        read_gold(path)


def test_sample_is_stratified_deterministic_and_in_body_order():
    results = (
        [_result(f"p1s{i}", ("r1", Label.SUPPORTED)) for i in range(1, 5)]
        + [_result(f"p2s{i}", ("r1", Label.PARTIAL)) for i in range(1, 5)]
        + [_result(f"p3s{i}") for i in range(1, 5)]
    )
    annotation = _annotation([results])
    ids = [r.sentence.id for r in sample_sentences(annotation, 6, seed=1)]
    assert len(set(ids)) == 6
    assert [i[:2] for i in ids] == sorted(i[:2] for i in ids)  # 依內文順序
    assert sum(i.startswith("p1") for i in ids) == 2  # 每層 2 句
    assert [r.sentence.id for r in sample_sentences(annotation, 6, seed=1)] == ids


def test_sample_fills_shortfall_from_other_strata():
    results = [_result("p1s1", ("r1", Label.SUPPORTED))] + [_result(f"p3s{i}") for i in range(1, 8)]
    assert len(sample_sentences(_annotation([results]), 6, seed=0)) == 6


def test_sheet_rows_are_blind_by_default_and_show_hints_on_request():
    annotation = _annotation([[_result("p1s1", ("r1", Label.SUPPORTED)), _result("p1s2")]])
    blind = sheet_rows(annotation, n=2)
    assert "system_label" not in blind[0]
    assert blind[0]["gold"] == ""
    assert blind[0]["available_refs"] == "r1;r2"
    hinted = sheet_rows(annotation, n=2, hints=True)
    assert hinted[0]["system_label"] == "supported"
    assert hinted[0]["system_refs"] == "r1"
    assert hinted[1]["system_label"] == "none"


def test_label_sheet_roundtrip_keeps_chinese(tmp_path: Path):
    path = tmp_path / "s.csv"
    row = {"sentence_id": "p1s1", "sentence": "中文句子。", "gold": "", "gold_refs": ""}
    write_label_sheet([row], path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        assert next(csv.DictReader(f))["sentence"] == "中文句子。"


def test_cli_sheet_then_eval_end_to_end(tmp_path: Path):
    from typer.testing import CliRunner

    from factcheck.cli import app

    annotations = tmp_path / "annotations.json"
    annotations.write_text(ANNOTATION.model_dump_json(), encoding="utf-8")
    runner = CliRunner()

    sheet_path = tmp_path / "sheet.csv"
    result = runner.invoke(
        app, ["sheet", "--annotations", str(annotations), "--out", str(sheet_path), "--n", "4"]
    )
    assert result.exit_code == 0, result.output
    with sheet_path.open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 4

    gold_path = tmp_path / "gold.csv"
    base = {"sentence_id": "", "gold": "", "gold_refs": ""}
    write_label_sheet(
        [
            {**base, "sentence_id": "p1s1", "gold": "full", "gold_refs": "r1"},
            {**base, "sentence_id": "p1s2", "gold": "full", "gold_refs": "r1"},
            {**base, "sentence_id": "p2s1", "gold": "full", "gold_refs": "r1"},
        ],
        gold_path,
    )
    args = ["eval", "--annotations", str(annotations), "--gold", str(gold_path)]
    ok = runner.invoke(app, args)
    assert ok.exit_code == 0, ok.output
    assert "precision=0.50" in ok.output
    assert runner.invoke(app, [*args, "--min-precision", "0.4"]).exit_code == 0
    assert runner.invoke(app, [*args, "--min-precision", "0.9"]).exit_code == 1

    write_label_sheet(
        [{**base, "sentence_id": "p1s1", "gold": "full", "gold_refs": "nope"}], gold_path
    )
    assert runner.invoke(app, args).exit_code == 2


def test_weighting_reweights_strata_to_estimate_the_whole_article():
    """審查發現：各層等量抽樣沒有加權，precision／recall 會被抽樣設計扭曲。"""
    annotation = _annotation(
        [
            [
                _result("a1", ("r1", Label.SUPPORTED)),  # supported 層：母體 2，抽 1
                _result("n1"),  # none 層：母體 8，抽 2（其中 n1 其實有來源 → 系統漏掉）
                _result("n2"),
            ]
        ]
    )
    gold = _gold(a1=("full", {"r1"}), n1=("full", {"r1"}), n2=("none", set()))
    unweighted = score_sentences(annotation, gold, FULL, GOLD_FULL)
    assert unweighted.recall == pytest.approx(1 / 2)  # 不加權：抽到 1 個命中、1 個漏掉

    meta = SampleMeta(
        population={"supported": 2, "partial": 0, "none": 8},
        sampled={"a1": "supported", "n1": "none", "n2": "none"},
    )
    weights = weights_from_meta(meta)
    assert weights == {"a1": 2.0, "n1": 4.0, "n2": 4.0}
    weighted = score_sentences(annotation, gold, FULL, GOLD_FULL, weights)
    # 估計母體：命中 2 句；漏掉 4 句 → recall = 2 / (2 + 4)
    assert weighted.recall == pytest.approx(2 / 6)
    assert weighted.precision == pytest.approx(1.0)


def test_cli_eval_with_strata_and_stray_ids(tmp_path: Path):
    from typer.testing import CliRunner

    from factcheck.cli import app

    annotations = tmp_path / "annotations.json"
    annotations.write_text(ANNOTATION.model_dump_json(), encoding="utf-8")
    runner = CliRunner()
    sheet_path = tmp_path / "sheet.csv"
    assert (
        runner.invoke(
            app, ["sheet", "--annotations", str(annotations), "--out", str(sheet_path), "--n", "5"]
        ).exit_code
        == 0
    )
    strata_path = tmp_path / "sheet.strata.json"
    assert strata_path.exists()

    gold_path = tmp_path / "gold.csv"
    base = {"sentence_id": "", "gold": "", "gold_refs": ""}
    write_label_sheet(
        [{**base, "sentence_id": "p1s1", "gold": "full", "gold_refs": "r1"}], gold_path
    )
    args = ["eval", "--annotations", str(annotations), "--gold", str(gold_path)]
    weighted = runner.invoke(app, [*args, "--strata", str(strata_path)])
    assert weighted.exit_code == 0, weighted.output
    assert "weighted by stratum" in weighted.output
    assert "--strata not given" in runner.invoke(app, args).output

    write_label_sheet([{**base, "sentence_id": "zzz", "gold": "none"}], gold_path)
    assert runner.invoke(app, args).exit_code == 2  # 不屬於這份 annotations 的句子


def test_cli_sheet_with_nothing_to_sample_exits_cleanly(tmp_path: Path):
    from typer.testing import CliRunner

    from factcheck.cli import app

    empty = _annotation([])
    annotations = tmp_path / "annotations.json"
    annotations.write_text(empty.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["sheet", "--annotations", str(annotations), "--out", str(tmp_path / "s.csv")]
    )
    assert result.exit_code == 2
    assert "Cannot create label sheet" in result.output
