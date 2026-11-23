import json

import pytest

from factcheck.module.app.text import parse_body
from factcheck.module.cite.verifier import MAX_QUOTE_CHARS, LLMVerifier, pick_best
from factcheck.module.models import Label, Reference
from factcheck.module.utils.llm import LLMTransportError
from tests.fakes import FakeClient

SOURCE = "We never control the entry point. Merchants keep their own customers and brand."


def _sentences():
    return [s for b in parse_body("商家保有自己的顧客。入口由別人控制。") for s in b.sentences]


def _reply(*items: dict[str, str]) -> str:
    return json.dumps({"results": list(items)})


def test_supported_with_verified_quote_gets_span_in_reference():
    client = FakeClient(
        [
            _reply(
                {
                    "id": "p1s1",
                    "label": "supported",
                    "quote": "Merchants keep their own customers",
                    "note": "對應",
                }
            ),
            json.dumps({"verdict": "full", "missing": "無"}),
        ]
    )
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    judgement = result.judgements["p1s1"]
    assert judgement.label is Label.SUPPORTED
    assert judgement.span is not None
    assert SOURCE[judgement.span[0] : judgement.span[1]] == judgement.quote
    assert result.judgements["p1s2"].label is Label.NOT_ENOUGH_INFO
    assert result.calls == 2
    assert result.errors == []


def test_quote_not_in_source_is_downgraded():
    client = FakeClient(
        [
            _reply(
                {
                    "id": "p1s1",
                    "label": "supported",
                    "quote": "a sentence the source never says",
                    "note": "",
                }
            )
        ]
    )
    judgement = (
        LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE)).judgements["p1s1"]
    )
    assert judgement.label is Label.NOT_ENOUGH_INFO
    assert "quote check failed" in judgement.note
    assert judgement.quote is None


def test_unknown_sentence_id_is_reported_not_applied():
    client = FakeClient(
        [
            _reply(
                {
                    "id": "p9s9",
                    "label": "supported",
                    "quote": "Merchants keep their own customers",
                    "note": "",
                }
            )
        ]
    )
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert any("p9s9" in e for e in result.errors)
    assert all(j.label is Label.NOT_ENOUGH_INFO for j in result.judgements.values())


def test_format_failure_is_recorded_and_does_not_crash():
    client = FakeClient(["garbage"] * 3)
    result = LLMVerifier(client, retries=2).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert len(result.errors) == 1
    assert all(j.label is Label.NOT_ENOUGH_INFO for j in result.judgements.values())


def test_batches_and_chunks_each_cost_one_call_and_any_chunk_supporting_wins():
    long_source = (
        "filler words " * 30 + "Merchants keep their own customers and brand. " + "tail text " * 30
    )
    quote = "Merchants keep their own customers and brand"

    def respond(prompt: str) -> str:
        if "STATEMENT (Chinese)" in prompt:
            return _confirm("full")
        if quote in prompt:
            return _reply({"id": "p1s1", "label": "supported", "quote": quote, "note": ""})
        return _reply()

    client = FakeClient(respond)
    verifier = LLMVerifier(client, max_chars=300, overlap=50, batch_size=1)
    result = verifier.verify(_sentences(), Reference(id="r1", text=long_source))
    assert result.judgements["p1s1"].label is Label.SUPPORTED
    stage1 = [p for p in client.prompts if "<source>" in p]
    chunks = len(stage1) // 2  # 2 個句子、batch_size=1
    assert chunks >= 2
    assert len(stage1) == chunks * 2
    assert result.calls == len(client.prompts) == chunks * 2 + 1  # 另加 1 次第二階段核對


def test_pick_best_prefers_supported_then_partial_then_notes():
    sid, ref = "p1s1", "r1"
    supported, partial, unknown = _j(Label.SUPPORTED), _j(Label.PARTIAL), _j(Label.NOT_ENOUGH_INFO)
    assert pick_best(sid, ref, [unknown, partial, supported]).label is Label.SUPPORTED
    assert pick_best(sid, ref, [unknown, partial]).label is Label.PARTIAL
    assert pick_best(sid, ref, [unknown]).label is Label.NOT_ENOUGH_INFO
    assert pick_best(sid, ref, []).label is Label.NOT_ENOUGH_INFO
    noted = _j(Label.NOT_ENOUGH_INFO).model_copy(update={"note": "降級原因"})
    assert pick_best(sid, ref, [unknown, noted]).note == "降級原因"


def _j(label: Label):
    from factcheck.module.models import Judgement

    return Judgement(sentence_id="p1s1", ref_id="r1", label=label)


def _confirm(verdict: str, missing: str = "無") -> str:
    return json.dumps({"verdict": verdict, "missing": missing})


def _supported_reply() -> str:
    return _reply(
        {
            "id": "p1s1",
            "label": "supported",
            "quote": "Merchants keep their own customers",
            "note": "對應",
        }
    )


def test_confirm_full_keeps_supported_and_costs_one_extra_call():
    client = FakeClient([_supported_reply(), _confirm("full")])
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.judgements["p1s1"].label is Label.SUPPORTED
    assert result.judgements["p1s1"].quote is not None
    assert result.calls == 2


def test_confirm_partial_becomes_partially_supported_with_missing_note():
    client = FakeClient([_supported_reply(), _confirm("partial", "沒有提到品牌")])
    judgement = (
        LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE)).judgements["p1s1"]
    )
    assert judgement.label is Label.PARTIAL
    assert judgement.note == "Missing: 沒有提到品牌"
    assert judgement.quote is not None  # 部分相關仍保留引文供人工判斷


def test_confirm_unrelated_drops_to_not_enough_info_and_clears_quote():
    client = FakeClient([_supported_reply(), _confirm("unrelated")])
    judgement = (
        LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE)).judgements["p1s1"]
    )
    assert judgement.label is Label.NOT_ENOUGH_INFO
    assert judgement.quote is None
    assert judgement.span is None


def test_confirm_failure_is_recorded_and_treated_as_partial():
    client = FakeClient([_supported_reply(), "garbage", "garbage", "garbage"])
    result = LLMVerifier(client, retries=2).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.judgements["p1s1"].label is Label.PARTIAL
    assert any("recheck" in e for e in result.errors)


def test_confirm_is_not_called_when_nothing_is_supported():
    client = FakeClient([_reply()])
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.calls == 1
    assert len(client.prompts) == 1


def test_sentence_id_with_brackets_from_model_is_accepted():
    client = FakeClient(
        [
            _reply(
                {
                    "id": " [p1s1] ",
                    "label": "supported",
                    "quote": "Merchants keep their own customers",
                    "note": "",
                }
            ),
            _confirm("full"),
        ]
    )
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.errors == []
    assert result.judgements["p1s1"].label is Label.SUPPORTED


def test_later_chunk_with_full_support_wins_when_first_candidate_is_downgraded():
    """審查發現：只核對第一個 supported 候選，被降級後後面 chunk 更好的引文就沒機會。"""
    first = "Merchants keep their own customers"
    second = "Merchants keep their own customers and brand forever"
    source = f"{first} in some cases. " + "filler words " * 40 + f"{second}. " + "tail " * 40

    def respond(prompt: str) -> str:
        if "STATEMENT (Chinese)" in prompt:
            return _confirm("full" if second in prompt else "partial", "細節")
        quote = second if second in prompt else first
        return _reply({"id": "p1s1", "quote": quote, "note": ""})

    client = FakeClient(respond)
    verifier = LLMVerifier(client, max_chars=400, overlap=50)
    judgement = verifier.verify(_sentences(), Reference(id="r1", text=source)).judgements["p1s1"]
    assert judgement.label is Label.SUPPORTED
    assert judgement.quote is not None
    assert second.lower() in judgement.quote.lower()


def test_one_malformed_item_does_not_discard_the_rest_of_the_batch():
    client = FakeClient(
        [
            json.dumps(
                {
                    "results": [
                        {"id": "p1s2", "quote": None, "note": ""},  # 壞項目
                        {"id": "p1s1", "quote": "Merchants keep their own customers"},
                    ]
                }
            ),
            _confirm("full"),
        ]
    )
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.judgements["p1s1"].label is Label.SUPPORTED
    assert any("skipped malformed" in e for e in result.errors)


def test_overlong_quote_is_downgraded():
    long_source = "Merchants keep their own customers and brand. " + "x " * MAX_QUOTE_CHARS
    client = FakeClient([_reply({"id": "p1s1", "quote": long_source, "note": ""})])
    judgement = (
        LLMVerifier(client, max_chars=10_000)
        .verify(_sentences(), Reference(id="r1", text=long_source))
        .judgements["p1s1"]
    )
    assert judgement.label is Label.NOT_ENOUGH_INFO
    assert "too long" in judgement.note


def test_quote_appearing_twice_in_source_cannot_be_located():
    source = "Merchants keep their own customers. Other text. Merchants keep their own customers."
    client = FakeClient([_reply({"id": "p1s1", "quote": "Merchants keep their own customers"})])
    result = LLMVerifier(client).verify(_sentences(), Reference(id="r1", text=source))
    judgement = result.judgements["p1s1"]
    assert judgement.label is Label.NOT_ENOUGH_INFO
    assert "multiple times" in judgement.note


def test_consecutive_transport_failures_abort_the_run():
    class Down:
        model = "down"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, system: str | None = None) -> str:
            self.calls += 1
            raise LLMTransportError("連不上")

    client = Down()
    verifier = LLMVerifier(client, max_chars=200, overlap=20, batch_size=1)
    source = "word " * 400  # 多個 chunk × 多個批次，若不中止會呼叫很多次
    with pytest.raises(LLMTransportError, match="3 consecutive"):
        verifier.verify(_sentences(), Reference(id="r1", text=source))
    assert client.calls == 3  # 連線失敗不重試，第 3 次就中止
