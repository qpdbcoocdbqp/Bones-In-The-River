import json
from collections.abc import Sequence

import pytest

from factcheck.module.app.text import parse_body
from factcheck.module.cite.annotate import annotate
from factcheck.module.cite.retrieval import RetrievalVerifier, merge_ranges
from factcheck.module.models import Label, Reference
from factcheck.module.utils.llm import LLMTransportError
from tests.fakes import FakeClient

FILLER = "The weather was mild and the harbour stayed quiet all afternoon. " * 30
TARGET = "Merchants keep their own customers and brand."
SOURCE = FILLER + TARGET + " " + FILLER
VOCAB = ["merchants", "customers", "brand", "weather", "harbour"]


def _sentences():
    return [
        s for b in parse_body("merchants customers brand。weather harbour。") for s in b.sentences
    ]


class FakeEmbedder:
    """Bag of the VOCAB words: texts sharing words are close."""

    model = "fake-embed"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(t.lower().count(w)) + 0.01 for w in VOCAB] for t in texts]


class FakeReranker:
    model = "fake-rerank"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.queries.append(query)
        if self.fail:
            raise LLMTransportError("rerank down")
        return [float(sum(w in d.lower() for w in query.lower().split())) for d in documents]


def _llm(prompts: list[str]):
    def reply(prompt: str) -> str:
        prompts.append(prompt)
        if "<statements>" in prompt:
            if "[p1s1]" in prompt:
                quote = TARGET if TARGET in prompt else "text the source never says"
                return json.dumps({"results": [{"id": "p1s1", "quote": quote, "note": "match"}]})
            return json.dumps({"results": []})
        return json.dumps({"verdict": "full", "missing": "none"})

    return reply


def test_merge_ranges_joins_overlapping_and_touching():
    assert merge_ranges([(10, 20), (0, 5), (18, 30), (30, 35), (50, 60)]) == [
        (0, 5),
        (10, 35),
        (50, 60),
    ]


def test_judge_sees_only_retrieved_excerpt_and_span_points_into_full_text():
    prompts: list[str] = []
    verifier = RetrievalVerifier(
        FakeClient(_llm(prompts)), FakeEmbedder(), FakeReranker(), top_k=1, pad=50
    )
    result = verifier.verify(_sentences(), Reference(id="r1", text=SOURCE))

    judgement = result.judgements["p1s1"]
    assert judgement.label is Label.SUPPORTED
    assert judgement.span is not None
    assert SOURCE[judgement.span[0] : judgement.span[1]] == TARGET.rstrip(".")
    judge_prompt = next(p for p in prompts if "[p1s1]" in p and "<statements>" in p)
    assert TARGET in judge_prompt
    assert len(judge_prompt) < len(SOURCE)  # not the whole source
    assert result.errors == []
    assert result.calls == 3  # two judge calls (one per sentence) + one recheck


def test_quote_outside_shown_excerpts_is_downgraded():
    prompts: list[str] = []
    verifier = RetrievalVerifier(FakeClient(_llm(prompts)), FakeEmbedder(), top_k=1, pad=0)
    # the weather sentence's excerpt does not contain TARGET, so a quote of TARGET cannot be located
    sentences = [s for b in parse_body("weather harbour。") for s in b.sentences]
    sentences[0] = sentences[0].model_copy(update={"id": "p1s1"})
    result = verifier.verify(sentences, Reference(id="r1", text=SOURCE))
    assert result.judgements["p1s1"].label is Label.NOT_ENOUGH_INFO


def test_min_score_skips_the_llm():
    prompts: list[str] = []
    verifier = RetrievalVerifier(
        FakeClient(_llm(prompts)), FakeEmbedder(), FakeReranker(), min_score=100.0
    )
    result = verifier.verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert prompts == []
    assert result.calls == 0
    assert all(j.label is Label.NOT_ENOUGH_INFO for j in result.judgements.values())


def test_rerank_failure_falls_back_to_embedding_order_and_is_reported():
    prompts: list[str] = []
    verifier = RetrievalVerifier(
        FakeClient(_llm(prompts)), FakeEmbedder(), FakeReranker(fail=True), top_k=1, pad=50
    )
    result = verifier.verify(_sentences(), Reference(id="r1", text=SOURCE))
    assert result.judgements["p1s1"].label is Label.SUPPORTED
    assert any("rerank failed" in e for e in result.errors)


def test_claim_embeddings_are_reused_across_references():
    embedder = FakeEmbedder()
    verifier = RetrievalVerifier(FakeClient(_llm([])), embedder)
    for ref_id in ("r1", "r2"):
        verifier.verify(_sentences(), Reference(id=ref_id, text=SOURCE))
    claim_calls = [c for c in embedder.calls if len(c) == 2 and "merchants" in c[0] + c[1]]
    assert len(claim_calls) == 1


def test_connection_failures_abort():
    def down(prompt: str) -> str:
        raise LLMTransportError("down")

    sentences = [s for b in parse_body("a b c d。e f g h。i j k l。m n o p。") for s in b.sentences]
    verifier = RetrievalVerifier(FakeClient(down), FakeEmbedder(), workers=1)
    with pytest.raises(LLMTransportError):
        verifier.verify(sentences, Reference(id="r1", text=SOURCE))


def test_top_refs_judges_each_sentence_only_against_its_closest_reference():
    prompts: list[str] = []
    verifier = RetrievalVerifier(FakeClient(_llm(prompts)), FakeEmbedder(), top_k=1, top_refs=1)
    refs = [
        Reference(id="weather", text=FILLER),
        Reference(id="merchants", text=SOURCE),
    ]
    annotation = annotate("merchants customers brand。", refs, verifier)
    by_ref = {j.ref_id: j for j in annotation.blocks[0].results[0].judgements}
    assert by_ref["merchants"].label is Label.SUPPORTED
    assert by_ref["weather"].label is Label.NOT_ENOUGH_INFO
    assert len([p for p in prompts if "<statements>" in p]) == 1  # the other pair was never judged
