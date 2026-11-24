"""Verifier that shows the LLM only the passages worth reading.

Per (sentence, reference): embed passages -> cosine recall -> optional rerank -> merge the best few
passages (plus some context) -> ONE small judge call -> the same quote check and strict recheck as
LLMVerifier. Sentences run on a thread pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from factcheck.module.app.text import chunk_spans
from factcheck.module.models import Judgement, Label, Reference, Sentence
from factcheck.module.utils.embed import Embedder, Reranker
from factcheck.module.utils.llm import LLMClient, LLMTransportError

from .prompts import PROMPT_VERSION, build_judge_prompt
from .verifier import JudgeItem, JudgeResponse, LLMVerifier, VerifyResult

_GAP = "\n\n[...]\n\n"  # between non-adjacent excerpts; `[...]` is ignored by the quote check

Vectors = NDArray[np.float64]


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of (start, end) ranges: overlapping or touching ones are joined, sorted by start."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _unit(vectors: Sequence[Sequence[float]]) -> Vectors:
    """Rows scaled to length 1, so a dot product is the cosine similarity."""
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class RetrievalVerifier(LLMVerifier):
    def __init__(
        self,
        client: LLMClient,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        passage_chars: int = 1000,
        passage_overlap: int = 250,
        recall_k: int = 20,
        top_k: int = 3,
        pad: int = 100,
        min_score: float | None = None,
        workers: int = 4,
        retries: int = 2,
    ) -> None:
        if not 1 <= top_k <= recall_k:
            raise ValueError("need 1 <= top_k <= recall_k")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        super().__init__(client, retries=retries)
        self.prompt_version = f"{PROMPT_VERSION}+retrieve"
        self._embedder = embedder
        self._reranker = reranker
        self._passage_chars = passage_chars
        self._passage_overlap = passage_overlap
        self._recall_k = recall_k
        self._top_k = top_k
        self._pad = pad
        self._min_score = min_score
        self._workers = workers
        self._claim_vecs: dict[str, Vectors] = {}  # claim -> unit vector, reused across references

    def verify(self, sentences: Sequence[Sentence], reference: Reference) -> VerifyResult:
        self._transport_failures = 0
        text = reference.text
        spans = [
            (a, b)
            for a, b in chunk_spans(text, self._passage_chars, self._passage_overlap)
            if text[a:b].strip()
        ]
        if not spans or not sentences:
            empty = {
                s.id: Judgement(sentence_id=s.id, ref_id=reference.id, label=Label.NOT_ENOUGH_INFO)
                for s in sentences
            }
            return VerifyResult(judgements=empty)
        passage_vecs = _unit(self._embedder.embed([text[a:b] for a, b in spans]))
        similarity = self._claim_matrix(sentences) @ passage_vecs.T  # (sentences, passages)

        judgements: dict[str, Judgement] = {}
        errors: list[str] = []
        calls = 0
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = [
                pool.submit(self._verify_sentence, s, similarity[i], spans, reference)
                for i, s in enumerate(sentences)
            ]
            for sentence, future in zip(sentences, futures, strict=True):
                judgement, used, found_errors = future.result()
                judgements[sentence.id] = judgement
                calls += used
                errors += found_errors
        return VerifyResult(judgements=judgements, errors=errors, calls=calls)

    def _claim_matrix(self, sentences: Sequence[Sentence]) -> Vectors:
        todo = sorted({s.claim for s in sentences} - self._claim_vecs.keys())
        if todo:
            for claim, vector in zip(
                todo, _unit(self._embedder.embed(todo, query=True)), strict=True
            ):
                self._claim_vecs[claim] = vector
        return np.stack([self._claim_vecs[s.claim] for s in sentences])

    def _verify_sentence(
        self,
        sentence: Sentence,
        similarity: Vectors,
        spans: Sequence[tuple[int, int]],
        reference: Reference,
    ) -> tuple[Judgement, int, list[str]]:
        text = reference.text
        where = f"{reference.id}/{sentence.id}"
        errors: list[str] = []
        recalled = [int(i) for i in np.argsort(-similarity)[: self._recall_k]]

        scores: list[float] | None = None
        if self._reranker is not None:
            try:
                scores = self._reranker.rerank(
                    sentence.claim, [text[spans[i][0] : spans[i][1]] for i in recalled]
                )
            except LLMTransportError as exc:
                errors.append(f"{where} rerank failed, using embedding order: {exc}")
        if scores is None:
            ranked = recalled
        else:
            if self._min_score is not None and max(scores) < self._min_score:
                skipped = Judgement(
                    sentence_id=sentence.id,
                    ref_id=reference.id,
                    label=Label.NOT_ENOUGH_INFO,
                    note="No passage scored above the rerank threshold",
                )
                return skipped, 0, errors
            order = sorted(zip(scores, recalled, strict=True), key=lambda p: p[0], reverse=True)
            ranked = [i for _, i in order]

        ranges = merge_ranges(
            [
                (max(0, spans[i][0] - self._pad), min(len(text), spans[i][1] + self._pad))
                for i in ranked[: self._top_k]
            ]
        )
        source = _GAP.join(text[a:b] for a, b in ranges)
        response = self._call(build_judge_prompt([sentence], source), JudgeResponse, where, errors)

        found: list[Judgement] = []
        for raw in response.results if response else []:
            try:
                item = JudgeItem.model_validate(raw)
            except ValidationError:
                errors.append(f"{where} skipped malformed item: {str(raw)[:80]}")
                continue
            if item.id != sentence.id:
                errors.append(f"{where} model returned a different sentence id: {item.id}")
                continue
            found.append(self._locate(item, reference.id, text, ranges))
        judgement, confirm_calls = self._resolve(
            sentence.id, sentence.claim, reference.id, found, errors
        )
        return judgement, 1 + confirm_calls, errors

    def _locate(
        self, item: JudgeItem, ref_id: str, text: str, ranges: Sequence[tuple[int, int]]
    ) -> Judgement:
        """The quote must lie inside one shown excerpt; its span is relative to the full text."""
        attempts = [self._check(item, ref_id, text[a:b], a) for a, b in ranges]
        return next((j for j in attempts if j.label is Label.SUPPORTED), attempts[0])
