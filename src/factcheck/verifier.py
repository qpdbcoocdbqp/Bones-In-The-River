from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import batched
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm import LLMClient, LLMFormatError, LLMTransportError, complete_json
from .models import Judgement, Label, Reference, Sentence
from .prompts import PROMPT_VERSION, SYSTEM, build_confirm_prompt, build_judge_prompt
from .text import chunk_spans, locate_quote, means_nothing

# the prompt asks for ~300 chars; over twice that is nearly a whole-passage paste, not evidence for
# "that sentence"
MAX_QUOTE_CHARS = 600
MAX_TRANSPORT_FAILURES = (
    3  # abort after this many consecutive connection failures instead of timing out on every batch
)

# Of several judgements for the same (sentence, reference), keep the best: supported >
# partially_supported > not_enough_info
_RANK = {Label.SUPPORTED: 0, Label.PARTIAL: 1, Label.NOT_ENOUGH_INFO: 2}


class JudgeItem(BaseModel):
    id: str
    quote: str
    note: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _strip_brackets(cls, value: object) -> object:
        """The prompt marks sentences as `[p1s1]`; the model sometimes returns the brackets too."""
        return value.strip().strip("[]").strip() if isinstance(value, str) else value


class JudgeResponse(BaseModel):
    # Items are not validated yet: one bad item (e.g. quote is null) must not invalidate the other
    # valid items in the batch
    results: list[Any] = Field(default_factory=list)


class ConfirmResponse(BaseModel):
    verdict: Literal["full", "partial", "unrelated"]
    missing: str = ""  # details the quote does not state


@dataclass
class VerifyResult:
    judgements: dict[
        str, Judgement
    ]  # sentence_id -> final judgement of that sentence against this reference
    errors: list[str] = field(default_factory=list)
    calls: int = 0


class Verifier(Protocol):
    model: str
    prompt_version: str

    def verify(self, sentences: Sequence[Sentence], reference: Reference) -> VerifyResult: ...


def pick_best(sentence_id: str, ref_id: str, candidates: Sequence[Judgement]) -> Judgement:
    """Pick the best judgement; on ties keep the first; among not_enough_info prefer one with an
    explanation (downgrade reason)."""
    if not candidates:
        return Judgement(sentence_id=sentence_id, ref_id=ref_id, label=Label.NOT_ENOUGH_INFO)
    return min(candidates, key=lambda j: (_RANK[j.label], not j.note))


class LLMVerifier:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_chars: int = 30_000,
        overlap: int = 500,
        batch_size: int = 20,
        retries: int = 2,
    ) -> None:
        self.model = client.model
        self.prompt_version = PROMPT_VERSION
        self._client = client
        self._max_chars = max_chars
        self._overlap = overlap
        self._batch_size = batch_size
        self._retries = retries
        self._transport_failures = 0

    def verify(self, sentences: Sequence[Sentence], reference: Reference) -> VerifyResult:
        self._transport_failures = 0
        candidates: dict[str, list[Judgement]] = {s.id: [] for s in sentences}
        errors: list[str] = []
        calls = 0
        for start, end in chunk_spans(reference.text, self._max_chars, self._overlap):
            chunk = reference.text[start:end]
            # consecutive calls on the same chunk (friendly to prompt caching)
            for batch in batched(sentences, self._batch_size):
                calls += 1
                where = f"{reference.id}[{start}:{end}]"
                response = self._call(
                    build_judge_prompt(batch, chunk), JudgeResponse, where, errors
                )
                if response is None:
                    continue
                batch_ids = {s.id for s in batch}
                for raw in response.results:
                    try:
                        item = JudgeItem.model_validate(raw)
                    except ValidationError:
                        errors.append(f"{where} skipped malformed item: {str(raw)[:80]}")
                        continue
                    if item.id not in batch_ids:
                        errors.append(
                            f"{where} model returned a sentence id not in this batch: {item.id}"
                        )
                        continue
                    candidates[item.id].append(self._check(item, reference.id, chunk, start))

        claims = {s.id: s.claim for s in sentences}
        judgements: dict[str, Judgement] = {}
        for sid, found in candidates.items():
            # Stage 2: strictly check supported candidates one by one until one passes. Checking
            # only the first would
            # be wrong: if it gets downgraded, a better quote in a later chunk would never get a
            # chance.
            confirmed: list[Judgement] = []
            for judgement in (j for j in found if j.label is Label.SUPPORTED):
                calls += 1
                confirmed.append(self._confirm(judgement, claims[sid], errors))
                if confirmed[-1].label is Label.SUPPORTED:
                    break
            rest = [j for j in found if j.label is Label.NOT_ENOUGH_INFO]
            judgements[sid] = pick_best(sid, reference.id, confirmed + rest)
        return VerifyResult(judgements=judgements, errors=errors, calls=calls)

    def _call[T: BaseModel](
        self, prompt: str, schema: type[T], where: str, errors: list[str]
    ) -> T | None:
        """Call and validate; on failure record it and return None. Abort the whole run after
        repeated connection failures."""
        try:
            result = complete_json(
                self._client, prompt, schema, system=SYSTEM, retries=self._retries
            )
        except LLMTransportError as exc:
            self._transport_failures += 1
            errors.append(f"{where} {exc}")
            if self._transport_failures >= MAX_TRANSPORT_FAILURES:
                raise LLMTransportError(
                    f"{MAX_TRANSPORT_FAILURES} consecutive connection failures, aborting: {exc}"
                ) from exc
            return None
        except LLMFormatError as exc:
            self._transport_failures = 0  # the server responded, just in the wrong format
            errors.append(f"{where} {exc}")
            return None
        self._transport_failures = 0
        return result

    def _confirm(self, judgement: Judgement, claim: str, errors: list[str]) -> Judgement:
        """Strictly check whether the quote ALONE fully states the sentence.

        full is kept; partial is downgraded to partially related; unrelated is downgraded to
        not_enough_info. If the check call fails, conservatively treat as partially related.
        """
        where = f"{judgement.ref_id}/{judgement.sentence_id} recheck"
        result = self._call(
            build_confirm_prompt(claim, judgement.quote or "", judgement.ref_id),
            ConfirmResponse,
            where,
            errors,
        )
        if result is None:
            return judgement.model_copy(
                update={
                    "label": Label.PARTIAL,
                    "note": "Recheck failed; full support not confirmed",
                }
            )
        if result.verdict == "full":
            return judgement
        if result.verdict == "partial":
            missing = "" if means_nothing(result.missing) else result.missing.strip()
            note = f"Missing: {missing}" if missing else judgement.note
            return judgement.model_copy(update={"label": Label.PARTIAL, "note": note})
        return judgement.model_copy(
            update={
                "label": Label.NOT_ENOUGH_INFO,
                "quote": None,
                "span": None,
                "note": "Recheck: the quote is unrelated to the sentence",
            }
        )

    @staticmethod
    def _check(item: JudgeItem, ref_id: str, chunk: str, offset: int) -> Judgement:
        """Check the model's quote: reasonable length, actually present in the chunk and uniquely
        locatable; otherwise downgrade."""

        def downgrade(reason: str) -> Judgement:
            return Judgement(
                sentence_id=item.id, ref_id=ref_id, label=Label.NOT_ENOUGH_INFO, note=reason
            )

        if len(item.quote) > MAX_QUOTE_CHARS:
            return downgrade(
                f"quote too long ({len(item.quote)} chars), "
                "cannot serve as evidence for the sentence"
            )
        span = locate_quote(item.quote, chunk)
        if span is None:
            return downgrade(
                "quote check failed (not in source, too short, "
                "or appears multiple times so cannot be located)"
            )
        start, end = span
        return Judgement(
            sentence_id=item.id,
            ref_id=ref_id,
            label=Label.SUPPORTED,
            quote=chunk[start:end],
            span=(offset + start, offset + end),
            note=item.note,
        )
