from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Label(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = (
        "partially_supported"  # relevant but incomplete: the quote covers only part of the sentence
    )
    NOT_ENOUGH_INFO = "not_enough_info"


class Reference(BaseModel):
    id: str
    text: str


class Sentence(BaseModel):
    id: str  # e.g. "p3s2": sentence 2 of paragraph 3
    text: str  # original text (Markdown kept)
    claim: str  # plain text sent to the judge (link syntax removed)


class Block(BaseModel):
    kind: Literal["verbatim", "paragraph"]  # verbatim: headings, rules; emitted as-is, not verified
    text: str
    sentences: list[Sentence] = Field(default_factory=list)


class Judgement(BaseModel):
    """One sentence's judgement against one reference. quote and span are set only if the quote
    check passed."""

    sentence_id: str
    ref_id: str
    label: Label
    quote: str | None = None
    span: tuple[int, int] | None = None  # character range of the quote in the full reference text
    note: str = ""


class SentenceResult(BaseModel):
    sentence: Sentence
    judgements: list[Judgement] = Field(default_factory=list)

    def with_label(self, label: Label) -> list[Judgement]:
        return [j for j in self.judgements if j.label is label]


class BlockResult(BaseModel):
    block: Block
    results: list[SentenceResult] = Field(default_factory=list)

    def ref_ids(self, label: Label) -> list[str]:
        """Union of reference ids judged as `label` over all sentences in the paragraph, in order of
        first appearance."""
        seen: dict[str, None] = {}
        for result in self.results:
            for judgement in result.with_label(label):
                seen.setdefault(judgement.ref_id)
        return list(seen)


class RunMeta(BaseModel):
    model: str
    prompt_version: str
    calibrated: bool = False
    ref_ids: list[str]
    calls: int = 0
    errors: list[str] = Field(default_factory=list)


class Annotation(BaseModel):
    meta: RunMeta
    blocks: list[BlockResult]


class SampleMeta(BaseModel):
    """Sampling info of the label sheet: stratum population sizes, and the stratum of each sampled
    sentence. Used for weighting in evaluation."""

    population: dict[str, int]
    sampled: dict[str, str]  # sentence_id -> stratum


GoldLevel = Literal["full", "partial", "none"]


class GoldRow(BaseModel):
    """Human label: whether the sentence is fully supported / partially related / unsourced in the
    given references, and by which."""

    sentence_id: str
    level: GoldLevel
    refs: frozenset[str] = frozenset()


class RefResult(BaseModel):
    """Outcome of downloading one reference link (one entry of sources.json)."""

    id: str
    title: str
    url: str
    file: str | None  # None when the download failed or the text was unusable
    error: str | None = None


class ManualRef(BaseModel):
    """A reference whose content is unusable and has to be fetched by hand."""

    id: str
    title: str
    url: str | None
    reason: str
