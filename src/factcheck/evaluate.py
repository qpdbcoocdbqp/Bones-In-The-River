from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from .models import Annotation, GoldLevel, GoldRow, Label, SampleMeta, SentenceResult

FULL: frozenset[Label] = frozenset({Label.SUPPORTED})
ANY_RELEVANT: frozenset[Label] = frozenset({Label.SUPPORTED, Label.PARTIAL})
GOLD_FULL: frozenset[GoldLevel] = frozenset({"full"})
GOLD_ANY: frozenset[GoldLevel] = frozenset({"full", "partial"})


@dataclass(frozen=True)
class Score:
    tp: float
    system: float  # number judged positive by the system (estimated population count when weighted)
    gold: float  # number labeled positive by humans (same)

    @property
    def precision(self) -> float | None:
        return self.tp / self.system if self.system else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.gold if self.gold else None


def weights_from_meta(meta: SampleMeta) -> dict[str, float]:
    """Each sampled sentence stands for (stratum population / stratum sample size) sentences."""
    sampled: dict[str, int] = {}
    for stratum in meta.sampled.values():
        sampled[stratum] = sampled.get(stratum, 0) + 1
    return {
        sid: meta.population[stratum] / sampled[stratum] for sid, stratum in meta.sampled.items()
    }


def sentence_ids(annotation: Annotation) -> set[str]:
    return {r.sentence.id for block in annotation.blocks for r in block.results}


def _refs(result: SentenceResult, labels: Collection[Label]) -> set[str]:
    return {j.ref_id for label in labels for j in result.with_label(label)}


def score_sentences(
    annotation: Annotation,
    gold: Mapping[str, GoldRow],
    system_labels: Collection[Label],
    gold_levels: Collection[GoldLevel],
    weights: Mapping[str, float] | None = None,
) -> Score:
    """Per sentence: TP only if system and human are both positive and at least one source id
    matches. Only labeled sentences count."""
    tp = system = positives = 0.0
    for block in annotation.blocks:
        for result in block.results:
            row = gold.get(result.sentence.id)
            if row is None:
                continue
            w = weights[result.sentence.id] if weights else 1.0
            system_refs = _refs(result, system_labels)
            is_gold_positive = row.level in gold_levels
            system += w * bool(system_refs)
            positives += w * is_gold_positive
            tp += w * (bool(system_refs) and is_gold_positive and bool(system_refs & row.refs))
    return Score(tp=tp, system=system, gold=positives)


def score_paragraphs(
    annotation: Annotation,
    gold: Mapping[str, GoldRow],
    system_labels: Collection[Label],
    gold_levels: Collection[GoldLevel],
) -> Score:
    """Per paragraph: compare the set of source ids over labeled sentences only (micro average).
    **Unweighted**."""
    tp = system = positives = 0
    for block in annotation.blocks:
        system_refs: set[str] = set()
        gold_refs: set[str] = set()
        for result in block.results:
            row = gold.get(result.sentence.id)
            if row is None:
                continue
            system_refs |= _refs(result, system_labels)
            if row.level in gold_levels:
                gold_refs |= row.refs
        tp += len(system_refs & gold_refs)
        system += len(system_refs)
        positives += len(gold_refs)
    return Score(tp=tp, system=system, gold=positives)


def cohen_kappa(a: Mapping[str, GoldRow], b: Mapping[str, GoldRow]) -> float | None:
    """Agreement of two annotators on level (full/partial/none) over commonly labeled sentences.
    Returns None if undefined."""
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    labels = sorted({a[s].level for s in shared} | {b[s].level for s in shared})
    n = len(shared)
    observed = sum(a[s].level == b[s].level for s in shared) / n
    expected = sum(
        (sum(a[s].level == lv for s in shared) / n) * (sum(b[s].level == lv for s in shared) / n)
        for lv in labels
    )
    if expected == 1:
        return None  # both annotators used a single label, kappa is undefined
    return (observed - expected) / (1 - expected)
