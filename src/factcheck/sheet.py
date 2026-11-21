from __future__ import annotations

import random

from .models import Annotation, Label, SampleMeta, SentenceResult

STRATA = ("supported", "partial", "none")


def stratum(result: SentenceResult) -> str:
    if result.with_label(Label.SUPPORTED):
        return "supported"
    if result.with_label(Label.PARTIAL):
        return "partial"
    return "none"


def sample_sentences(annotation: Annotation, n: int, seed: int) -> list[SentenceResult]:
    """Take about n/3 from each of the three strata; shortfalls are filled from other strata.
    Returned in body order (stratum not revealed)."""
    pools: dict[str, list[SentenceResult]] = {name: [] for name in STRATA}
    for block in annotation.blocks:
        for result in block.results:
            pools[stratum(result)].append(result)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    quota = {name: min(len(pools[name]), n // len(STRATA)) for name in STRATA}
    chosen = {name: pools[name][: quota[name]] for name in STRATA}
    leftover = [r for name in STRATA for r in pools[name][quota[name] :]]
    rng.shuffle(leftover)
    shortfall = n - sum(len(v) for v in chosen.values())
    picked = {id(r) for v in chosen.values() for r in v} | {id(r) for r in leftover[:shortfall]}
    return [r for block in annotation.blocks for r in block.results if id(r) in picked]


def sample_meta(annotation: Annotation, n: int = 30, seed: int = 0) -> SampleMeta:
    """Same sampling as sheet_rows (same seed, same result); returns stratum population sizes and
    each sentence's stratum."""
    population = dict.fromkeys(STRATA, 0)
    for block in annotation.blocks:
        for result in block.results:
            population[stratum(result)] += 1
    sampled = {r.sentence.id: stratum(r) for r in sample_sentences(annotation, n, seed)}
    return SampleMeta(population=population, sampled=sampled)


def sheet_rows(
    annotation: Annotation, n: int = 30, seed: int = 0, hints: bool = False
) -> list[dict[str, str]]:
    paragraph_of = {r.sentence.id: b.block.text for b in annotation.blocks for r in b.results}
    refs = ";".join(annotation.meta.ref_ids)
    rows = []
    for result in sample_sentences(annotation, n, seed):
        row = {
            "sentence_id": result.sentence.id,
            "sentence": result.sentence.claim,
            "paragraph": paragraph_of[result.sentence.id],
            "available_refs": refs,
            "gold": "",  # fill in full / partial / none
            "gold_refs": "",  # reference ids when gold is full / partial, separated by ;
            "notes": "",
        }
        if hints:
            found = result.with_label(Label.SUPPORTED) + result.with_label(Label.PARTIAL)
            row["system_label"] = stratum(result)
            row["system_refs"] = ";".join(j.ref_id for j in found)
            row["system_quote"] = " ".join((found[0].quote or "").split()) if found else ""
        rows.append(row)
    return rows
