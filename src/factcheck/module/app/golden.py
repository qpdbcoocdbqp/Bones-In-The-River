from __future__ import annotations

from typing import Any

from factcheck.module.app.text import parse_body
from factcheck.module.models import (
    Annotation,
    BlockResult,
    Judgement,
    Label,
    RunMeta,
    SentenceResult,
)

_LABELS = {"full": Label.SUPPORTED, "partial": Label.PARTIAL, "none": Label.NOT_ENOUGH_INFO}


def golden_annotation(body: str, golden: dict[str, Any]) -> Annotation:
    """Human golden labels (examples/<name>/golden.json) laid over the body, as an Annotation that
    `to_markdown` can render like a `mark` result.

    Sentences are matched by id (p<paragraph>s<n>, as `parse_body` numbers them); a golden sentence
    whose text differs from the body's means the body changed after labeling, which is an error.
    A body sentence with no golden entry is left without judgements (shown as [No source]).
    """
    entries = {e["id"]: e for e in golden["sentences"]}
    blocks = parse_body(body)
    body_ids = {s.id for b in blocks for s in b.sentences}
    if unknown := sorted(set(entries) - body_ids):
        raise ValueError(f"golden ids not found in body: {', '.join(unknown)}")

    results: list[BlockResult] = []
    for block in blocks:
        sentence_results = []
        for sentence in block.sentences:
            entry = entries.get(sentence.id)
            judgements: list[Judgement] = []
            if entry is not None:
                if " ".join(entry["sentence"].split()) != " ".join(sentence.text.split()):
                    raise ValueError(f"{sentence.id}: golden sentence differs from the body")
                label = _LABELS[entry["label"]]
                if label is not Label.NOT_ENOUGH_INFO:
                    judgements = [
                        Judgement(
                            sentence_id=sentence.id,
                            ref_id=c["ref"],
                            label=label,
                            quote=c["quote"],
                            note=entry.get("note", ""),
                        )
                        for c in entry["citations"] or []
                    ]
            sentence_results.append(SentenceResult(sentence=sentence, judgements=judgements))
        results.append(BlockResult(block=block, results=sentence_results))

    meta = RunMeta(
        model="human-golden",
        prompt_version="golden",
        calibrated=True,
        ref_ids=list(golden["refs"]),
    )
    return Annotation(meta=meta, blocks=results)
