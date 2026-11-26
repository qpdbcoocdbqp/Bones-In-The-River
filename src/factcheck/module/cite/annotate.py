from __future__ import annotations

from collections.abc import Callable, Sequence

from factcheck.module.app.text import parse_body
from factcheck.module.models import Annotation, BlockResult, Reference, RunMeta, SentenceResult

from .verifier import Preparing, Verifier


def annotate(
    body: str,
    references: Sequence[Reference],
    verifier: Verifier,
    on_progress: Callable[[str], None] | None = None,
) -> Annotation:
    ids = [r.id for r in references]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(
            f"Duplicate reference id (same file name, different extension?): {duplicated}"
        )
    blocks = parse_body(body)
    sentences = [s for block in blocks for s in block.sentences]
    per_reference = {}
    errors: list[str] = []
    calls = 0
    if isinstance(verifier, Preparing):
        verifier.prepare(sentences, references)
    for reference in references:
        if on_progress:
            on_progress(f"Checking {reference.id} ({len(sentences)} sentences)")
        result = verifier.verify(sentences, reference)
        per_reference[reference.id] = result.judgements
        errors += result.errors
        calls += result.calls

    block_results = [
        BlockResult(
            block=block,
            results=[
                SentenceResult(
                    sentence=s, judgements=[per_reference[r.id][s.id] for r in references]
                )
                for s in block.sentences
            ],
        )
        for block in blocks
    ]
    meta = RunMeta(
        model=verifier.model,
        prompt_version=verifier.prompt_version,
        ref_ids=[r.id for r in references],
        calls=calls,
        errors=errors,
    )
    return Annotation(meta=meta, blocks=block_results)
