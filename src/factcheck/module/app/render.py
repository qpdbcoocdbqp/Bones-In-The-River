from __future__ import annotations

from factcheck.module.models import Annotation, BlockResult, Label

NO_SOURCE = "[No source]"
_QUOTE_LIMIT = 200


def _short(quote: str) -> str:
    quote = " ".join(quote.split())
    return quote if len(quote) <= _QUOTE_LIMIT else quote[:_QUOTE_LIMIT] + "…"


_MARKERS = (
    (Label.SUPPORTED, "Source"),
    (Label.PARTIAL, "Partial"),
)


def _marker(result: BlockResult) -> str:
    parts = [
        f"[{name}: {', '.join(ids)}]" for label, name in _MARKERS if (ids := result.ref_ids(label))
    ]
    return "".join(parts) or NO_SOURCE


_DETAIL_PREFIX = {
    Label.SUPPORTED: "← **{ref}**",
    Label.PARTIAL: "◐ Partially related **{ref}**",
}


def _details(result: BlockResult) -> list[str]:
    lines = []
    for n, sentence_result in enumerate(result.results, start=1):
        for label, prefix in _DETAIL_PREFIX.items():
            for j in sentence_result.with_label(label):
                head = prefix.format(ref=j.ref_id)
                lines.append(f'> - Sentence {n} {head}: "{_short(j.quote or "")}" {j.note}')
    return lines


def to_markdown(annotation: Annotation) -> str:
    meta = annotation.meta
    header = [
        f"> ⚠ Judged by an LLM (model `{meta.model}`, prompt `{meta.prompt_version}`), "
        f"**{'calibrated' if meta.calibrated else 'not calibrated'}; for reference only, "
        "needs human review**.",
        f"> References: {', '.join(meta.ref_ids)}; "
        f"calls: {meta.calls}; errors: {len(meta.errors)}.",
        '> "No source" only means no support was found in the given references; '
        "it does not mean the content is wrong.",
    ]
    if meta.errors:
        header.append(
            f"> ⚠⚠ **{len(meta.errors)} call(s) failed, result is incomplete**: "
            'the failed parts were not checked, so "No source" does not mean nothing exists. '
            "See meta.errors in annotations.json."
        )
    body: list[str] = []
    for result in annotation.blocks:
        if result.block.kind == "verbatim":
            body.append(result.block.text)
            continue
        lines = [f"{result.block.text}  {_marker(result)}", *_details(result)]
        body.append("\n".join(lines))
    return "\n".join(header) + "\n\n" + "\n\n".join(body) + "\n"
