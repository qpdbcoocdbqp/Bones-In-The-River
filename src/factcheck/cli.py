from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .annotate import annotate
from .evaluate import (
    ANY_RELEVANT,
    FULL,
    GOLD_ANY,
    GOLD_FULL,
    Score,
    cohen_kappa,
    score_paragraphs,
    score_sentences,
    sentence_ids,
    weights_from_meta,
)
from .io import (
    load_annotation,
    load_body,
    load_references,
    read_gold,
    read_sample_meta,
    write_label_sheet,
    write_outputs,
    write_sample_meta,
)
from .llm import LLMError, OpenAICompatClient
from .render import to_markdown
from .sheet import sample_meta, sheet_rows
from .verifier import LLMVerifier

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Citation source marker: match the body against references and mark the source of each
    paragraph."""


@app.command()
def mark(
    body: Annotated[Path, typer.Option(help="Body file (Markdown / plain text)")],
    refs: Annotated[Path, typer.Option(help="Reference folder (one .txt / .md per reference)")],
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("out"),
    env_file: Annotated[Path, typer.Option(help=".env containing BASE_URL, API_KEY, MODEL")] = Path(
        ".env"
    ),
    model: Annotated[str | None, typer.Option(help="Model name; defaults to MODEL in .env")] = None,
    max_chars: Annotated[
        int, typer.Option(help="Split a reference into chunks only above this length")
    ] = 30_000,
    batch_size: Annotated[int, typer.Option(help="Sentences per call")] = 20,
    retries: Annotated[int, typer.Option(help="Retries on malformed responses")] = 2,
) -> None:
    try:
        client = OpenAICompatClient(model, env_file)
    except (LLMError, ImportError) as exc:
        typer.echo(f"Cannot create LLM backend: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    verifier = LLMVerifier(client, max_chars=max_chars, batch_size=batch_size, retries=retries)
    try:
        annotation = annotate(
            load_body(body),
            load_references(refs),
            verifier,
            on_progress=lambda msg: typer.echo(msg, err=True),
        )
    except (LLMError, ValueError, OSError) as exc:
        typer.echo(f"Run failed, no output written: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    md_path, json_path = write_outputs(annotation, to_markdown(annotation), out)
    typer.echo(f"Wrote {md_path} and {json_path}")
    if annotation.meta.errors:
        typer.echo(
            f"WARNING: {len(annotation.meta.errors)} call(s) failed, result is incomplete: "
            "the failed parts were not checked, "
            'so "no source" does not mean nothing exists. See meta.errors in annotations.json.',
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def sheet(
    annotations: Annotated[Path, typer.Option(help="annotations.json produced by mark")],
    out: Annotated[Path, typer.Option(help="Output path of the label sheet CSV")] = Path(
        "label_sheet.csv"
    ),
    n: Annotated[int, typer.Option(help="Number of sentences to sample")] = 30,
    seed: Annotated[int, typer.Option(help="Sampling random seed")] = 0,
    hints: Annotated[
        bool,
        typer.Option(help="Include system output (off by default to avoid anchoring annotators)"),
    ] = False,
) -> None:
    """Stratified-sample sentences into a label sheet.

    Annotators fill gold with full/partial/none and gold_refs with source ids.
    """
    annotation = load_annotation(annotations)
    meta_path = out.with_suffix(".strata.json")
    try:
        rows = sheet_rows(annotation, n=n, seed=seed, hints=hints)
        write_label_sheet(rows, out)
    except ValueError as exc:
        typer.echo(f"Cannot create label sheet: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    write_sample_meta(sample_meta(annotation, n=n, seed=seed), meta_path)
    typer.echo(
        f"Wrote {out} ({len(rows)} sentences; {'with' if hints else 'without'} system output)"
    )
    typer.echo(f"Sampling info (for weighting in eval --strata): {meta_path}")


def _fmt(score: Score) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    return (
        f"precision={pct(score.precision)} recall={pct(score.recall)} "
        f"(TP={score.tp:.1f}, system positive={score.system:.1f}, gold positive={score.gold:.1f})"
    )


@app.command("eval")
def evaluate_command(
    annotations: Annotated[Path, typer.Option(help="annotations.json produced by mark")],
    gold: Annotated[Path, typer.Option(help="Filled-in label sheet CSV")],
    gold2: Annotated[Path | None, typer.Option(help="Second annotator CSV, for agreement")] = None,
    min_precision: Annotated[
        float | None, typer.Option(help="Threshold: sentence-level precision of full support")
    ] = None,
    min_recall: Annotated[
        float | None, typer.Option(help="Threshold: sentence-level recall of full support")
    ] = None,
    strata: Annotated[
        Path | None,
        typer.Option(
            help="The .strata.json from sheet; if given, sentence-level metrics are weighted "
            "(recommended)"
        ),
    ] = None,
) -> None:
    """Evaluate precision / recall against human labels.

    With thresholds, pass/fail is decided (exit code 1 on failure).
    """
    annotation = load_annotation(annotations)
    labels = read_gold(gold)
    unknown = {r for row in labels.values() for r in row.refs} - set(annotation.meta.ref_ids)
    if unknown:
        typer.echo(f"gold_refs contains unknown reference ids: {sorted(unknown)}", err=True)
        raise typer.Exit(code=2)
    stray = sorted(labels.keys() - sentence_ids(annotation))
    if stray:
        typer.echo(
            "Label sheet has sentence_ids not in this annotations.json "
            f"(possibly a different parse): {stray[:5]}...",
            err=True,
        )
        raise typer.Exit(code=2)
    weights = None
    if strata is None:
        typer.echo(
            "WARNING: --strata not given: the sheet is stratified, so the numbers below describe "
            "only the sampled sentences, not the whole article."
        )
    else:
        weights = weights_from_meta(read_sample_meta(strata))
        if labels.keys() - weights.keys():
            typer.echo(
                "Labeled sentences are missing from the sampling info, cannot weight", err=True
            )
            raise typer.Exit(code=2)
        typer.echo(
            "Sentence-level metrics are weighted by stratum (whole-article estimate); "
            "paragraph-level metrics are unweighted."
        )
    meta = annotation.meta
    typer.echo(f"Labeled {len(labels)} sentences; model {meta.model}, prompt {meta.prompt_version}")
    full = score_sentences(annotation, labels, FULL, GOLD_FULL, weights)
    typer.echo(f"Full support, per sentence: {_fmt(full)}")
    full_paragraph = score_paragraphs(annotation, labels, FULL, GOLD_FULL)
    typer.echo(f"Full support, per paragraph: {_fmt(full_paragraph)}")
    any_sentence = score_sentences(annotation, labels, ANY_RELEVANT, GOLD_ANY, weights)
    typer.echo(f"Any relevance, per sentence: {_fmt(any_sentence)}")
    any_paragraph = score_paragraphs(annotation, labels, ANY_RELEVANT, GOLD_ANY)
    typer.echo(f"Any relevance, per paragraph: {_fmt(any_paragraph)}")
    if gold2 is not None:
        kappa = cohen_kappa(labels, read_gold(gold2))
        typer.echo("Annotator agreement kappa: " + ("n/a" if kappa is None else f"{kappa:.2f}"))
    checks = (("precision", full.precision, min_precision), ("recall", full.recall, min_recall))
    failed = [
        name
        for name, value, floor in checks
        if floor is not None and (value is None or value < floor)
    ]
    if failed:
        typer.echo(f"Below threshold: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
