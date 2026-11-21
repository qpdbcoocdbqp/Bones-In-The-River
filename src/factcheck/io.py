from __future__ import annotations

import csv
from pathlib import Path
from typing import get_args

from .models import Annotation, GoldLevel, GoldRow, Reference, SampleMeta

REFERENCE_SUFFIXES = {".txt", ".md"}


def load_body(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_references(directory: Path) -> list[Reference]:
    """Each .txt / .md in the folder is one reference; the file name (without extension) is its id;
    sorted by name."""
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in REFERENCE_SUFFIXES)
    references = [Reference(id=p.stem, text=p.read_text(encoding="utf-8")) for p in files]
    references = [r for r in references if r.text.strip()]
    if not references:
        raise FileNotFoundError(f"No usable .txt / .md references in {directory}")
    return references


def load_annotation(path: Path) -> Annotation:
    return Annotation.model_validate_json(path.read_text(encoding="utf-8"))


def write_outputs(annotation: Annotation, markdown: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "annotated.md"
    json_path = out_dir / "annotations.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path


SHEET_ENCODING = "utf-8-sig"  # lets Excel recognize UTF-8


def write_sample_meta(meta: SampleMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def read_sample_meta(path: Path) -> SampleMeta:
    return SampleMeta.model_validate_json(path.read_text(encoding="utf-8"))


def write_label_sheet(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("No sentences to sample (body has no paragraphs, or --n is 0)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=SHEET_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_gold(path: Path) -> dict[str, GoldRow]:
    """Read the label sheet. Rows with an empty `gold` column are treated as unlabeled and skipped;
    when gold is full / partial, `gold_refs` (reference ids separated by ;) must not be empty."""
    levels = get_args(GoldLevel)
    gold: dict[str, GoldRow] = {}
    with path.open(encoding=SHEET_ENCODING, newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            level = (row.get("gold") or "").strip().lower()
            if not level:
                continue
            sentence_id = (row.get("sentence_id") or "").strip()
            if level not in levels:
                raise ValueError(
                    f"{path.name} line {line_no}: gold must be one of "
                    f"{'/'.join(levels)}, got {level!r}"
                )
            refs = frozenset(
                r.strip() for r in (row.get("gold_refs") or "").split(";") if r.strip()
            )
            if level != "none" and not refs:
                raise ValueError(
                    f"{path.name} line {line_no}: gold_refs must not be empty when gold={level}"
                )
            if sentence_id in gold:
                raise ValueError(f"{path.name} line {line_no}: duplicate sentence_id {sentence_id}")
            gold[sentence_id] = GoldRow.model_validate(
                {"sentence_id": sentence_id, "level": level, "refs": refs}
            )
    return gold
