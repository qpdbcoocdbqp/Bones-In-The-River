from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import get_args

from .models import (
    Annotation,
    GoldLevel,
    GoldRow,
    ManualRef,
    Reference,
    RefResult,
    SampleMeta,
)

REFERENCE_SUFFIXES = {".txt", ".md"}
MIN_REF_CHARS = 500  # a shorter reference is a failed download (error page, login wall, stub)
_BLOCK_PAGE_MAX_CHARS = 3000
# phrases of bot-check / login-wall / error pages; only looked for in short texts, because a long
# transcript may legitimately say "access denied"
_BLOCK_MARKERS = (
    "failed to verify your browser",
    "security checkpoint",
    "just a moment",
    "checking your browser",
    "verify you are human",
    "captcha",
    "access denied",
    "enable javascript",
    "content restricted",
    "upgrade to access",
    "sign in to continue",
    "log in to continue",
    "403 forbidden",
    "404 not found",
)


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


def ref_problem(text: str, min_chars: int = MIN_REF_CHARS) -> str | None:
    """Why this reference text cannot be trusted as the page content, or None if it looks fine."""
    stripped = text.strip()
    lowered = stripped.lower()
    marker = next((m for m in _BLOCK_MARKERS if m in lowered), None)
    if len(stripped) < min_chars:
        suffix = f"; looks like a block / error page ({marker!r})" if marker else ""
        return f"only {len(stripped)} characters (< {min_chars}){suffix}"
    if marker and len(stripped) < _BLOCK_PAGE_MAX_CHARS:
        return f"looks like a block / error page ({marker!r})"
    return None


def load_sources(refs_dir: Path) -> dict[str, RefResult]:
    """sources.json written by `prep` next to the refs folder (id -> title, url); {} if absent."""
    path = refs_dir.parent / "sources.json"
    if not path.is_file():
        return {}
    entries = (RefResult.model_validate(e) for e in json.loads(path.read_text(encoding="utf-8")))
    return {e.id: e for e in entries}


def screen_references(
    references: list[Reference], refs_dir: Path, min_chars: int = MIN_REF_CHARS
) -> tuple[list[Reference], list[ManualRef]]:
    """Split references into usable ones and ones whose content failed `ref_problem`; the latter
    carry their URL (from sources.json, when known) so they can be fetched by hand."""
    sources = load_sources(refs_dir)
    usable: list[Reference] = []
    skipped: list[ManualRef] = []
    for ref in references:
        problem = ref_problem(ref.text, min_chars)
        if problem is None:
            usable.append(ref)
            continue
        source = sources.get(ref.id)
        skipped.append(
            ManualRef(
                id=ref.id,
                title=source.title if source else ref.id,
                url=source.url if source else None,
                reason=problem,
            )
        )
    return usable, skipped


def write_manual_list(items: list[ManualRef], path: Path) -> None:
    """Markdown checklist of the references to fetch by hand; removes a stale file when empty."""
    if not items:
        path.unlink(missing_ok=True)
        return
    lines = ["# References to fetch by hand", ""]
    lines.append("Save each page's text as `refs/<id>.md` (same id), then run again.")
    lines.append("")
    for item in items:
        target = f"[{item.title}]({item.url})" if item.url else item.title
        lines.append(f"- `{item.id}` {target} - {item.reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
