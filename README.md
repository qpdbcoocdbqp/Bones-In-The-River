# Bones In The River

Tracing cite. Playing with [Bones In The River](https://www.youtube.com/watch?v=iOveXLWjpZE)

> Bones In The River·The Devil Inside
>
> Bones In The River


## Citation Source Marker

Give it an article and some reference docs. It marks, paragraph by paragraph, which reference (and which sentence) supports each claim.

> Not calibrated. Output is a suggestion, **review it yourself**.

## Install
Python >= 3.12.

```bash
uv sync
```

Then put a `.env` in the project root (has secrets, **don't commit**):
```
BASE_URL=<OpenAI-compatible endpoint, e.g. local llama.cpp>
API_KEY=<key>
MODEL=<model name>
```
Environment variables win over `.env`.

## Run
```bash
PYTHONUTF8=1 PYTHONPATH=src python -m factcheck.cli mark \
  --body examples/shopify/body.md \
  --refs examples/shopify/refs \
  --out out/shopify
```

| Option | What it does |
|---|---|
| `--body` | The article (`.md` / `.txt`), paragraphs split by blank lines |
| `--refs` | Folder of reference `.txt` / `.md` files (file name = source id). Prepare the text yourself |
| `--out` | Output folder |
| `--model` | Override `MODEL` |
| `--max-chars` | Split a reference into chunks above this length (default 30000) |
| `--batch-size` | Sentences per call (default 20) |
| `--retries` | Retry count |

Exit codes: `0` ok, `1` some calls failed (results written but incomplete), `2` couldn't run (nothing written).

## Output
- `out/.../annotated.md`: your article plus `[Source: id]` / `[Partial: id]` / `[No source]` after each paragraph. `[No source]` doesn't mean it's wrong.
- `out/.../annotations.json`: per-sentence details.

## Dev checks
```bash
export PYTHONUTF8=1
python -m ruff check src tests ; python -m mypy
PYTHONPATH=src lint-imports
python -m pytest -q
```
