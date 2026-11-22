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

## Prepare inputs (optional)
Starting from a clipped web article (Obsidian Web Clipper Markdown) whose "reference materials" list holds links:
```bash
PYTHONUTF8=1 PYTHONPATH=src python -m factcheck.cli prep \
  --clipping "dev/article.md" --out examples/myarticle
```
Writes `body.md` (frontmatter, title, date, intro and the link list removed), `refs/rN_<host or youtube_id>.md` (YouTube links → captions, other links → main page text) and `sources.json` (id → title, url, status). Failed downloads are listed and exit code is `1`; add those references to `refs/` by hand. Downloaded text is not cleaned beyond tag removal, check it before `mark`. `--no-download` writes `body.md` only.

`--browser` downloads with a Playwright-controlled Chromium instead (needs `python -m playwright install chromium` once; add `--headed` to see the window). Web pages are rendered with JavaScript before the text is taken; YouTube reads the page's transcript panel first and falls back to the caption API (which is often blocked by IP).

**Unusable references.** A reference with less than 500 characters (`--min-ref-chars`) or that looks like a bot-check / login / error page is treated as a failed download: `prep` does not save it, and `mark` skips it in the comparison. Both print the URL to fetch by hand and write `manual_refs.md` (`mark` finds the URL through the `sources.json` next to `refs/`). Save the text as `refs/<id>.md` and run again. `mark` exits `1` when it skipped any, because claims those references support show `[No source]`.

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
