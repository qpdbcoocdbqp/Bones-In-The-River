import type { Doc, Evidence, Paragraph, Sentence, Status } from "./types.ts";

const TAG_RE =
  /\s*(?:【(來源|部分相關|無來源)(?:：([^】]*))?】|\[(Source|Partial|No source)(?::\s*([^\]]*))?\])/g;
const EVIDENCE_RE = /^>\s*-\s*(?:第\s*(\d+)\s*句|Sentence\s+(\d+))\s*(.*)$/;
const EVIDENCE_BODY_RE = /^(\S+)\s*[^*]*?\*\*([^*]+)\*\*[^：:]*[：:]\s*(.*)$/;
const OPEN_QUOTE_RE = /^[「"“]/;
const CLOSE_QUOTE_RE = /[」"”]/;
const ELLIPSIS_CLOSE_RE = /(?:…|\.\.\.)[」"”]/;
const MISSING_RE = /(?:缺少|Missing)\s*[：:]\s*/;
const SENTENCE_RE = /[^。！？!?]*[。！？!?]+[」』”）)]*|[^。！？!?]+$/g;

export function splitSentences(text: string): string[] {
  return (text.match(SENTENCE_RE) ?? [text]).map((s) => s.trim()).filter(Boolean);
}

function kindOfSymbol(sym: string): Evidence["kind"] {
  if (sym === "←") return "source";
  if (sym === "⚠") return "conflict";
  return "partial";
}

/**
 * 引用內文可能含有同樣的引號字元，不能取第一個結尾引號。
 * 優先順序：被截斷的引用以「…」收尾；其次若有「缺少/Missing」，取它之前最後一個引號；最後才取第一個結尾引號。
 */
export function splitQuote(tail: string): { quote: string; after: string } {
  if (!OPEN_QUOTE_RE.test(tail)) return { quote: "", after: tail };
  const body = tail.slice(1);
  let end = -1;
  const ell = ELLIPSIS_CLOSE_RE.exec(body);
  if (ell) {
    end = ell.index + ell[0].length - 1;
  } else {
    const miss = MISSING_RE.exec(body);
    const seg = miss ? body.slice(0, miss.index) : body;
    if (miss) {
      end = Math.max(seg.lastIndexOf("」"), seg.lastIndexOf('"'), seg.lastIndexOf("”"));
    } else {
      end = seg.search(CLOSE_QUOTE_RE);
    }
  }
  if (end < 0) return { quote: body.trim(), after: "" };
  return { quote: body.slice(0, end).trim(), after: body.slice(end + 1).trim() };
}

function parseEvidence(rest: string): Evidence | null {
  const m = EVIDENCE_BODY_RE.exec(rest);
  if (!m) return null;
  const [, sym, refId, tail] = m;
  const { quote, after } = splitQuote(tail);
  const [note, missing = ""] = after.split(MISSING_RE, 2);
  return { kind: kindOfSymbol(sym), refId, quote, note: note.trim(), missing: missing.trim() };
}

function paragraphStatus(kinds: string[]): Status {
  if (kinds.length === 0) return "unmarked";
  if (kinds.every((k) => k === "none")) return "none";
  return kinds.includes("partial") ? "partial" : "source";
}

function sentenceStatus(ev: Evidence[], para: Status): Status {
  if (ev.some((e) => e.kind === "conflict")) return "conflict";
  if (ev.some((e) => e.kind === "source")) return "source";
  if (ev.length > 0) return "partial";
  return para === "none" ? "none" : "unmarked";
}

function buildParagraph(raw: string, evLines: Map<number, Evidence[]>): Paragraph {
  const kinds: string[] = [];
  const refs: string[] = [];
  const text = raw.replace(TAG_RE, (_m, zh: string | undefined, zhRefs: string | undefined, en: string | undefined, enRefs: string | undefined) => {
    const label = zh ?? en ?? "";
    kinds.push(label === "來源" || label === "Source" ? "source" : label === "部分相關" || label === "Partial" ? "partial" : "none");
    for (const r of (zhRefs ?? enRefs ?? "").split(/[、,，]\s*/)) if (r.trim()) refs.push(r.trim());
    return "";
  }).trim();
  const status = paragraphStatus(kinds);
  const sentences: Sentence[] = splitSentences(text).map((t, i) => {
    const evidence = evLines.get(i + 1) ?? [];
    return { index: i + 1, text: t, status: sentenceStatus(evidence, status), evidence };
  });
  return { status, refs: [...new Set(refs)], sentences };
}

export function parseAnnotated(name: string, md: string): Doc {
  const notice: string[] = [];
  const doc: Doc = {
    name,
    notice,
    sections: [],
    counts: { source: 0, partial: 0, conflict: 0, none: 0, unmarked: 0 },
  };
  let section = { title: "", paragraphs: [] as Paragraph[] };
  let pending: { raw: string; ev: Map<number, Evidence[]> } | null = null;

  const flush = () => {
    if (pending) section.paragraphs.push(buildParagraph(pending.raw, pending.ev));
    pending = null;
  };
  const closeSection = () => {
    flush();
    if (section.paragraphs.length > 0 || section.title) doc.sections.push(section);
  };

  for (const line of md.split(/\r?\n/)) {
    const ev = EVIDENCE_RE.exec(line);
    if (ev && pending) {
      const parsed = parseEvidence(ev[3]);
      if (parsed) {
        const n = Number(ev[1] ?? ev[2]);
        pending.ev.set(n, [...(pending.ev.get(n) ?? []), parsed]);
      }
      continue;
    }
    if (line.startsWith(">")) {
      if (doc.sections.length === 0 && !section.title && !pending) notice.push(line.replace(/^>\s?/, ""));
      continue;
    }
    const h = /^#{1,6}\s+(.*)$/.exec(line);
    if (h) {
      closeSection();
      section = { title: h[1].trim(), paragraphs: [] };
      continue;
    }
    if (line.trim() === "") {
      flush();
      continue;
    }
    if (pending) flush();
    pending = { raw: line, ev: new Map() };
  }
  closeSection();

  for (const s of doc.sections)
    for (const p of s.paragraphs) for (const sent of p.sentences) doc.counts[sent.status]++;
  return doc;
}
