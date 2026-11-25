import type { Doc, Sentence, Status } from "./types.ts";

declare const __DOCS__: Doc[];

const docs = __DOCS__;
const ORDER: Status[] = ["source", "partial", "conflict", "none", "unmarked"];
const LABEL: Record<Status, string> = { source: "有來源", partial: "部分相關", conflict: "與資料矛盾", none: "無來源", unmarked: "未標註" };
const GLYPH: Record<Status, string> = { source: "←", partial: "◐", conflict: "⚠", none: "✗", unmarked: "·" };

type Filter = Status | "all";
const state = {
  doc: Math.max(0, docs.findIndex((d) => d.name === decodeURIComponent(location.hash.slice(1)))),
  filter: "all" as Filter,
  query: "",
  selected: null as Sentence | null,
};

type Child = Node | string | null;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, string> = {}, ...kids: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) el.setAttribute(k, v);
  for (const c of kids) if (c !== null) el.append(c);
  return el;
}

/** 行內格式只處理 [text](url) 與 **bold**，其餘一律當純文字。 */
function inline(text: string, query: string): Node[] {
  const out: Node[] = [];
  const re = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|\*\*([^*]+)\*\*/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    out.push(...highlight(text.slice(last, m.index), query));
    out.push(m[1] ? h("a", { href: m[2], target: "_blank", rel: "noopener" }, m[1]) : h("b", {}, ...highlight(m[3], query)));
    last = m.index + m[0].length;
  }
  out.push(...highlight(text.slice(last), query));
  return out;
}

function highlight(text: string, query: string): Node[] {
  if (!query) return [document.createTextNode(text)];
  const out: Node[] = [];
  const lower = text.toLowerCase();
  let i = 0;
  for (let j = lower.indexOf(query); j >= 0; j = lower.indexOf(query, i)) {
    out.push(document.createTextNode(text.slice(i, j)), h("mark", {}, text.slice(j, j + query.length)));
    i = j + query.length;
  }
  out.push(document.createTextNode(text.slice(i)));
  return out;
}

function bar(counts: Record<Status, number>): HTMLElement {
  const total = ORDER.reduce((n, s) => n + counts[s], 0) || 1;
  const el = h("div", { class: "bar" });
  for (const s of ORDER) {
    if (!counts[s]) continue;
    const seg = h("i", { class: `s-${s}`, title: `${LABEL[s]} ${counts[s]}` });
    seg.style.width = `${(counts[s] / total) * 100}%`;
    el.append(seg);
  }
  return el;
}

const $ = (id: string) => document.getElementById(id)!;

function renderHeader(): void {
  const doc = docs[state.doc];
  const tabs = h("div", { class: "tabs", role: "tablist", "aria-label": "選擇文件" });
  docs.forEach((d, i) => {
    const b = h("button", { type: "button", role: "tab", "aria-selected": String(i === state.doc) }, d.name);
    b.onclick = () => {
      state.doc = i;
      state.selected = null;
      location.hash = encodeURIComponent(d.name);
      render();
    };
    tabs.append(b);
  });
  const total = ORDER.reduce((n, s) => n + doc.counts[s], 0);
  const filters = h("div", { class: "chips", role: "group", "aria-label": "篩選（同時顯示句數）" });
  for (const f of ["all", ...ORDER] as Filter[]) {
    const n = f === "all" ? total : doc.counts[f];
    if (f !== "all" && !n) continue;
    const b = h("button", { type: "button", class: "chip", "aria-pressed": String(state.filter === f) },
      f === "all" ? null : h("i", { class: `dot s-${f}` }),
      `${f === "all" ? "全部" : LABEL[f]} ${n}`);
    b.onclick = () => {
      state.filter = f;
      renderHeader();
      renderBody();
    };
    filters.append(b);
  }
  const search = h("input", { type: "search", placeholder: "搜尋內文…", "aria-label": "搜尋" });
  search.value = state.query;
  search.oninput = () => {
    state.query = search.value.trim().toLowerCase();
    renderBody();
  };
  $("header").replaceChildren(tabs, filters, search);
}

function matches(s: Sentence): boolean {
  if (state.filter !== "all" && s.status !== state.filter) return false;
  return !state.query || s.text.toLowerCase().includes(state.query);
}

function renderPanel(): void {
  const s = state.selected;
  if (!s) {
    $("panel").replaceChildren(h("h3", {}, "證據面板"), h("p", { class: "hint" }, "點選內文中的句子，查看來源引用、缺少的部分與判定說明。"));
    return;
  }
  const kids: Node[] = [
    h("h3", {}, `第 ${s.index} 句`),
    h("span", { class: `badge s-${s.status}` }, `${GLYPH[s.status]} ${LABEL[s.status]}`),
    h("p", {}, ...inline(s.text, "")),
  ];
  if (s.evidence.length === 0) {
    kids.push(h("p", { class: "hint" }, s.status === "none" ? "在給定參考資料中沒有找到支持（不代表內容錯誤）。" : "此句沒有逐句證據。"));
  }
  for (const e of s.evidence) {
    kids.push(
      h("div", { class: "ev" },
        h("span", { class: `badge s-${e.kind}` }, `${GLYPH[e.kind]} ${LABEL[e.kind]}`),
        h("b", {}, e.refId),
        e.quote ? h("div", { class: "lab" }, "參考資料原文") : null,
        e.quote ? h("blockquote", {}, e.quote) : null,
        e.note || e.missing ? h("div", { class: "lab" }, "判定說明") : null,
        e.note ? h("div", { class: "miss" }, e.note) : null,
        e.missing ? h("div", { class: "miss" }, `缺少：${e.missing}`) : null,
      ),
    );
  }
  $("panel").replaceChildren(...kids);
}

function renderBody(): void {
  const doc = docs[state.doc];
  const article = $("article");
  article.replaceChildren();
  const spans = new Map<Sentence, HTMLElement>();
  doc.sections.forEach((sec, i) => {
    if (sec.title) article.append(h("h2", { id: `sec-${i}` }, sec.title));
    for (const p of sec.paragraphs) {
      const para = h("p");
      for (const s of p.sentences) {
        const cls = `sent st-${s.status}${matches(s) ? "" : " dim"}${state.selected === s ? " sel" : ""}`;
        const span = h("span", { class: cls, tabindex: "0" },
          s.status === "unmarked" ? null : h("sup", { class: "g" }, GLYPH[s.status]),
          ...inline(s.text, state.query));
        const pick = () => {
          state.selected = s;
          spans.forEach((el, k) => el.classList.toggle("sel", k === s));
          renderPanel();
        };
        span.onclick = pick;
        span.onkeydown = (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            pick();
          }
        };
        spans.set(s, span);
        para.append(span, " ");
      }
      article.append(para);
    }
  });
  renderPanel();
}

function render(): void {
  const doc = docs[state.doc];
  renderHeader();
  const [first = "", ...rest] = doc.notice.map((l) => l.replace(/`/g, ""));
  $("notice").replaceChildren(
    h("details", {}, h("summary", {}, ...inline(first, "")), ...rest.map((l) => h("p", {}, ...inline(l, "")))),
  );
  $("stats").replaceChildren(bar(doc.counts));
  const toc = $("toc");
  toc.replaceChildren(h("h3", {}, `目錄（${doc.sections.length} 節）`));
  doc.sections.forEach((sec, i) => {
    const c: Record<Status, number> = { source: 0, partial: 0, conflict: 0, none: 0, unmarked: 0 };
    for (const p of sec.paragraphs) for (const s of p.sentences) c[s.status]++;
    const a = h("a", {}, `${i + 1}　${sec.title || "（前言）"}`, bar(c));
    a.onclick = () => document.getElementById(`sec-${i}`)?.scrollIntoView({ behavior: "smooth" });
    toc.append(a);
  });
  renderBody();
}

render();
