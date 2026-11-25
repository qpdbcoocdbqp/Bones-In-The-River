import assert from "node:assert/strict";
import { test } from "node:test";
import { parseAnnotated } from "./parse.ts";

const zh = `> ⚠ 未校準
> 參考資料：a、b

## 標題

第一句。第二句。  【來源：a】【部分相關：b】
> - 第 1 句 ← **a**：「quote one」說明
> - 第 2 句 ◐ 部分相關 **b**：「quote two」缺少：某細節

只有一句。  【無來源】
> - 第 1 句 ⚠ 與 **a** 矛盾：「q」不一致
`;

const en = `> ⚠ not calibrated

## T

甲。乙。  [Partial: R4]
> - Sentence 2 ◐ Partially related **R4**: "quote" Missing: detail

丙。  [No source]
`;

test("zh: sections, tags, evidence", () => {
  const d = parseAnnotated("x", zh);
  assert.equal(d.notice.length, 2);
  assert.equal(d.sections.length, 1);
  const [p1, p2] = d.sections[0].paragraphs;
  assert.equal(p1.status, "partial");
  assert.deepEqual(p1.refs, ["a", "b"]);
  assert.equal(p1.sentences.length, 2);
  assert.equal(p1.sentences[0].status, "source");
  assert.equal(p1.sentences[0].evidence[0].quote, "quote one");
  assert.equal(p1.sentences[1].evidence[0].missing, "某細節");
  assert.equal(p2.status, "none");
  assert.equal(p2.sentences[0].status, "conflict");
  assert.equal(d.counts.conflict, 1);
});

test("quote containing the same quote character", () => {
  const md = `## T

甲。  [Partial: R1]
> - Sentence 1 ◐ Partially related **R1**: "And instead, you're like, Okay? How do I move past it?" And that is the loop" Missing: detail

乙。  [Partial: R2]
> - Sentence 1 ◐ Partially related **R2**: "first part … and 'inner' more…" Missing: x
`;
  const [p1, p2] = parseAnnotated("z", md).sections[0].paragraphs;
  const e1 = p1.sentences[0].evidence[0];
  assert.equal(e1.quote, `And instead, you're like, Okay? How do I move past it?" And that is the loop`);
  assert.equal(e1.missing, "detail");
  assert.equal(p2.sentences[0].evidence[0].quote, "first part … and 'inner' more…");
});

test("en: tags and evidence", () => {
  const d = parseAnnotated("y", en);
  const [p1, p2] = d.sections[0].paragraphs;
  assert.equal(p1.sentences[0].status, "unmarked");
  assert.equal(p1.sentences[1].status, "partial");
  assert.equal(p1.sentences[1].evidence[0].missing, "detail");
  assert.equal(p2.sentences[0].status, "none");
});
