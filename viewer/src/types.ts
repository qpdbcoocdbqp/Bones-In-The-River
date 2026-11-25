/** 句子／證據的標註狀態。conflict = 與參考資料矛盾（⚠）。 */
export type Status = "source" | "partial" | "conflict" | "none" | "unmarked";

export interface Evidence {
  kind: "source" | "partial" | "conflict";
  refId: string;
  quote: string;
  note: string;
  missing: string;
}

export interface Sentence {
  index: number; // 段內 1 起算，對應「第 N 句」
  text: string;
  status: Status;
  evidence: Evidence[];
}

export interface Paragraph {
  status: Status; // 段尾標籤彙整：只有無來源→none，含部分相關→partial，否則 source；無標籤→unmarked
  refs: string[];
  sentences: Sentence[];
}

export interface Section {
  title: string;
  paragraphs: Paragraph[];
}

export interface Doc {
  name: string;
  notice: string[];
  sections: Section[];
  counts: Record<Status, number>;
}
