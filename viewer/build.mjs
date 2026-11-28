// 把 ../out/<name>/annotated.md 解析後內嵌，輸出單檔 dist/index.html
import { build } from "esbuild";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseAnnotated } from "./src/parse.ts";

const { names: NAMES } = JSON.parse(
  await readFile(new URL("./docs.config.json", import.meta.url), "utf8"),
);
const docs = [];
for (const name of NAMES) {
  const md = await readFile(new URL(`../out/${name}/annotated.md`, import.meta.url), "utf8");
  docs.push(parseAnnotated(name, md));
}

const { outputFiles } = await build({
  entryPoints: ["src/main.ts"],
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2022",
  write: false,
  define: { __DOCS__: JSON.stringify(docs) },
});
const js = outputFiles[0].text.replaceAll("</script", "<\\/script");
const css = await readFile(new URL("./src/style.css", import.meta.url), "utf8");

const html = `<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>annotated.md 檢視器</title><style>${css}</style></head>
<body>
<header id="header"></header>
<div class="notice" id="notice"></div>
<div class="stats" id="stats"></div>
<main><nav id="toc"></nav><article id="article"></article><aside id="panel"></aside></main>
<script>${js}</script>
</body></html>`;
await mkdir(new URL("./dist/", import.meta.url), { recursive: true });
await writeFile(new URL("./dist/index.html", import.meta.url), html);
console.log(`dist/index.html  ${(html.length / 1024).toFixed(0)} KB`);
for (const d of docs) console.log(d.name, JSON.stringify(d.counts));
