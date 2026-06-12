#!/usr/bin/env node
// Generates feed.xml (RSS 2.0) from posts/posts.json.
// Source of truth is the index; run after validate-posts.mjs in CI.

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const SITE = "https://llavkush.github.io";
const TITLE = "Lavkush Gupta — Blog";
const DESC = "Case studies, field notes and the occasional vlog by Lavkush Gupta.";
const AUTHOR = "lavkushsg@gmail.com (Lavkush Gupta)";

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
}

function rfc822(date) {
  // date is "YYYY-MM-DD"; anchor at noon UTC to avoid TZ rollover.
  const d = new Date(date + "T12:00:00Z");
  return isNaN(d) ? "" : d.toUTCString();
}

const list = JSON.parse(await readFile(join(ROOT, "posts", "posts.json"), "utf8"));
list.sort((a, b) => (a.date < b.date ? 1 : -1));

const items = list.map((p) => {
  const url = `${SITE}/post.html?p=${encodeURIComponent(p.slug)}`;
  const cats = (p.tags || []).map((t) => `      <category>${esc(t)}</category>`).join("\n");
  return [
    "    <item>",
    `      <title>${esc(p.title)}</title>`,
    `      <link>${esc(url)}</link>`,
    `      <guid isPermaLink="false">${esc(p.slug)}</guid>`,
    p.date ? `      <pubDate>${rfc822(p.date)}</pubDate>` : "",
    p.summary ? `      <description>${esc(p.summary)}</description>` : "",
    cats,
    "    </item>",
  ].filter(Boolean).join("\n");
}).join("\n");

const lastBuild = list[0] && list[0].date ? rfc822(list[0].date) : new Date(0).toUTCString();

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${esc(TITLE)}</title>
    <link>${SITE}/blog.html</link>
    <atom:link href="${SITE}/feed.xml" rel="self" type="application/rss+xml"/>
    <description>${esc(DESC)}</description>
    <language>en</language>
    <managingEditor>${esc(AUTHOR)}</managingEditor>
    <lastBuildDate>${lastBuild}</lastBuildDate>
${items}
  </channel>
</rss>
`;

await writeFile(join(ROOT, "feed.xml"), xml);
console.log(`✓ feed.xml written (${list.length} item${list.length === 1 ? "" : "s"}).`);
