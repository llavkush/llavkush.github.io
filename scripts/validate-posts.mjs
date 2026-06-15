#!/usr/bin/env node
// Validates posts/posts.json against the actual markdown files.
// Fails (exit 1) if the index and the posts/ directory have drifted apart,
// or if any entry is missing required fields. Run in CI on every push.

import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const POSTS_DIR = join(ROOT, "posts");

const errors = [];

let list;
try {
  list = JSON.parse(await readFile(join(POSTS_DIR, "posts.json"), "utf8"));
} catch (e) {
  console.error("✗ posts/posts.json is missing or not valid JSON:", e.message);
  process.exit(1);
}

if (!Array.isArray(list)) {
  console.error("✗ posts/posts.json must be a JSON array.");
  process.exit(1);
}

// Every index entry must be well-formed and have a matching .md file.
const slugs = new Set();
const mdFiles = (await readdir(POSTS_DIR)).filter((f) => f.endsWith(".md"));
const mdSet = new Set(mdFiles.map((f) => f.replace(/\.md$/, "")));
// Files that are docs, not posts — never expected in the index.
const NON_POST = new Set(["STYLE-GUIDE"]);

for (const [i, p] of list.entries()) {
  const where = `entry #${i}` + (p && p.slug ? ` (${p.slug})` : "");
  if (!p || typeof p !== "object") { errors.push(`${where}: not an object`); continue; }
  for (const field of ["slug", "title", "date"]) {
    if (!p[field]) errors.push(`${where}: missing "${field}"`);
  }
  if (p.date && !/^\d{4}-\d{2}-\d{2}$/.test(p.date)) {
    errors.push(`${where}: date "${p.date}" is not YYYY-MM-DD`);
  }
  if (p.tags && !Array.isArray(p.tags)) {
    errors.push(`${where}: "tags" must be an array`);
  }
  if (!p.category) {
    errors.push(`${where}: missing "category" (e.g. "Engineering Notes")`);
  }
  // "part" only makes sense inside a "series"; when present it must be a number.
  if (p.part != null && typeof p.part !== "number") {
    errors.push(`${where}: "part" must be a number`);
  }
  if (p.part != null && !p.series) {
    errors.push(`${where}: has "part" but no "series"`);
  }
  if (p.slug) {
    if (slugs.has(p.slug)) errors.push(`${where}: duplicate slug`);
    slugs.add(p.slug);
    if (!mdSet.has(p.slug)) errors.push(`${where}: no posts/${p.slug}.md file`);
  }
}

// Every .md file (except known docs) should be indexed.
for (const name of mdSet) {
  if (!NON_POST.has(name) && !slugs.has(name)) {
    errors.push(`posts/${name}.md exists but is not listed in posts.json`);
  }
}

if (errors.length) {
  console.error("✗ posts.json validation failed:\n  - " + errors.join("\n  - "));
  process.exit(1);
}

console.log(`✓ posts.json is consistent (${list.length} post${list.length === 1 ? "" : "s"}).`);
