---
name: write-post
description: Generate a new blog post for this site (llavkush.github.io) in Lavkush's voice and house style, then wire it into posts.json and feed.xml. Use when the user asks to "write a post", "draft a note", "turn this into a blog post", "add a post about X", or points at source material (a link, paper, or repo) to be explained as a post. Handles the full pipeline: research source, write the .md, register it in the index and RSS feed, and verify any code runs.
---

# Write a blog post (Lavkush's style)

This site is a static blog. A post is **two things**: a Markdown body at `posts/<slug>.md` and an index entry in `posts/posts.json`. The RSS feed at `feed.xml` must be kept in sync. Follow this end to end.

## 0. Before writing — gather and confirm

1. **If the user gave a source** (a link, GitHub repo, paper), READ it first with WebFetch. For GitHub folders, hit the API contents endpoint (`https://api.github.com/repos/<owner>/<repo>/contents/<path>`) to list files, then fetch the raw `docs/*.md` and `code/*.py` via `raw.githubusercontent.com`. Understand it thoroughly before writing a word.
2. **Read an existing post** as a style anchor before drafting — `posts/pm2.md` (practical/tool style) or `posts/linear-algebra-intuition.md` (concept/intuition style). Match whichever fits the topic.
3. **Confirm with the user** (only what you can't infer): the angle, the slug, and whether it belongs to the "AI Engineering from Scratch" series (and its part number). Don't over-ask — pick sensible defaults and state them.

## 1. The voice and structure (this is the important part)

Lavkush's posts follow a **theory-then-practical, interleaved** rhythm. Each idea is introduced in plain words, then *immediately* followed by the concrete thing that proves it — a command, or short runnable code. Never dump all the code at the end.

- **Open** with an italic one-line summary, a `**N minute read**`, then a `---` divider. Do NOT put an `# H1` title in the body (the title comes from posts.json and renders as the page H1).
- **Use a light STAR spine** where it fits: `## Situation`, `## Task`, `## Action`, `## Result`, optionally `## What I learned`. The concept posts use `## S — ...` style headers; prefer plain `## Situation` etc. (and remember: **no em dashes**, see §3).
- **Explain to anyone.** Lead every technical idea with an everyday analogy (a kitchen, a group chat, a grocery list) *before* the jargon. Assume a smart non-technical reader. The litmus test: could a friend who doesn't code follow it?
- **Pair theory with practical.** For each concept, show the real artifact next to it:
  - tool/devops posts → the actual shell commands.
  - ML/AI concept posts → **short, dependency-free Python** the reader can paste into a REPL. Prefer raw Python (`math`, lists) over NumPy/PyTorch so it runs anywhere.
- **Make it land (applied, not just intuitive).** Include at least one example where the code *does a real task* and you show the output (e.g. self-attention resolving a pronoun), not only a toy illustration. Then say what the output means.
- **Close** with: a one-page summary table when useful, a `## Result` recap, an out-of-the-box one-line takeaway as a `>` quote, and a final bold invitation question to the reader.
- Optionally end with a hashtag line like `#transformers #attention #intuition` (plain text, renders as-is).
- Use Mermaid blocks and small ASCII diagrams where they clarify; fence all code with its language.

## 2. Verify any code

If the post contains code, RUN IT before publishing (write it to a temp file and execute with `python`/`node`). Paste real output into the post. Never ship code you haven't run.

## 3. House rules (hard constraints)

- **No em dashes (—) anywhere.** Not in the body, not in titles, not in summaries. Use a comma, colon, or parentheses instead. This applies to the .md, posts.json, and feed.xml. (En dashes "–" in date/number ranges are fine.)
- Headings: only `##` and `###` in the body, never `#`.
- Dividers: `---` on its own line between major sections.
- Tags: lowercase, hyphenated, 1–4 of them, in posts.json.

## 4. Register it (wiring)

### `posts/posts.json` — add a NEW entry at the TOP (newest first):
```json
{
  "slug": "<kebab-case-slug>",
  "title": "Title in Title Case (no em dash)",
  "date": "YYYY-MM-DD",
  "category": "Machine Learning / AI",
  "series": "AI Engineering from Scratch",
  "part": 2,
  "tags": ["tag-a", "tag-b", "tag-c"],
  "summary": "One or two sentences, no em dash."
}
```
- `slug` MUST equal the filename: `posts/<slug>.md`. The loader (`app.js`) fetches `posts/<slug>.md` and links `post.html?p=<slug>`.
- `series` + `part` drive the prev/next series navigation. Omit both for a standalone post. `category` and `series`/`part` are optional but used by the site.
- Use today's real date. Validate the JSON parses after editing.

### `feed.xml` — add a matching `<item>` at the TOP of the items, and bump `<lastBuildDate>`:
```xml
<item>
  <title>SAME title as posts.json</title>
  <link>https://llavkush.github.io/post.html?p=<slug></link>
  <guid isPermaLink="false"><slug></guid>
  <pubDate>Day, DD Mon YYYY 12:00:00 GMT</pubDate>
  <description>SAME summary as posts.json (escape &amp; as &amp;amp;)</description>
  <category>tag-a</category>
  <category>tag-b</category>
</item>
```
- The `<title>` and `<description>` MUST match posts.json exactly. Convert the ISO date to RFC-822 (e.g. 2026-06-20 → `Sat, 20 Jun 2026 12:00:00 GMT`).

## 5. Final checklist
- [ ] `posts/<slug>.md` written, slug matches filename, no `# H1` in body.
- [ ] Theory paired with runnable code; at least one applied example with real output shown.
- [ ] All code was actually run and its output is accurate.
- [ ] Zero em dashes in .md, posts.json, feed.xml.
- [ ] posts.json entry added at top; JSON validates.
- [ ] feed.xml item added at top; title/description match posts.json; lastBuildDate bumped.
- [ ] Tell the user the local preview URL: `http://127.0.0.1:5500/post.html?p=<slug>` (Live Server).
- [ ] Note that the new .md is untracked; ask before committing.

## Reference
- Full formatting rules: `posts/STYLE-GUIDE.md`.
- Loader logic: `app.js` (fetches `posts/<slug>.md`, builds series nav).
- The `write.html` page is the manual UI for the same thing; this skill is the automated path.
