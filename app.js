/* ---- Bear-style portfolio: sidebar chrome + content ---- */
(function () {
  "use strict";

  /* ===== Themes ===== */
  var THEMES = [
    { id: "red-graphite",    name: "Red Graphite" },
    { id: "charcoal",        name: "Charcoal" },
    { id: "solarized-light", name: "Solarized Light" },
    { id: "solarized-dark",  name: "Solarized Dark" },
    { id: "dracula",         name: "Dracula" },
    { id: "gotham",          name: "Gotham" }
  ];
  var root = document.documentElement;
  function prefersDark() { return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches; }
  function currentTheme() { return localStorage.getItem("theme") || (prefersDark() ? "charcoal" : "red-graphite"); }
  root.setAttribute("data-theme", currentTheme());
  window.setTheme = function (id) { root.setAttribute("data-theme", id); localStorage.setItem("theme", id); syncPickers(); };
  function syncPickers() { document.querySelectorAll(".theme-select").forEach(function (s) { s.value = currentTheme(); }); }

  /* ===== Mobile drawer ===== */
  window.toggleNav = function () { document.body.classList.toggle("nav-open"); };
  window.closeNav = function () { document.body.classList.remove("nav-open"); };

  /* ===== Icons (Feather-style, 24×24 stroke) ===== */
  var ICONS = {
    home:     '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    blog:     '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    projects: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    about:    '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    write:    '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>'
  };
  function icon(key) {
    return '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
           'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
           (ICONS[key] || "") + "</svg>";
  }

  /* ===== Nav model ===== */
  var NAV = [
    { href: "index.html",    label: "Home",     key: "home" },
    { href: "blog.html",     label: "Blog",     key: "blog" },
    { href: "projects.html", label: "Projects", key: "projects" },
    { href: "about.html",    label: "About",    key: "about" },
    { href: "write.html",    label: "Write",    key: "write" }
  ];
  function activeKey() {
    var f = (location.pathname.split("/").pop() || "index.html").toLowerCase();
    if (f === "" || f === "index.html") return "home";
    if (f === "post.html") return "blog";
    return f.replace(".html", "");
  }

  /* ===== Post index ===== */
  function loadIndex() {
    return fetch("posts/posts.json", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (l) { l.sort(function (a, b) { return a.date < b.date ? 1 : -1; }); return l; })
      .catch(function () { return []; });
  }
  function fmtDate(iso) {
    var d = new Date(iso + "T00:00:00");
    return isNaN(d) ? iso : d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* ===== Build the sidebar ===== */
  function buildSidebar() {
    var el = document.getElementById("sidebar");
    if (!el) return;
    var act = activeKey();
    var nav = NAV.map(function (n, i) {
      return '<a href="' + n.href + '"' + (n.key === act ? ' class="active" aria-current="page"' : "") +
             ' style="animation-delay:' + (i * 0.03) + 's">' +
             icon(n.key) + "<span>" + n.label + "</span></a>";
    }).join("");

    el.innerHTML =
      '<a class="brand" href="index.html">' +
        '<div class="brand-badge">ल</div>' +
        '<div class="brand-text">' +
          '<div class="name">Lavkush Gupta</div>' +
          '<div class="role">Senior AI Engineer</div>' +
        "</div>" +
      "</a>" +
      '<nav class="side-nav">' + nav + "</nav>" +
      '<div class="side-label">Tags</div>' +
      '<div class="side-tags" id="side-tags"><a href="blog.html">all</a></div>' +
      '<div class="side-foot">' +
        '<div class="socials">' +
          '<a href="https://github.com/llavkush" target="_blank" rel="noopener">GitHub</a>' +
          '<a href="https://www.linkedin.com/in/lavkushsg" target="_blank" rel="noopener">LinkedIn</a>' +
          '<a href="mailto:lavkushsg@gmail.com">Email</a>' +
        "</div>" +
        '<select class="theme-select" onchange="setTheme(this.value)" aria-label="Theme"></select>' +
      "</div>";

    // theme options
    var sel = el.querySelector(".theme-select");
    THEMES.forEach(function (t) { var o = document.createElement("option"); o.value = t.id; o.textContent = t.name; sel.appendChild(o); });
    syncPickers();

    // tags from posts
    loadIndex().then(function (list) {
      var seen = {};
      list.forEach(function (p) { (p.tags || []).forEach(function (t) { seen[t] = (seen[t] || 0) + 1; }); });
      var tags = Object.keys(seen).sort();
      var box = document.getElementById("side-tags");
      if (box && tags.length) {
        box.innerHTML = '<a href="blog.html">all</a>' + tags.map(function (t) {
          return '<a href="blog.html?tag=' + encodeURIComponent(t) + '">' + escapeHtml(t) + "</a>";
        }).join("");
      }
    });

    // close drawer when a link is tapped
    el.addEventListener("click", function (e) { if (e.target.closest("a")) closeNav(); });
  }

  /* ===== Note list (Bear note cards w/ snippet) ===== */
  window.renderNoteList = function (selector, opts) {
    opts = opts || {};
    var el = document.querySelector(selector);
    if (!el) return;
    loadIndex().then(function (list) {
      if (opts.tag) list = list.filter(function (p) { return (p.tags || []).indexOf(opts.tag) !== -1; });
      if (opts.limit) list = list.slice(0, opts.limit);
      if (!list.length) {
        el.innerHTML = '<li class="muted">No posts yet — <a href="write.html">write the first one</a>.</li>';
        return;
      }
      el.innerHTML = list.map(function (p) {
        var tags = (p.tags || []).map(function (t) { return '<span class="tag">' + escapeHtml(t) + "</span>"; }).join(" ");
        return '<li><a class="note-card" href="post.html?p=' + encodeURIComponent(p.slug) + '">' +
                 '<div class="nt">' + escapeHtml(p.title) + "</div>" +
                 (p.summary ? '<div class="snippet">' + escapeHtml(p.summary) + "</div>" : "") +
                 '<div class="meta"><span class="date">' + fmtDate(p.date) + "</span>" +
                 (tags ? " " + tags : "") + "</div>" +
               "</a></li>";
      }).join("");
    });
  };

  /* ===== Single post ===== */
  window.renderPost = function () {
    var slug = new URLSearchParams(location.search).get("p");
    var titleEl = document.getElementById("post-title");
    var metaEl = document.getElementById("post-meta");
    var bodyEl = document.getElementById("post-body");
    if (!slug || !bodyEl) { if (bodyEl) bodyEl.innerHTML = "<p>Post not found.</p>"; return; }
    loadIndex().then(function (list) {
      var meta = list.filter(function (p) { return p.slug === slug; })[0];
      if (meta) {
        document.title = meta.title + " · Lavkush";
        if (titleEl) titleEl.textContent = meta.title;
        if (metaEl) {
          var tags = (meta.tags || []).map(function (t) {
            return '<a class="tag" href="blog.html?tag=' + encodeURIComponent(t) + '">' + escapeHtml(t) + "</a>";
          }).join(" ");
          metaEl.innerHTML = '<span class="date">' + fmtDate(meta.date) + "</span>" + (tags ? " " + tags : "");
        }
      }
      return fetch("posts/" + slug + ".md", { cache: "no-store" });
    }).then(function (r) {
      if (!r || !r.ok) throw new Error("missing");
      return r.text();
    }).then(function (md) {
      bodyEl.innerHTML = window.marked ? marked.parse(md) : "<pre>" + escapeHtml(md) + "</pre>";
      highlightCode(bodyEl);
    }).catch(function () { bodyEl.innerHTML = "<p>Could not load this post.</p>"; });
  };

  /* ===== Code syntax highlighting (highlight.js, if present) ===== */
  window.highlightCode = function (scope) {
    if (!window.hljs) return;
    (scope || document).querySelectorAll("pre code").forEach(function (block) {
      try { hljs.highlightElement(block); } catch (e) {}
    });
  };

  document.addEventListener("DOMContentLoaded", buildSidebar);
})();
