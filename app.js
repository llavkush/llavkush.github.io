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
  function currentTheme() { return localStorage.getItem("theme") || "red-graphite"; }
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
    { href: "about.html",    label: "About",    key: "about" }
  ];
  function activeKey() {
    var f = (location.pathname.split("/").pop() || "index.html").toLowerCase();
    if (f === "" || f === "index.html") return "home";
    if (f === "post.html") return "blog";
    return f.replace(".html", "");
  }

  /* ===== Sections (top-level categories) =====
     Display order for the blog. Anything not listed sorts after these,
     alphabetically; posts with no category fall back to "Notes". */
  var CATEGORY_ORDER = [
    "Machine Learning / AI",
    "Engineering Notes",
    "DSA Notes",
    "Case Studies"
  ];
  function catOf(p) { return (p && p.category) || "Notes"; }
  function catRank(c) { var i = CATEGORY_ORDER.indexOf(c); return i === -1 ? 999 : i; }

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
      '<div class="side-label">Sections</div>' +
      '<div class="side-sections" id="side-sections"></div>' +
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

    // sections + tags from posts
    loadIndex().then(function (list) {
      // sections (categories) in display order
      var catSeen = {};
      list.forEach(function (p) { var c = catOf(p); catSeen[c] = (catSeen[c] || 0) + 1; });
      var cats = Object.keys(catSeen).sort(function (a, b) {
        return (catRank(a) - catRank(b)) || (a < b ? -1 : 1);
      });
      var secBox = document.getElementById("side-sections");
      if (secBox && cats.length) {
        secBox.innerHTML = cats.map(function (c) {
          return '<a href="blog.html?category=' + encodeURIComponent(c) + '">' +
                 escapeHtml(c) + ' <span class="count">' + catSeen[c] + "</span></a>";
        }).join("");
      }

      // tags
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

  /* ===== Note card (Bear-style, w/ snippet) ===== */
  function noteCardLi(p) {
    var tags = (p.tags || []).map(function (t) { return '<span class="tag">' + escapeHtml(t) + "</span>"; }).join(" ");
    var part = p.part ? '<span class="part-badge">Part ' + escapeHtml(p.part) + "</span>" : "";
    return '<li><a class="note-card" href="post.html?p=' + encodeURIComponent(p.slug) + '">' +
             '<div class="nt">' + (part ? part + " " : "") + escapeHtml(p.title) + "</div>" +
             (p.summary ? '<div class="snippet">' + escapeHtml(p.summary) + "</div>" : "") +
             '<div class="meta"><span class="date">' + fmtDate(p.date) + "</span>" +
             (tags ? " " + tags : "") + "</div>" +
           "</a></li>";
  }
  // Paint a list of <li> into el — wrap in a <ul> unless el already is one.
  function paintList(el, items, empty) {
    if (!items.length) {
      var msg = empty || 'No posts yet — <a href="write.html">write the first one</a>.';
      el.innerHTML = el.tagName === "UL" ? '<li class="muted">' + msg + "</li>" : '<p class="muted">' + msg + "</p>";
      return;
    }
    var html = items.join("");
    el.innerHTML = el.tagName === "UL" ? html : '<ul class="note-list">' + html + "</ul>";
  }

  /* ===== Flat note list (Home recent, tag / series / category filters) ===== */
  window.renderNoteList = function (selector, opts) {
    opts = opts || {};
    var el = document.querySelector(selector);
    if (!el) return;
    loadIndex().then(function (list) {
      if (opts.tag) list = list.filter(function (p) { return (p.tags || []).indexOf(opts.tag) !== -1; });
      if (opts.category) list = list.filter(function (p) { return catOf(p) === opts.category; });
      if (opts.series) {
        list = list.filter(function (p) { return p.series === opts.series; });
        list.sort(function (a, b) { return (a.part || 0) - (b.part || 0); });
      }
      if (opts.limit) list = list.slice(0, opts.limit);
      paintList(el, list.map(noteCardLi));
    });
  };

  /* ===== Group posts into sections → series + standalone notes ===== */
  function groupSections(list) {
    var byCat = {};
    list.forEach(function (p) { var c = catOf(p); (byCat[c] = byCat[c] || []).push(p); });
    return Object.keys(byCat).sort(function (a, b) {
      return (catRank(a) - catRank(b)) || (a < b ? -1 : 1);
    }).map(function (c) {
      var seriesMap = {}, order = [], loose = [];
      byCat[c].forEach(function (p) {
        if (p.series) {
          if (!seriesMap[p.series]) { seriesMap[p.series] = []; order.push(p.series); }
          seriesMap[p.series].push(p);
        } else { loose.push(p); }
      });
      var series = order.map(function (name) {
        var parts = seriesMap[name].slice().sort(function (a, b) { return (a.part || 0) - (b.part || 0); });
        return { name: name, parts: parts };
      });
      return { category: c, series: series, loose: loose };
    });
  }

  function sectionHtml(s, withHead) {
    var seriesHtml = s.series.map(function (se) {
      var preview = se.parts.slice(0, 3).map(function (p) { return escapeHtml(p.title); }).join(" · ");
      if (se.parts.length > 3) preview += " …";
      return '<a class="series-card" href="blog.html?series=' + encodeURIComponent(se.name) + '">' +
               '<div class="series-top">' +
                 '<span class="series-name">' + escapeHtml(se.name) + "</span>" +
                 '<span class="series-count">' + se.parts.length + " part" + (se.parts.length === 1 ? "" : "s") + "</span>" +
               "</div>" +
               '<div class="series-parts">' + preview + "</div>" +
             "</a>";
    }).join("");
    var looseHtml = s.loose.length ? '<ul class="note-list">' + s.loose.map(noteCardLi).join("") + "</ul>" : "";
    return '<section class="blog-section">' +
             (withHead ? '<h2 class="section-head">' + escapeHtml(s.category) + "</h2>" : "") +
             (seriesHtml ? '<div class="series-grid">' + seriesHtml + "</div>" : "") +
             looseHtml +
           "</section>";
  }

  /* ===== Sectioned blog index (optionally a single category) ===== */
  window.renderBlogIndex = function (selector, opts) {
    opts = opts || {};
    var el = document.querySelector(selector);
    if (!el) return;
    loadIndex().then(function (list) {
      if (opts.category) list = list.filter(function (p) { return catOf(p) === opts.category; });
      if (!list.length) {
        el.innerHTML = '<p class="muted">No posts yet — <a href="write.html">write the first one</a>.</p>';
        return;
      }
      // single-category view omits the redundant section heading (it's the page H1)
      el.innerHTML = groupSections(list)
        .map(function (s) { return sectionHtml(s, !opts.category); })
        .join("");
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

        // breadcrumb: Category › Series
        var crumbEl = document.getElementById("post-crumb");
        if (crumbEl) {
          var crumbs = ['<a href="blog.html?category=' + encodeURIComponent(catOf(meta)) + '">' + escapeHtml(catOf(meta)) + "</a>"];
          if (meta.series) {
            crumbs.push('<a href="blog.html?series=' + encodeURIComponent(meta.series) + '">' + escapeHtml(meta.series) + "</a>");
          }
          crumbEl.innerHTML = crumbs.join('<span class="sep">›</span>');
        }

        if (metaEl) {
          var tags = (meta.tags || []).map(function (t) {
            return '<a class="tag" href="blog.html?tag=' + encodeURIComponent(t) + '">' + escapeHtml(t) + "</a>";
          }).join(" ");
          var part = meta.part ? '<span class="part-badge">Part ' + escapeHtml(meta.part) + "</span> " : "";
          metaEl.innerHTML = part + '<span class="date">' + fmtDate(meta.date) + "</span>" + (tags ? " " + tags : "");
        }

        // prev / next within the same series (ordered by part)
        var navEl = document.getElementById("post-nav");
        if (navEl && meta.series) {
          var sibs = list.filter(function (p) { return p.series === meta.series; })
                         .sort(function (a, b) { return (a.part || 0) - (b.part || 0); });
          var idx = sibs.map(function (p) { return p.slug; }).indexOf(slug);
          var prev = idx > 0 ? sibs[idx - 1] : null;
          var next = idx > -1 && idx < sibs.length - 1 ? sibs[idx + 1] : null;
          var link = function (p, dir) {
            return '<a class="post-nav-link ' + dir + '" href="post.html?p=' + encodeURIComponent(p.slug) + '">' +
                     '<span class="dir">' + (dir === "prev" ? "← Previous" : "Next →") + "</span>" +
                     '<span class="t">' + escapeHtml(p.title) + "</span></a>";
          };
          navEl.innerHTML = (prev ? link(prev, "prev") : "<span></span>") + (next ? link(next, "next") : "<span></span>");
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
    var root = scope || document;
    if (window.hljs) {
      root.querySelectorAll("pre code").forEach(function (block) {
        try { hljs.highlightElement(block); } catch (e) {}
      });
    }
    addCopyButtons(root);
  };

  /* ===== Copy-to-clipboard buttons on code blocks ===== */
  function addCopyButtons(root) {
    root.querySelectorAll("pre").forEach(function (pre) {
      if (pre.querySelector(".copy-btn")) return;
      var code = pre.querySelector("code");
      if (!code) return;
      pre.classList.add("has-copy");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.setAttribute("aria-label", "Copy code");
      btn.addEventListener("click", function () {
        var text = code.innerText;
        var done = function () { btn.textContent = "Copied"; btn.classList.add("copied"); setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
        } else { fallbackCopy(text); done(); }
      });
      pre.appendChild(btn);
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
  }

  document.addEventListener("DOMContentLoaded", buildSidebar);
})();
