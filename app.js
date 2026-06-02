/* ---- Bear-style portfolio: shared client logic ---- */
(function () {
  "use strict";

  /* ----- Themes (Bear-style named palettes) ----- */
  var THEMES = [
    { id: "red-graphite",    name: "Red Graphite" },
    { id: "charcoal",        name: "Charcoal" },
    { id: "solarized-light", name: "Solarized Light" },
    { id: "solarized-dark",  name: "Solarized Dark" },
    { id: "dracula",         name: "Dracula" },
    { id: "gotham",          name: "Gotham" }
  ];
  var root = document.documentElement;

  function prefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function currentTheme() {
    return localStorage.getItem("theme") || (prefersDark() ? "charcoal" : "red-graphite");
  }

  // apply ASAP to avoid a flash
  root.setAttribute("data-theme", currentTheme());

  window.setTheme = function (id) {
    root.setAttribute("data-theme", id);
    localStorage.setItem("theme", id);
    syncPickers();
  };

  function buildPickers() {
    var cur = currentTheme();
    document.querySelectorAll(".theme-select").forEach(function (sel) {
      if (!sel.options.length) {
        THEMES.forEach(function (t) {
          var o = document.createElement("option");
          o.value = t.id; o.textContent = t.name;
          sel.appendChild(o);
        });
      }
      sel.value = cur;
    });
  }
  function syncPickers() {
    var cur = currentTheme();
    document.querySelectorAll(".theme-select").forEach(function (sel) { sel.value = cur; });
  }

  /* ----- Date helper ----- */
  function fmtDate(iso) {
    var d = new Date(iso + "T00:00:00");
    if (isNaN(d)) return iso;
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }

  /* ----- Load the post index ----- */
  function loadIndex() {
    return fetch("posts/posts.json", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (list) {
        list.sort(function (a, b) { return (a.date < b.date) ? 1 : -1; });
        return list;
      })
      .catch(function () { return []; });
  }

  /* ----- Render a post list into an element ----- */
  window.renderPostList = function (selector, limit) {
    var el = document.querySelector(selector);
    if (!el) return;
    loadIndex().then(function (list) {
      if (limit) list = list.slice(0, limit);
      if (!list.length) {
        el.innerHTML = '<li class="muted">No posts yet — <a href="write.html">write the first one</a>.</li>';
        return;
      }
      el.innerHTML = list.map(function (p) {
        return '<li><span class="date">' + fmtDate(p.date) + '</span>' +
               '<a href="post.html?p=' + encodeURIComponent(p.slug) + '">' +
               escapeHtml(p.title) + '</a></li>';
      }).join("");
    });
  };

  /* ----- Render a single post ----- */
  window.renderPost = function () {
    var slug = new URLSearchParams(location.search).get("p");
    var titleEl = document.getElementById("post-title");
    var metaEl = document.getElementById("post-meta");
    var bodyEl = document.getElementById("post-body");
    if (!slug || !bodyEl) {
      if (bodyEl) bodyEl.innerHTML = "<p>Post not found.</p>";
      return;
    }
    loadIndex().then(function (list) {
      var meta = list.filter(function (p) { return p.slug === slug; })[0];
      if (meta) {
        document.title = meta.title + " · Lavkush";
        if (titleEl) titleEl.textContent = meta.title;
        if (metaEl) {
          var tags = (meta.tags || []).map(function (t) {
            return '<span class="tag">' + escapeHtml(t) + "</span>";
          }).join(" ");
          metaEl.innerHTML = fmtDate(meta.date) + (tags ? " &nbsp; " + tags : "");
        }
      }
      return fetch("posts/" + slug + ".md", { cache: "no-store" });
    }).then(function (r) {
      if (!r) throw new Error("missing meta");
      if (!r.ok) throw new Error("missing md");
      return r.text();
    }).then(function (md) {
      bodyEl.innerHTML = window.marked ? marked.parse(md) : ("<pre>" + escapeHtml(md) + "</pre>");
    }).catch(function () {
      bodyEl.innerHTML = "<p>Could not load this post.</p>";
    });
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  document.addEventListener("DOMContentLoaded", buildPickers);
})();
