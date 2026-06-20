// Google Analytics 4 (gtag.js) — shared loader for all pages.
// Measurement ID lives here only, so it can be rotated in one place.
(function () {
  var GA_ID = "G-V8ZEGXDB2Q";

  var s = document.createElement("script");
  s.async = true;
  s.src = "https://www.googletagmanager.com/gtag/js?id=" + GA_ID;
  document.head.appendChild(s);

  window.dataLayer = window.dataLayer || [];
  function gtag() { dataLayer.push(arguments); }
  window.gtag = gtag;
  gtag("js", new Date());
  gtag("config", GA_ID);
})();
