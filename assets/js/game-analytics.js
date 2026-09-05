
// Analytics for the standalone pages under /games/. Those are front-matter-less
// static files, so they never render _includes/scripts.liquid — they load this
// instead. The measurement ID stays single-sourced in _config.yml.
(function () {
  var id = "G-S3VHEYH05S";
  var s = document.createElement("script");
  s.async = true;
  s.src = "https://www.googletagmanager.com/gtag/js?id=" + id;
  document.head.appendChild(s);

  window.dataLayer = window.dataLayer || [];
  function gtag() {
    window.dataLayer.push(arguments);
  }
  window.gtag = gtag;
  gtag("js", new Date());
  gtag("config", id);
})();

