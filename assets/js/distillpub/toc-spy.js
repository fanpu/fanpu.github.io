// Highlights the section being read in the distill table of contents.
(function () {
  // A heading counts as reached once its top is this far from the top of the
  // window: the 81px fixed navbar plus 39px of breathing room.
  var READ_LINE = 120;
  // Height of the fixed footer, which covers the bottom of the window.
  var FOOTER_HEIGHT = 53;
  // Space kept around the active link when the contents scroll on their own.
  var TOC_MARGIN = 8;

  var toc = document.querySelector("d-contents");
  if (!toc) return;

  var entries = Array.prototype.slice
    .call(toc.querySelectorAll('nav a[href^="#"]'))
    .map(function (link) {
      return { link: link, heading: document.getElementById(decodeURIComponent(link.hash.slice(1))) };
    })
    .filter(function (entry) {
      return entry.heading;
    });
  if (!entries.length) return;

  var marker = document.createElement("div");
  marker.className = "toc-marker";
  marker.setAttribute("aria-hidden", "true");
  toc.appendChild(marker);
  toc.classList.add("toc-spy");

  var active = null;
  var queued = false;
  // A clicked section stays active until the reader scrolls away from it.
  var pinned = null;
  var pinnedY = null;

  // The last headings may never reach READ_LINE because the page ends first.
  // Over the final screen of scrolling, move the line down to the bottom of the
  // window so that each of them still becomes active in turn.
  function readLine() {
    var remaining = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
    var span = window.innerHeight - FOOTER_HEIGHT - READ_LINE;
    return READ_LINE + span * Math.max(0, 1 - remaining / span);
  }

  function keepInView(link) {
    if (toc.scrollHeight <= toc.clientHeight) return;
    var linkRect = link.getBoundingClientRect();
    var tocRect = toc.getBoundingClientRect();
    if (linkRect.top < tocRect.top + TOC_MARGIN) {
      toc.scrollTop -= tocRect.top + TOC_MARGIN - linkRect.top;
    } else if (linkRect.bottom > tocRect.bottom - TOC_MARGIN) {
      toc.scrollTop += linkRect.bottom - tocRect.bottom + TOC_MARGIN;
    }
  }

  function placeMarker() {
    if (!active) {
      marker.style.opacity = "0";
      return;
    }
    var linkRect = active.link.getBoundingClientRect();
    var tocRect = toc.getBoundingClientRect();
    marker.style.height = linkRect.height + "px";
    marker.style.transform = "translateY(" + (linkRect.top - tocRect.top + toc.scrollTop) + "px)";
    marker.style.opacity = "1";
  }

  function update() {
    queued = false;
    if (pinned) {
      if (pinnedY === null) pinnedY = window.scrollY;
      if (Math.abs(window.scrollY - pinnedY) > 2) pinned = null;
    }
    var current = pinned;
    if (!current) {
      var line = readLine();
      entries.forEach(function (entry) {
        if (entry.heading.getBoundingClientRect().top <= line) current = entry;
      });
    }
    if (current !== active) {
      if (active) active.link.removeAttribute("aria-current");
      if (current) current.link.setAttribute("aria-current", "true");
      active = current;
      if (active) keepInView(active.link);
    }
    placeMarker();
  }

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(update);
  }

  entries.forEach(function (entry) {
    entry.link.addEventListener("click", function (event) {
      if (event.metaKey || event.ctrlKey || event.shiftKey) return;
      pinned = entry;
      pinnedY = null;
      schedule();
    });
  });

  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  window.addEventListener("load", schedule);
  // Images and math change the height of the article after the page loads.
  if (window.ResizeObserver) {
    new ResizeObserver(schedule).observe(document.querySelector("d-article"));
  }
  update();
})();
