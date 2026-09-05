// add bootstrap classes to tables
$(document).ready(function () {
  $("table").each(function () {
    // Only follow the theme that was actually applied to the page. Reading the OS
    // preference directly would add `table-dark` (white text) even when site dark
    // mode is off and the page is still light -- white text on a white background.
    if (document.documentElement.getAttribute("data-theme") == "dark") {
      $(this).addClass("table-dark");
    } else {
      $(this).removeClass("table-dark");
    }

    // only select tables that are not inside an element with "news" (about page) or "card" (cv page) class
    if (
      $(this).parents('[class*="news"]').length == 0 &&
      $(this).parents('[class*="card"]').length == 0 &&
      $(this).parents('[class*="archive"]').length == 0 &&
      $(this).parents("code").length == 0
    ) {
      // make table use bootstrap-table
      $(this).attr("data-toggle", "table");
      // add some classes to make the table look better
      // $(this).addClass('table-sm');
      $(this).addClass("table-hover");
    }
  });
});
