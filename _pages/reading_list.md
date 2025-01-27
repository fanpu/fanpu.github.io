---
layout: page
permalink: /reading-list/
title: ML Reading List
description: >
  Curated list of papers I have bookmarked to read/have read,
  accompanied by a description on why I think it is worth reading. You can click
  on the tags to filter for only papers in that category.
nav: true
nav_order: 10
giscus_comments: true
---

<div class="mb-3">
  <span id="filter-badge" class="badge" style="display:none;"></span>
</div>

<ul class="card-text font-weight-light list-group list-group-flush" id="papers-list">
  {% assign papers = site.data.papers | reverse %}
  {% for paper in papers %}
    <li class="list-group-item" data-tags="{{ paper.tags | join: ' ' }}">
      <div class="row">
        <div class="col">
          <h6 class="title font-weight-bold ml-1 ml-md-4">
            {{ forloop.index }}. <a href="{{ paper.url }}">{{ paper.title }}</a>
          </h6>
          <h6 class="ml-1 ml-md-4" style="font-size: 0.95rem; font-style: italic">
            {{ paper.authors }}
          </h6>
          <h6 class="ml-1 ml-md-4" style="font-size: 0.95rem; font-style: italic">
            {{ paper.notes }}
          </h6>
          {% if paper.tags %}
            <div class="ml-1 ml-md-4">
              {% for tag in paper.tags %}
                <a href="#{{ tag | downcase | replace: ' ', '-' }}"
                   class="badge"
                   style="background-color:{{ site.data.tag_colors[tag] }}">
                  {{ tag }}
                </a>
              {% endfor %}
            </div>
          {% endif %}
        </div>
        <div class="col-auto text-right">
          {% if paper.published %}
            {% assign date = paper.published | split: '-' | join: '.' %}
            <span class="text-muted" style="font-size: 0.85rem;">
              {{ date }}
            </span>
          {% endif %}
        </div>
      </div>
    </li>
  {% endfor %}
</ul>

<script>
(function() {
  // 1. Build a color map from normalized tag -> color
  //    Also build a text lookup from normalized tag -> original text
  var tagColors = {};
  var tagLookup = {};

  {% for tagName in site.data.tag_colors %}
    (function() {
      var originalText = "{{ tagName[0] }}"; // e.g. "Deep Learning"
      var normalizedTag = originalText.toLowerCase().replace(/\s+/g, '-'); // e.g. "deep-learning"
      var colorValue = "{{ tagName[1] }}";

      // fill in the dictionaries
      tagColors[normalizedTag] = colorValue;
      tagLookup[normalizedTag] = originalText;
    })();
  {% endfor %}

  // Helper: fallback color
  function getTagColor(normalizedTag) {
    return tagColors[normalizedTag] || '#ccc';
  }

  // 2. DOM references
  var filterBadge = document.getElementById('filter-badge');
  var papersListItems = document.querySelectorAll('#papers-list li');

  // 3. Show/hide the filter badge with a clickable ×
  function updateFilterBadge(normalizedTag) {
    if (normalizedTag) {
      // Look up the original text for display
      var originalText = tagLookup[normalizedTag] || normalizedTag;

      filterBadge.style.display = '';
      filterBadge.style.backgroundColor = getTagColor(normalizedTag);
      // If you want consistent text color (like white) for all badges, do: filterBadge.style.color = '#fff';
      // Then ensure the cross inherits that.
      filterBadge.innerHTML = originalText
        + ' <span id="cancel-filter" style="cursor:pointer; margin-left:6px; color:inherit;">&times;</span>';
    } else {
      filterBadge.style.display = 'none';
      filterBadge.innerHTML = '';
    }
  }

  // 4. Filter items by the normalized tag
  function filterByTag(normalizedTag) {
    updateFilterBadge(normalizedTag);

    // Show items whose 'data-tags' includes the original text.
    // But note your <li> has data-tags like "Deep Learning" (untransformed).
    // We'll check if the lowercased/ hyphenated version is present.
    papersListItems.forEach(function(item) {
      var rawTags = item.getAttribute('data-tags') || "";
      var rawTagsLower = rawTags.toLowerCase();
      var rawTagsHyphenated = rawTagsLower.replace(/\s+/g, '-'); 
      // Quick approach: just see if normalizedTag is in rawTagsHyphenated
      // This means "Deep Learning" -> "deep-learning" is matched.
      var show = !normalizedTag || rawTagsHyphenated.includes(normalizedTag);
      item.style.display = show ? '' : 'none';
    });
  }

  // 5. Handle changes to the hash
  window.addEventListener('hashchange', function() {
    var hashVal = window.location.hash.replace('#','').toLowerCase();
    filterByTag(hashVal);
  });

  // 6. Filter on page load if a hash is present
  var initialTag = window.location.hash.replace('#','').toLowerCase();
  filterByTag(initialTag);

  // 7. Clicking the cross resets the filter
  filterBadge.addEventListener('click', function(e) {
    if (e.target.id === 'cancel-filter') {
      window.location.hash = ''; // triggers hashchange
    }
  });
})();
</script>