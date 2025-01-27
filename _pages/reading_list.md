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
          <h6 class="ml-1 ml-md-4" style="font-size: 0.95rem; font-style: italic; color: #888">
            {% assign authors_list = paper.authors | split: ',' %}
            {% if authors_list.size > 7 %}
              {{ authors_list | slice: 0, 7 | join: ', ' }} et al.
            {% else %}
              {{ authors_list | join: ', ' }}
            {% endif %}
          </h6>
          <h6 class="ml-1 ml-md-4" style="font-size: 0.95rem">
            {{ paper.notes }}
          </h6>

          {% if paper.tags %}
            <div class="ml-1 ml-md-4">
              {% assign sorted_tags = paper.tags | sort %}
              {% for tag in sorted_tags %}
                {% assign bgColor = site.data.tag_colors[tag] | default: "#ccc" %}
                <a href="#{{ tag | downcase | replace: ' ', '-' }}"
                   class="badge"
                   data-color="{{ bgColor }}"
                   style="background-color:{{ bgColor }}">
                  {{ tag }}
                </a>
              {% endfor %}
            </div>
          {% endif %}
        </div>
        <div class="col-auto text-right">
          {% if paper.published %}
            {% assign date = paper.published %}
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
  // Helper: returns '#000' or '#fff' depending on color brightness
  function getContrastColor(hexColor) {
    // Defensive check
    if (!hexColor || !/^#([A-Fa-f0-9]{3}|[A-Fa-f0-9]{6})$/.test(hexColor)) {
      return '#000';
    }
    // Normalize short form #abc => #aabbcc
    if (hexColor.length === 4) {
      hexColor = '#' + hexColor[1] + hexColor[1] 
                     + hexColor[2] + hexColor[2] 
                     + hexColor[3] + hexColor[3];
    }
    // Extract r, g, b
    var r = parseInt(hexColor.substr(1, 2), 16);
    var g = parseInt(hexColor.substr(3, 2), 16);
    var b = parseInt(hexColor.substr(5, 2), 16);

    // Approximate luminance
    // (You can tweak the 128 threshold if needed)
    var luminance = 0.299*r + 0.587*g + 0.114*b;
    return (luminance >= 128) ? '#000' : '#fff';
  }

  // 1. Build a color map from normalized tag -> color
  var tagColors = {};
  var tagLookup = {};

  {% for tagName in site.data.tag_colors %}
    (function() {
      var originalText = "{{ tagName[0] }}"; // e.g. "Deep Learning"
      var normalizedTag = originalText.toLowerCase().replace(/\s+/g, '-'); // e.g. "deep-learning"
      var colorValue = "{{ tagName[1] }}";
      tagColors[normalizedTag] = colorValue;
      tagLookup[normalizedTag] = originalText;
    })();
  {% endfor %}

  // Fallback color
  function getTagColor(normalizedTag) {
    return tagColors[normalizedTag] || '#ccc';
  }

  // 2. DOM references
  var filterBadge = document.getElementById('filter-badge');
  var papersListItems = document.querySelectorAll('#papers-list li');
  var allBadges = document.querySelectorAll('a.badge');

  // 3. Show/hide the filter badge with a clickable ×
  function updateFilterBadge(normalizedTag) {
    if (normalizedTag) {
      var originalText = tagLookup[normalizedTag] || normalizedTag;
      var colorValue = getTagColor(normalizedTag);
      filterBadge.style.display = '';
      filterBadge.style.backgroundColor = colorValue;
      var contrast = getContrastColor(colorValue);
      filterBadge.style.setProperty('color', contrast, 'important');
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

    papersListItems.forEach(function(item) {
      var rawTags = item.getAttribute('data-tags') || "";
      var rawTagsLower = rawTags.toLowerCase();
      var rawTagsHyphenated = rawTagsLower.replace(/\s+/g, '-'); 
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

  // 8. Adjust text color for each inline badge
  allBadges.forEach(function(badge) {
  var rawColor = badge.getAttribute('data-color') || '#ccc';
  var contrast = getContrastColor(rawColor);
  // Force override with !important
  badge.style.setProperty('color', contrast, 'important');
});

})();
</script>
