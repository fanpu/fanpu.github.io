---
layout: page
permalink: /artifacts/
title: Artifacts
description: Interactive things I've made with LLMs that I think are actually useful for learning.
nav: true
nav_order: 4
---

{% if site.data.artifacts %}
{% include artifacts.liquid %}
{% else %}
Nothing here yet.
{% endif %}
