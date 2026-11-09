---
layout: page
permalink: /artifacts/
title: artifacts
description: Interactive things I've made with LLMs that I think are actually useful for learning.
nav: false # flip to true once there are a few entries
nav_order: 6
---

{% if site.data.artifacts %}
{% include artifacts.liquid %}
{% else %}
Nothing here yet.
{% endif %}
