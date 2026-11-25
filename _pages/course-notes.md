---
layout: page
permalink: /course-notes/
title: Course Notes
description: Lecture notes from courses I took at CMU, typeset in LaTeX from my handwritten notes.
nav: true
nav_order: 3.5
---

These are my own notes, not official course material, so they may contain
mistakes. Please let me know if you find one. Each course is a single PDF, and
clicking a lecture jumps to its page. For what I thought of each course, see my
[course reviews]({% link _pages/courses.md %}).

{% if site.data.course_notes %}
{% include course_notes.liquid %}
{% else %}
Nothing here yet.
{% endif %}
