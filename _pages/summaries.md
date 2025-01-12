---
layout: page
permalink: /summaries/
title: ML Paper Summaries
description: >
nav: true
nav_order: 5
---

Summaries and critiques of papers (mostly in machine learning) I've read in detail. This is not a summary of
the traditional sense which will carefully go over all the major concepts in the
paper (due to time constraints); instead, it will be rather concise and only
contain the key points that I find interesting, with the expectation that the
reader already has some familiarity with the paper.

This serves to both catalog my own reading and academic progress, and may also
be of interest to others to find interesting papers to check out.

The format is inspired by the <a href="https://www.cs.cmu.edu/~15712/summaries.html">paper summaries</a> of a class I took.

{% include scripts/mathjax_macros.html %}

---

<ol>
    {% for summary in site.summaries reversed %}
    <li>
        <a href="{{ summary.url | relative_url }}">
            ({{ summary.date | date: '%b %-d, %Y' }})
            {{ summary.title }}
        </a>
    </li>
    {% endfor %}
</ol>
