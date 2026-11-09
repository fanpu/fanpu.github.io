---
layout: about
title: About
permalink: /
subtitle: ML Research Engineer, based in New York

profile:
  align: left
  image: untermyer_closer.webp
  image_circular: false # crops the image to make it circular
  more_info: >

news: true # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
selected_posts: true # includes a list of posts marked as "selected={true}"
latest_posts: false # includes a list of the newest posts
social: true # includes social icons at the bottom of the page
---

Hello! I am Fan Pu (曾繁朴). I am currently on a personal research sabbatical.
Previously, I spent 3 happy years at Jane Street, where I worked on LLM research and training,
and developing on our in-house coding models and evals.

I graduated with a B.S (2022) and M.S (2023) in Computer
Science from [Carnegie Mellon University](https://www.cmu.edu/).

At CMU, I was actively involved in the open-source programming assignment
auto-grading platform [Autolab](https://autolabproject.com/) from 2018-2023. I
served as the Masters Student Liaison for the [Singapore Students
Association](https://cmussa.org/). I also used to play Capture-The-Flag (CTF)
competitions with [PPP](https://pwning.net/). I previously interned at [Jane
Street](https://www.janestreet.com/), [Meta](https://about.meta.com/),
[Asana](https://asana.com/), and [Saleswhale](https://www.saleswhale.com/)
(acquired 2022). I was a TA for [10-708 Probabilistic Graphical
Models](https://andrejristeski.github.io/10708-S23/) in the Spring 2023
semester.

My academic interests lie in understanding the theoretical foundations for the
success of deep learning, particularly in optimization and generalization.

In my free time, I enjoy climbing, running, skiing, reading and
learning new things, and writing things for my blog. I used to
do sprint canoe competitively. If I have an extended break I enjoy traveling,
especially hiking and exploring the great outdoors. Most of the banner pictures
on my blog posts were taken during these hikes. My favorite classroom in CMU is
GHC 4303.

I grew up in my hometown Singapore before moving to the US for college and work.
I try to go back and visit once a year.

Feel free to reach out to me at fzeng[at]alumni[dot]cmu[dot]edu.

Regrettably, I am unable to provide referrals for people that I have not
directly collaborated with, as I cannot write you a meaningful recommendation.

I have a Technician amateur radio license, with callsign
[KC3UFE](https://www.fccbulletin.com/callsign/?q=KC3UFE).

This blog was originally started on 24 June 2018, although it has taken many
forms since then. All banner pictures on the blog are taken by yours truly!

<section class="home-section" id="talks">
<div class="home-section-side">
{% include home_section_head.liquid title="Talks" note="Slides I developed for talks on various LLM-related topics. You are free to share, adapt, and reuse these materials, provided that you give appropriate credit." %}
</div>
<div class="home-section-body">
{% include thumb_list.liquid items=site.data.talks limit=4 %}
</div>
</section>

<section class="home-section" id="starred">
<div class="home-section-side">
{% include home_section_head.liquid title="Starred posts" note="Some of my more popular posts." more_url="/blog/" more_label="All posts" %}
</div>
<div class="home-section-body">
{% include starred_posts.liquid %}
</div>
</section>

<section class="home-section" id="artifacts">
<div class="home-section-side">
{% include home_section_head.liquid title="Artifacts" note="Interactive explainers I made with Claude while trying to understand something." more_url="/artifacts/" more_label="All artifacts" %}
</div>
<div class="home-section-body">
{% include artifacts.liquid selected=true %}
</div>
</section>

<section class="home-section" id="games">
<div class="home-section-side">
{% include home_section_head.liquid title="Games" note="Some games that I wish existed but didn't, so I had Claude make them." %}
</div>
<div class="home-section-body">
{% include thumb_list.liquid items=site.data.games %}
</div>
</section>
