---
layout: page
permalink: /cmu-online/
title: CMU Online
description: Publicly Accessible CMU Courses
nav: true
nav_order: 5
toc:
  sidebar: left
giscus_comments: true
---
Education and knowledge should be accessible to anyone who is willing to learn.
There are many great course offerings from the top-ranked Computer Science
program in Carnegie Mellon University whose lectures are available on the public
domain. The purpose of this page is to curate these courses, so people can self-study and learn
from the best professors in the world without the hefty college price tag.
I hope that this resource will be helpful for both people currently in industry
aiming to improve their knowledge, current students to make course decisions
or to do self-learning, and even prospective students to get a taste of what
a CMU education looks like. In addition to courses, I have also included links
to weekly seminar series which are held by various research groups
for people interested in getting closer to state-of-the-art developments.

Courses which are accessible to undergraduates will have a
<span class="badge rounded-pill undergraduate"> </span> tag.
This does not mean that the difficulty is not suitable for graduate students;
on the contrary, almost all such classes are cross-listed as graduate courses
and are commonly taken by graduate students at CMU. There are also some
graduate classes (based on the course number) which are suitable for advanced
undergraduates which I marked as undergraduate as well.
Classes that require significant background and which are less accessible will
have a <span class="badge rounded-pill graduate"> </span> tag.

Some courses are only publicly available on Panopto instead of Youtube, in which
case I provided a link to the Panopto playlist. Don't feel too sad that this is
the case, because it can be more conducive to use Panopto than Youtube due to
the ability to customize speaker and presentation view.

Within each category, the courses are not ordered in any particular
way that is indicative of difficulty or the order in which they
should be taken. Consider it to be arbitrary.

Please let me know if there are any dead links, or if you spot an error with the
page. In addition, if there are courses that I missed which might be useful for
others, you can either modify the [relevant data file
here](https://github.com/fanpu/website/blob/master/_data/cmu_online.yml) and
make a pull request, or let me know via email. I will very much appreciate it!

The list of courses available publicly is heavily skewed towards which
departments or faculty members make their lecture recordings publicly available,
and therefore omits a lot of classes that Computer Science majors in CMU tend to
take. In particular, courses from the following departments in the School of
Computer Science are not present: Computational Biology, Human-Computer
Interaction, Institute for Software Research.

*The contents on this page is personally curated and does not reflects the views of CMU or SCS, and is also not endorsed by either parties.*

---

### Machine Learning
The machine learning courses in this section generally assumes working knowledge
of probability, statistics, calculus, and linear algebra. 
10-701 Introduction to Machine Learning or 10-715 Advanced Introduction to
Machine Learning are the recommended pre-requisites
for most of these courses if you do not have taken any prior machine learning classes.
Both of these introductory courses are also available in the table below.

{% include cmu_course_table.liquid courses=site.data.cmu_online.ml_courses %}

---

### Computer Systems
15-213 Introduction to Computer Systems is a required course for all CS majors in CMU,
and is the pre-requisite for all subsequent systems classes. It is a good place to start
if you are new to computer systems. In most other universities, this class is usually
referred to as an operating systems class.

{% include cmu_course_table.liquid courses=site.data.cmu_online.systems_courses %}

---

### Computer Science Theory
15-251 Great Ideas in Theoretical Computer Science is the introductory CS theory
class taken by CS majors in their first year, and would be a good place to start
if you are new to CS theory. All subsequent theory classes assumes knowledge from
15-251.

{% include cmu_course_table.liquid courses=site.data.cmu_online.theory_courses %}

---

### Programming Language Theory
Unfortunately, there are not a lot of publicly available programming language
(PL) theory lectures. However, [Robert Harper](http://www.cs.cmu.edu/~rwh/) and
[Jan Hoffmann](https://www.cs.cmu.edu/~janh/), staples of the PL scene in CMU,
both frequently give lectures for the [Oregon Programming Languages Summer
School](https://www.cs.uoregon.edu/research/summerschool/) that are much smaller
in scope than what is covered in a comparable full-semester course in CMU, but
still goes through many key ideas nonetheless.  Therefore, I have also decided
to include these resources than not have anything, which would be a shame as
CMU is a powerhouse in PL after all.

Why are so many courses titled 15-819? This is because the course number
is a catch-all for advanced topics in PL theory, so while the course name
is the same, the content can vary.

15-312 Foundations of Programming Languages is the introductory class
of the PL track, and would be a good place to start. The other introductory PL
class 15-317 Constructive Logic unfortunately does not have any publicly
available lectures.

{% include cmu_course_table.liquid courses=site.data.cmu_online.pl_courses %}

---

### Computer Graphics
{% include cmu_course_table.liquid courses=site.data.cmu_online.graphics_courses %}

---

### Robotics
{% include cmu_course_table.liquid courses=site.data.cmu_online.robotics_courses %}

---
### Mathematics and Statistics
In general, the math department does not record any of their lectures, so
this portion is pretty dry. However, it was worth having it just
to include [Po-Shen Loh](https://www.poshenloh.com/)'s lectures.
He coaches the USA IMO team and is well-loved by his students.

{% include cmu_course_table.liquid courses=site.data.cmu_online.math_courses %}