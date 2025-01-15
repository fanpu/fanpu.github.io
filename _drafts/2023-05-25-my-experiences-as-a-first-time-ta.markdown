---
title: "My Experiences as a First-Time TA in CMU"
layout: post
tags: [general, teaching, cmu]
cover: parthenon_diag.webp
cover_preview: parthenon_diag.webp
caption: The Parthenon of the Acropolis in Athens, Attica, Greece
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
    This semester was the first time that I was a teaching assistant (TA) for a class,
    which was for [10-708 Probabilistic Graphical Models](https://andrejristeski.github.io/10708-S23/)
    under professor Andrej Risteski. I really enjoyed the experience overall, and being able to
    lead two recitations for the class was one of the highlights of the semester for me. 
    The extensive interactions with students during office hours and on Piazza made
    me more aware of my own behaviors as a student in other classes, and
    having to read and grade many proofs also gave me insights on how to write
    clearer and more readable proofs from both positive and negative examples.

    In this post, I will share my personal journey of being a TA, from the unique
    perspective of being a TA for a graduate machine learning class in the Machine
    Learning Department. I will also share many of my observations (not intended
    to be prescriptive or suggestive in any way, I just found it interesting), some dos and
    donts of being a student to get help more easily and also to make life for
    course staff easier, and conclude with some remarks on why you might want to TA.
    I hope that this post will shed light on and provide a more complete picture of
    the TA experience for those who may be thinking about it, especially as TA culture
    is a really big thing in the School of Computer Science in CMU.
published: false
---
This semester was the first time that I was a teaching assistant (TA) for a
class, which was for [10-708 Probabilistic Graphical
Models](https://andrejristeski.github.io/10708-S23/) under professor Andrej
Risteski. I really enjoyed the experience overall, and being able to lead two
recitations for the class was one of the highlights of the semester for me.  The
extensive interactions with students during office hours and on Piazza made me
more aware of my own behaviors as a student in other classes, and having to read
and grade many proofs also gave me insights on how to write clearer and more
readable proofs from both positive and negative examples.

In this post, I will share my personal journey of being a TA, from the unique
perspective of being a TA for a graduate machine learning class in the Machine
Learning Department. I will also share many of my observations (not intended
to be prescriptive or suggestive in any way, I just found it interesting), some dos and
donts of being a student to get help more easily and also to make life for
course staff easier, and conclude with some remarks on why you might want to TA.
I hope that this post will shed light on and provide a more complete picture of
the TA experience for those who may be thinking about it, especially as TA culture
is a really big thing in the School of Computer Science in CMU.

## Previous Attempts to TA
Prior to this semester, I had applied once to be a TA for 15-410 Design and Implementation of 
Operating Systems for the Fall semester in my junior year, having taken the class
and found it extremely transformative the semester before. Unfortunately, I was
not selected to be one of the TAs, which was somewhat discouraging but also understandable
as I did not have any prior TA experience and was still quite early in my academic career
(all the other TAs selected were seniors/graduate students) and I gave up on the
idea of TA-ing for a while.

After that, I gradually developed an interest in computer science theory, and I seriously
considered the idea of being a TA 15-455 Undergraduate Complexity Theory during my senior
year. I wanted to do it under professor Ryan O'Donnell, who was by far my favorite
CS theory professor, but he was not the instructor for the class for any of the semesters,
and that plus my intended course load meant I ended up not applying for any TA roles again.

In hindsight, I realized that I was really picky in choosing a course to TA. 
I approached it with the same attitude as if I were going to join a company full-time: I had to
really feel passionate about the class and its materials, I had to like my boss
(the instructor), I had to feel sufficiently ready to know I could excel in my
role and have enough bandwidth for it... in reality, since it was only a
single-semester commitment, it was not really such a make-or-break decision. However,
being a TA also does represent a significant opportunity cost in terms of other things
that you could be doing on campus, so it should not be taken too lightly as well.

## Being a TA for 10-708
Halfway through the Fall'22 semester, I realized that I really enjoyed one of my
classes 10-708 Probabilistic Graphical Models, to the extent that I get excited
when it was Tuesdays or Thursdays because we would have lectures.  The course is
broadly about the theory of probabilistic graphical models, a framework used for
solving a wide spectrum of problems in machine learning, artificial
intelligence, computer vision, natural language processing, and so on. Examples
include undirected graphical models like Markov Random Fields, directed
graphical models like Bayesian networks, variational methods, Markov Chain Monte
Carlo, deep generative models, causality, and so on. The course was rigorous and
proof-heavy, drawing on many techniques used in computer science theory, math,
and statistics.

A few classes after this realization, I summoned up the courage after class to
ask the professor if he is looking for any TAs next semester, and that I would
be really excited if I could TA for the class. I was not really sure if I am
qualified for the role, as prior to this semester I had only taken two other
non-intro machine learning classes, and I often felt very out of my depth during
lectures. Furthermore, I had no prior TA experience. 

Andrej was really nice 

## Setting a New Homework Problem
I volunteered to set a new homework problem for our second assignment
on Markov Chain Monte Carlo methods. Previously, we had a problem
that required students to calculate the expected time to move 
from one node to some other node in a Markov Chain for 3 different graphs,
each with varying degrees of "connectedness": a clique, a line graph,
and a dumbbell-shaped graph. By computing this expected time,
students would be able to get a sense of how long it would take for
the Markov Chain to mix, based on the degree of the "connectedness" of the
graphs. This was important, since one of the key limitations of
Markov Chain Monte Carlo methods was that it was hard in practice to determine
when the Markov Chain has converged to a stationary distribution, and therefore
having some intuition about how the properties of the graph would influence that
is very helpful.

However, this was a relatively tedious problem involving computing many
recurrent expectations, and many students who were otherwise technically strong
got tripped up when they attempted to take limits of iterating the Markov Chain
instead. To make the problem more insightful and to provide a more general
result, I set about designing a new problem based on bounding
the mixing times of Markov Chains based on the difference between the
top two eigenvalues of the graph, i.e the spectral gap.



## Leading My First Recitation

## Some Personal Observations

### Impressive Students
The first thing that struck me was how good the students were. There were many
students in the class who have significantly more background than me in machine
learning. I don't think I ever had any PhD students in machine learning ever
come to my office hours, but I definitely had PhD students in other fields that
depended on tools from machine learning, as well as a lot of students from the MSML
and MCDS programs. Many of them are actively engaged in research,
and some of them have even worked as machine learning engineers full-time for
several years before enrolling in their programs. They were sharp, and there
were a couple of times during office hours where I was immediately corrected
when I said something that did not make sense.

Given the broad appeal of the class, there were also some students from
non-traditional backgrounds that were less prepared. As graduate students are
responsible for ensuring that they meet the course requirements themselves since
most of them did their undergrad elsewhere, there were some students that
did not really have sufficient background and it was difficult to help
them in a pedagogically useful manner since the chasm between their
current knowledge and what is required was simply too large.


I usually hosted one hour of office hours each week, with additional office
hours on the week the assignment was due for the assignments that I am in charge
of.

### Office Hours Preparation
Unfortunately I did not really have time to prepare much before my office hours.
In contrast, of my friends who were TAs for other classes would work through the entire
assignment to familiarize themselves with the problems and to get a sense of the
places where students would likely get stuck in order to prepare for their OH. 

However, it still mostly worked out fine as most of the proof-based problems
and programming problems were similar from the previous semester when I took it.
Furthermore, as
each assignment was released for two weeks, there were usually far fewer students
at OH during the first week, meaning I had more time to spend on each student
and to regain context on the trickier problems. 


and so it did not take too much time to regain context, though I still
felt apologetic when I had to take a few minutes to think 



### Types of Questions
Almost all the students who came to office hours were there to ask about
homework questions.  Occasionally, there were also questions about the direction
and content of their course projects. This was not too surprising,
as it makes sense that they would prefer to go to the professor's office hours
to ask about conceptual questions.

### Grading
I really enjoyed grading some of the proof-based questions that allowed creativity
in how it could be approached.

### Asking
When students 

## How To Be A Good Student

### Ask 


One large component of the
course is a semester-long project that involves doing research in groups related
to probabilistic graphical models.

<!-- I recall how when I was grading the student midway reports,  -->

The topics that the groups assigned to me chose were very diverse and
ranged from novel approaches to multi-agent reinforcement learning as evaluated
on StarCraft, to predicting the pathogenicity of protein variants using
graphical models. 


These could be PhD students

 It felt very much
different from wh


Students
<!-- We had a weekly hourly staff meeting, and used Slack for  -->



<!-- Back then, I was not entirely -->
<!-- sure whether he had a good impression of me, because I was always eating during class
(I had back-to-back classes before that and it was lunchtime) -->



<!-- (instead I took his classes 15-459 Quantum Computation and 15-751 A Theorist's Toolkit during
each of the semesters respectively), and I  -->


 and it was not as
stressful or all-consuming as I had worried. 



which was surprising given that I had been already thinking about doing so for quite
a number of semesters, but could not find a class that I felt really passionate about.


<!-- 
, which was surprising
given how it has been a while since I had wanted to try being a TA. -->