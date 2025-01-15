---
title: "Why Study Computer Systems"
layout: post
tags: [general, cmu]
cover: stoa_acropolis.webp
cover_preview: stoa_acropolis.webp
caption: The Stoa of Attalos and the Acropolis, Ancient Agora of Athens, Athina, Greece
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
published: false
---
Computer systems is the art and science of designing and building performant,
reliable, and secure systems. The remarkable development of computer systems has
brought about the third industrial revolution and created the greatest economic
opportunities of our times. 

From operating systems that run all our software, to compilers that lay the
foundation of all our programs, and even the Internet and distributed systems
powering services that we rely on daily, it is hard to imagine any aspect of our
lives that has not been transformed in one way or another due to computer
systems. 

In this post, I discuss why one should study Computer Systems.


The context of the post is targeted at undergraduates studying
Computer Science at Carnegie Mellon University, where
a common 

 wondering which concentration to
pursue as part of their degree requirements. but 
the underlying ideas and
principles are general and will be helpful regardless of your background.


## The Computer Systems Concentration
This section will briefly introduce the Computer Systems concentration, and what
to expect from system classes at CMU.  Feel free to skip this section if you are
not a CMU student. 

The concentration in Computer Systems is one of the many concentrations
offered to Computer Science majors in the School of Computer Science at CMU.
At CMU, CS majors are required to fulfill either a concentration or additional
minor in addition to their major requirements.

The Computer Systems concentration is by far the most popular concentration,
owing to its generality and broad appeal as it covers a wide range of topics,
and also due to its practicality, what it teaches permeates real-world systems
that people build and work with in industry, and often come up during technical
interviews.

In this post, I will talk about the Computer Systems concentration, what you will
gain by pursuing one, and the people for which this would be a good fit. 
I would like to remark that the content of this post is based on my personal
experiences and opinions, and other people could have very different experiences and 
takeaways. I decided to write this post as many juniors in SCS have asked me about the
different concentrations and my opinions of them, and I hope that sharing these
to a wider audience publicly will help to benefit more students in the future.
Roughly speaking, to complete the Computer Systems concentration, you will
have to take 2 core systems classes, and 2-3 other classes also considered
to live within the systems domain.

Most system classes carry a significant project component, where you will spend
most of your time of the class on, and where the majority of your grades will be
determined.  For instance, in 15-445 Database Systems you will have multiple
projects that will involve building different parts of a database system,
starting from the buffer pool manager to manage and optimize usage of your pages
requested from the OS, and culminating in support for concurrent query execution
and transactions.

Usually, there will also be written homeworks and exams, though those will be
relatively light compared to the projects. The questions on exams will usually
be similar to those on homeworks, and the course staff will usually also provide
previous year practice problems which are quite similar in content and style to
the actual exams. Mastery of knowledge gained from the projects is necessary but
not sufficient - exams will also tend to cover more of the theoretical or
design-oriented concepts that are difficult to incorporate into a course
project, so you will still have to review lecture slides in preparation.
Overall, this means that exams in systems classes are usually a low-stress
affair and you should expect to do fairly well if you prepare and
do a good job on the projects.

## What you will gain from a concentration in Computer Systems
The Computer Systems concentration was the first concentration that I fulfilled (the other
being Algorithms and Complexity). I had decided to go for it relatively early on, 
because it seemed to be so useful and applicable to such a wide variety of roles in industry.
However, little did I know that there was a whole lot more than simply this vague
notion that it would "be useful", which I hope to share in this section.

### Dirty Systems Programming Skills
In the words of Andy Pavlo, you will learn **dirty systems programming skills**
that will help you get the job done no matter what the scenario. 
You will become intimately familiar with a large portion of the POSIX interface, 
understand and be able to apply locking and concurrency mechanisms to appropriate
scenarios, know tricks for optimizing performance based on assumptions on 
how the system was built.

TODO: fanpu this should require more elaboration

### An Arsenal of Debugging Skills
Given the scale and complexity of the systems that we work with today, unless we
provably verify that each of the individual components of the system behaves
as expected and can never go wrong, it is *inevitable* that there will be bugs.
Having solid debugging skills is one of the most valuable skills of a software
engineer, and the process of debugging itself often reveals new perspectives and
insight into your code and its design that can inform its subsequent re-design.

For the more challenging course projects, expect to spend significantly more
time debugging and tuning your code than in writing the initial prototype itself.
These assignments are designed to require writing tricky code,
and to keep you outside your comfort zone.

Through these experiences, you will (be forced to) learn and master an arsenal of
debugging skills:
- The humble print statement is a versatile and formidable tool,
and its beauty is not in whether you use it but how you use it. You'll gain
intuition for what are the appropriate things to print, how they should be
formatted, when they should be invoked, and what information you should
be trying to learn from its output. For larger projects, you may even
write your own logging utility with several log levels.
- Debuggers like GDB are invaluable for low-level systems development, and for diagnosing
segmentation faults and memory corruption. Jumping between stack frames allows you
to easily view the calling context and the values of its variables,
without having to resort to a deluge of print statements.
Being able to see the contents of registers
is essential for operating systems and compiler development. You can also
easily view the memory layout of your data structures, allowing you to for instance verify
that fields are aligned in memory as you expected, and to view raw byte values that
may not be convenient to print due to the presence of 0 bytes.
The ability to watch addresses for write changes when debugging unexpected
memory corruptions also comes in extremely handy, which would be very difficult to trace otherwise.
Utilities like the record and replay framework
[rr](https://github.com/rr-debugger/rr) even allow you to perform
reverse-execution, allowing you to re-trace what happened to the program right
before a segfault.
- Tools like [Valgrind](TODO-add-here) allows you to easily detect memory leaks, double
frees, and use-after-free bugs with minimal effort. 
- Appropriate use of assertions to ensure that data structure invariants are preserved.
This helps you to catch violations early on (fail fast principle), instead of facing an obscure bug that has
become unrecognizable much further down the line. The complexity of these data structures means that
you will often have to write a function to check its invariants. The usage of these assertions in turn
means that you will re-design your code such that the invariants are preserved across function boundaries,
making them better designed and easier to reason about.
- Developing a robust "thought process" for debugging. System programming
novices tend to get discouraged when they encounter a bug that they have no idea
where it comes from, because "obviously my code is correct". Debugging is very much like doing
science: you come up with a promising hypothesis on what the issue could be, and then proceed to test
it using one of the strategies mentioned above. When you don't have any or run
out of such hypotheses, you will have no idea where to look. Some general guidelines:
    1. 
- When all else fails, going for the nuclear option of re-writing everything from scratch and making
minimal references to the old code. It is likely that there was some previous misconception that 
was deeply embedded in the code that you may no longer hold but still remains, 

### Programming and Debugging Resilience
While the number of lines of code is not the best measure of software complexity
for many well-known reasons, systems course projects can *easily* run into
thousands, if not tens of thousands of lines of code due to their
requirements and complexity. Being able to write so much code, and more
importantly *correct and robust* code, in a time-constrained environment with
frequent distractions and external obligations is *difficult*. Summoning the
internal motivation to begin writing a kernel from scratch from a `kernel.c` file
containing nothing but a single infinite while loop is *difficult*. 
Designing and planning out the different components of a large system without knowing
exactly what challenges you might face or what unintentional pathological
interactions might arise is *difficult*. The zero-to-one phase of systems development
where nothing works at all until when a minimal core set of components has been developed 
is daunting and requires faith in your abilities and code, since 
you are unable to develop incrementally or test anything until everything
comes together.
Coming back to your old code and re-orienting yourself to where you were
previously and determining what needs to be done next is *difficult* (in some
cases, it took me almost an hour re-gain my context). 
In short, it is *darn hard* to build systems, and it is the lifetime goal of practitioners
to build *good* systems. 

Through the trials and tribulations of the many course projects that you will
undertake, you will face and overcome these challenges again and again, and build
confidence in your own abilities. You will build the stamina for writing 
challenging and tricky code for many hours with intense focus. You will gain
the mental resilience and patience to track down the most subtle and transient bugs.
When things get tough, you'll be able to tell yourself that **you did it before,
and you'll do it again**.

### A Doctor, But Not That Kind
Doctors diagnose and prescribe treatment for sick patients. Similarly, as
someone who has spent a lot of time with computer systems, you will
also become proficient in figuring out why systems are slow or misbehaving,
and how to remedy it.

However, this requires an understanding of not just the specific part that you
are working with, but of how the entire system interacts as a whole, plus
the layer below the abstraction boundary that you are working with.
[TODO: insert doctor example here].
In a similar vein, the breadth of exposure that system courses will offer
you will allow you to understand and come up with ways to evaluate where things
could be going wrong.
For instance, suppose you realize that a RPC call occasionally takes an
exceptionally long time. Could this be due to periodic network congestion
resulting in dropped packets and repeated retries due to some other scheduled
network-heavy activities by other services? Or perhaps an expired edge cache that required
additional network hops to query the original overloaded server, which then suffered many
page faults as it attempts to re-compute the results? It helps to understand
the system from many different levels of abstraction to obtain a full picture.

### Performance Engineering

### From Principles to Practice
Something that may be initially surprising to many people is how the system
classes at CMU do not teach you how to use many of the standard tools in
industry, but rather focuses on principles and design trade-offs. This may
appear to put students at a disadvantage for internship searches and technical
interviews, but is in fact a deliberate and pedagogically crucial choice. 

The role of these classes is not to teach you how to use these tools and
frameworks, but to understand the timeless principles and ideas behind
how they were designed and why they were designed in this manner. Once you
understand and see the bigger picture, you will realize that all of these
industrial frameworks are imperfect and oftentimes lazy (for implementational
convenience) instantiations of one possible design of the system, out of 
many other possible competing designs with different trade-offs.

With this theoretical knowledge of how different systems are designed, 
learning a new framework becomes easy since it is just a matter of matching
concepts to their implementations. In fact, even before starting
you will already know roughly what kinds of APIs you need to achieve your goals,
and the options and settings you expect to be available.
The one slight complication is that what is done in practice is always
messier than what is prescribed in theory, and therefore there will likely be
more dials and knobs to configure, and more edge cases to consider. 

Practitioners who do not understand, or are not aware of the theory and
principles behind systems tend to compare one system against another based on
benchmarks such as their available features, speed, popularity, ecosystem
support, and so on. 
However, when you understand the theory, you will begin to view each system from the
use cases it was optimized for, and the design principles behind them that allows
them to perform well under these use cases. You realize that there is no *best*
way of doing things, only a *better* way based on balancing compromises.

### Preparation for Senior Technical Roles
There will come a time where you would probably 

If you want to be
work on
Google's globally-distributed database called
[Google Spanner](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf),
or 

### The Joy of Understanding
> For, verily, great love springs from great knowledge of the beloved object, and if you little know it, you will be able to love it only little or not at all. 
> *- Leonardo da Vinci*
