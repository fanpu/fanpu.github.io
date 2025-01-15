---
title: "Practical Systems Debugging"
layout: post
tags: [general, cmu]
cover: theatre_of_dionysus.webp
cover_preview: theatre_of_dionysus.webp
caption: Chek Jawa Wetlands, Pulau Ubin, Singapore
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
published: false
---

Given the scale and complexity of the systems that we work with today, unless we
can provably verify that each of the individual components of the system behaves
as expected and can never go wrong, it is inevitable that there will be bugs.
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
    1. TODO TODO
- When all else fails, going for the nuclear option of re-writing everything from scratch and making
minimal references to the old code. It is likely that there was some previous misconception that 
was deeply embedded in the code that you may no longer hold but still remains, 
