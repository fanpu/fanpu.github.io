---
title: "CMU 15712 Advanced Operating Systems and Distributed Systems Course Review"
layout: post
tags: [courses, cmu, systems]
keywords: 15712,15-712,cmu 15712,cmu 15-712,advanced os,advanced operating systems,cmu,carnegie mellon,computer systems,course review,courses,fanpu
cover: santorini_church.webp
cover_preview: santorini_church.webp
caption: Three Bells of Fira, Thera, Greece
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  15-712 Advanced OS was an excellent seminar-based graduate course that took us on a
  whirlwind tour through many of the most seminal SigOps Hall of Fame papers
  across several systems domains. It will prepare you to be a great systems
  designer and researcher. In this post, I will share my experience in the
  class, the course structure and content, what I thought were the biggest
  takeaways, and who this class might be suitable for.
---

This semester (Spring 2023), I took [15-712 Advanced Operating Systems and
Distributed Systems](https://www.cs.cmu.edu/~15712/) under professor [Phil
Gibbons](http://www.cs.cmu.edu/~gibbons/) and his TA and also PhD advisee [Val
Choung](http://nicebowlofsoup.com/). 

This class exceeded my expectations significantly.
I found it especially meaningful and apt since this was my last systems class before I
graduate, and the topics and discussions from class helped to unify all the
systems concepts that I had learnt from previous classes into a nice package
informed by common underlying principles: from distributed systems, to
networking, databases, filesystems, operating systems, and even machine learning
systems. 

# The First Lecture
The first lecture went through 2 Wisdom Papers, which no one was expected
to have read yet as it was the first class. You can refer
to the [slides here](https://www.cs.cmu.edu/~15712/lectures/01-intro.pdf) if you are curious.

The first paper, [Mythical Man-Month: Essays on Software
Engineering](https://www.cs.cmu.edu/~15712/papers/mythicalmanmonth00fred.pdf)
is a book by Turing-award winner Fred Brooks. It is about many of his observations
and principles on software engineering based on his own vast experiences. What
really brought it home to me was that a couple of them were also things that I
had some suspicions about previously, but never really thought it was universally
applicable, and thought they were simply artifacts of the way I approached things.

For instance, one of the principles is "Plan to Throw One Away", meaning that
one should first build a worthwhile system in a short amount of time, and then
re-build a better second version with the benefit of hindsight. This is because
one would end up having to re-build the system anyway after being confronted
with change and feedback, and also due to the following observation on program
maintenance:

> "Program maintenance is an entropy-increasing
> process, and even its most skillful execution only
> delays the subsidence of the system into unfixable
> obsolescence"

This had many parallels with my own experiences. For instance, my group ended up
having 4 major re-writes of our kernel during 15-410, and I also did a complete
re-write of my CloudFS filesystem for my 18-746 project. Similarly, many of my
internship projects were also re-writes and improvements on design of existing
systems that had accumulated too much technical debt. It does seem
a lot more reasonable to plan for this eventual change to begin with.

The paper also contained a lot of other great advice, such as the
importance of conceptual integrity to separate architecture
from implementation, structuring a team in a "surgical" fashion to drive software
development where the best programmer leads the most critical development
work like a surgeon and directs others on the other aspects, and of
course the famous Brook's law:

> “Adding manpower to a late
> software project makes it later”

The second paper, [You and Your
Research](https://www.cs.cmu.edu/~15712/papers/hamming86.pdf) by Richard Hamming
(of Hamming code fame), talks about how to become a great scientist. 
The following two slides gives a good sense of the spirit of the paper:

{% include figure.liquid 
    path="/assets/img/posts/adv_os/slide_1.webp"
    width="400px"
    class="z-depth-1"
    caption="How to be a Great Scientist (1)"
%}

{% include figure.liquid 
    path="/assets/img/posts/adv_os/slide_2.webp"
    width="400px"
    class="z-depth-1"
    caption="How to be a Great Scientist (2)"
%}

I mention the first lecture and the two papers that were discussed here not
simply because they were interesting, but because they helped to set the tone
and expectations for the rest of the semester going forward. The message is clear: this is going to be a practical and useful class that will help you on your journey to becoming great systems designers and researchers.


# Course Content

The class took us on a whirlwind tour through many [SIGOPS
Hall of Fame papers](https://www.sigops.org/awards/hof/), which the award
description states was "instituted in 2005 to recognize the most influential
Operating Systems papers that were published at least ten years in the past".
Reading through the papers helped to consolidate a lot of the knowledge that I
learned in previous systems classes, and it was cool to see how decades ago many
of these ideas that were once unappreciated or heavily criticized now form the
bedrock of many of the systems that we use today.

In addition to the Hall of Fame papers, there were also several relatively
recent papers that the course staff thought were conceptually interesting
and promising.

The following sections will go through each of the modules and
the required papers that you will read (refer to the [course website](https://www.cs.cmu.edu/~15712/syllabus.html)
if you are also interested in the optional papers), and a short description
of what the paper is about so you can get a pretty good sense of what is
covered. A cool thing to note is that the scope of all the papers will
touch almost all the systems classes offered at CMU.

## Part 1: Concurrency, Ordering, Races
- [Implementing Remote Procedure Calls (Birrell'84), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/birrell84.pdf) - introduced the now-standard design of RPC calls with interfaces and caller/callee stubs for distributed communication.
- [Time, Clocks, and the Ordering of Events in a Distributed System (Lamport'78),	SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/lamport78.pdf) - introduced Lamport clocks, the foundation for ordering distributed systems today
- [Distributed Snapshots: Determining Global States of Distributed Systems (Chandy'85), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/chandy85.pdf) - how to snapshot a consistent global state in a distributed system
- [Efficient Scalable Thread-Safety-Violation Detection (Li'19), SOSP'19 best paper](https://www.cs.cmu.edu/~15712/papers/li19.pdf) - using active testing to discover and reproduce concurrency bugs

## Part 2: File Systems and Disks
- [A Fast File System for UNIX (McKusick'84), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/mckusick84.pdf) - addresses many of the problems
of the original Unix filesystem by introducing the Fast File System (FFS),
and many of its ideas are now staple in modern filesystems like the Linux `ext*` filesystems
- [Scale and Performance in a Distributed File System (Howard'88), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/howard88.pdf) - introduces the Andrew File System (AFS) that significantly improved on NFS in terms of scalability. AFS was developed at CMU and is still widely used today.
- [The Design and Implementation of a Log-Structured File System (Rosenblum'92), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/rosenblum92.pdf) - 
introduced log-structured filesystems that support high write throughput which addresses the problem of disk traffic being dominated by slow writes
on traditional filesystems
- [A Case for Redundant Arrays of Inexpensive Disks (RAID) (Patterson'88), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/patterson88.pdf) - introduced RAID, solved problem of how to get cheap fault tolerance

## Part 3: Transactions and Databases
- [On Optimistic Methods for Concurrency Control (Kung'81), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/kung81.pdf) - introduced optimistic concurrency control, now standard in many database environments
with low contention and high throughput requirements
- [Concurrency Control and Recovery (Franklin'97)](https://www.cs.cmu.edu/~15712/papers/franklin97.pdf) - survey paper on concurrency
control and recovering from crashes in database systems (write-ahead logging, ARIES)

## Part 4: Fault Tolerance
- [Paxos (Lamport'01), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/paxos-simple.pdf) - simplified version of his original theatrical [The Part-Time Parliament](https://www.cs.cmu.edu/~15712/papers/part-time-parliament.pdf) paper that was mostly ignored, introduced how to get replicated logs
among unreliable (but not Byzantine) nodes
- [Practical Byzantine Fault Tolerance (Castro'99)](https://www.cs.cmu.edu/~15712/papers/castro99.pdf) - distributed consensus but in the
presence of Byzantine faults (i.e a fraction of the nodes can collude and behave maliciously)

## Part 5: OS Kernels and Virtual Machines
- [Microkernels (Liedtke'95), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/liedtke95.pdf) - the first demonstration of an
efficient microkernel designed with a very extreme minimality principle,
where as much OS functionality is moved outside of the microkernel
as possible, including even its memory manager, pagers, device drivers, software TLBs, etc.
- [seL4: Formal Verification of an OS Kernel (Klein'09), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/klein09.pdf) - first paper to perform a formal, machine-checked verification of a microkernel using the Isabelle/HOL theorem prover
- [Memory Resource Management in VMware ESX Server (Waldspurger'02), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/waldspurger02.pdf) - introduced many great ideas for hypervisor memory management, like memory ballooning, idle tax, and transparent page sharing that are now commonplace in modern virtual machines
- [The Scalable Commutativity Rule: Designing Scalable Software for Multicore Processors (Clements'15), SOSP'13 best paper](https://www.cs.cmu.edu/~15712/papers/clements15.pdf) - addresses
the problem of deciding whether there exists a scalable implementation (with respect to number of processors) of a program based on a restricted form of commutativity of the interfaces that it uses
called SIM commutativity (**S**tate-dependent, **I**nterface-based, **M**onotonic)
- [The Multikernel: A new OS architecture for scalable multicore systems (Baumann'09), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/baumann09.pdf) - solves the problem 
of modern operating systems not being able to take advantage of the current trend of increasing core counts due to the inherent limitations of the shared-memory kernel design,
by instead re-framing the OS as a distributed system split among different cores with event-driven execution, replicated state, and a hardware-neutral structure

## Part 6: Big Data Systems
- [Dynamo: Amazon’s Highly Available Key-value Store (DeCandia'07), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/decandia07.pdf) - the secret behind how
Amazon can support very high availability with eventual consistency due to
extreme business requirements (i.e users should never fail to add an item to
their shopping carts), by incorporating established techniques like consistent
hashing, sloppy quorums, gossip-based membership protocols, etc
- [Spanner: Google’s Globally-Distributed Database (Corbett'12), SigOps HoF paper](https://www.cs.cmu.edu/~15712/papers/corbett12.pdf) - built using lessons
from [Google Bigtable](https://cloud.google.com/bigtable), Spanner powers Google as their globally replicated transactional database system 
with 5 nines of availability with several novel ideas like TrueTime to quantify uncertainty in wall clock time and distributed transactions
- [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning (Qiao'21), OSDI'21 best paper](https://www.cs.cmu.edu/~15712/papers/qiao21.pdf) -
introduces a new notion of goodput that combines throughput and statistical efficiency to evaluate the performance of schedulers for training deep learning systems, 
with a practical implementation that optimizes both cluster-wide and per-job parameters called Pollux

## Part 7: Emerging Platforms
- [LineFS: Efficient SmartNIC Offload of a Distributed File System with Pipeline Parallelism (Kim'21), SOSP'21 best paper](https://www.cs.cmu.edu/~15712/papers/kim21.pdf) - optimizing the performance of distributed file systems (DFS) by decomposing DFS operations into pipelined stages and offloading networked stages to a SmartNIC asynchronously

# Takeaways From The Class
Here are my thoughts on the key takeaways from the class.

## Class Discussion
As a seminar-based class, one of the most surprising things for me was how fun
and valuable the class discussions were.  It was especially enlightening to
hear the comments of Ph.D. students who are working in systems and other fields
in computer science, who often had very different critiques and opinions of
the papers than what I had come up with, which often led me to wonder how they
got their perspectives and what their background is like. 
This was particularly true when someone mentioned glaring deficiencies and
problems with the paper that I had completely not even thought of.

However, one thing that made me sad was that attendance in class
started to fall after the halfway point of the semester. This
included quite a few of the students who used to give very insightful
and interesting responses and so the diversity of perspectives of the
discussions as a whole suffered.

While attendance is not strictly enforced, actively participating in the
discussions and being engaged in lectures is one of the most valuable
takeaways from this class, and positively impacts not just you but also
your classmates, and so I would strongly encourage anyone interested in the
class to attend all the lectures that you can.

## Tribal Knowledge
Another aspect of the class that I really appreciated was how Phil taught us a
lot of the spirit and tribal knowledge of doing CS research during his lively
lectures. These were often presented as off-hand remarks while presenting the
context or background of a paper, and provided insight into the zeitgeist of the
time, the motivations and challenges that the paper authors faced, and what the
authors went on to do in the future based on the impact (or lack of impact at
that time) of their work.

As someone who has not done a long-term research project with a faculty member
but am thinking about possibly doing a Ph.D. in the future, all of these were
very valuable wisdom which are not things that you can pick up easily yourself
from reading past papers or books. In fact, it almost felt as if I had my own
advisor at times.

## Witness the Evolution of Systems Research
As you read through the papers, you almost feel as if you are being put into the
driver's seat and can see how systems research has matured and evolved over the
past few decades. Seminal papers of the past tackled the most
general problems, although many of them lacked implementations or proper benchmarks
that would surely be grounds to be red-flagged and rejected from any systems
conference today. Many of the more recent papers strive to anticipate and build
for future changes in the computation landscape, have solid replicable
implementations and evaluations, and are a lot more careful about anticipating
and providing rebuttals for criticisms. 

## Personal Attention for Projects
I also really appreciated the personal attention that Phil and Val gave to us
by meeting with us every other week for our course projects. 
This is especially so if you consider that many advisors already have trouble
meeting their own Ph.D. students for an hour a week, whereas in this case the
course staff dedicated half an hour every two weeks for every single group in
the class (there were around 10), which I thought was some real dedication.
I will admit that I did not live up to my end of the bargain by spending
as much time on the project as I would have wanted to (compared to when I took 15-410).
One could always give excuses for anything so you don't have to listen to mine,
but if I had to reflect on it, it was due to a combination of high
workloads from other classes, the fact that this was not the highest priority for
the members in our group (my project partners were both quite busy with
their own research and I was busy with other classes), and some unexpected
obstacles in our project that forced us back to the drawing boards a few times
(our project interim report was drastically different from our initial
proposal).

## Fantastic Course Staff
Phil is a really good lecturer. He is very clear, the class pacing is great, and
the lecture slides are polished. He is very approachable and respectful
towards students, and puts in great effort to give a good and satisfying answer
to every question. 

Feedback for projects is prompt (there was no feedback for the paper summaries),
and the midterms were graded fairly quickly. 

Overall it is clear that the class is pedagogically mature and has benefited
from many rounds of feedback during past iterations.  It is rich in content,
is accessible and yet challenging to students from a wide range of backgrounds,
and will prepare one well for building systems in the future, be it in 
academia or industry.
# Course Structure
There are three main components to the class: paper summaries,
projects, and exams.

## 1. Paper Summaries
Before each lecture, the class is assigned a required reading and an optional reading. A paper summary of the required reading must be submitted before
the class, which will discuss both readings. 

The paper summary will contain 3 things: 
1. The 3 most important things in the paper, 
2. 1 most glaring deficiency of the paper (even highly celebrated papers have faults!), 
3. A conclusion on how you will use lessons from this paper to inform you on how
you will build systems in the future.

It took me on average 2-4 hours to read each paper and around 15 minutes for the summary.

The lectures for this class are front-loaded, meaning that during the first
two-thirds of the semester, you will meet 3 times a week for 80 minutes each,
while there will be no lectures at all during the final third of the semester,
and so "on average" throughout the semester you will meet twice a week. This is so that
students have enough knowledge and content to begin working on their course
projects early on in the semester.

There will be 3 short breaks in each lecture, where all students will get into
breakout groups and share and discuss among themselves one of the prompts
for the paper based on their paper summaries. Afterwards, all groups
are invited to share what they thought.

Reading and writing the paper summaries are the only "homework" you will get in
this class.

## 2. Course Project
There is also a semester-long course project with a significant systems
component in groups of three. This will begin in earnest after a third of the
semester, and all the project groups met with Phil and the TA Val once every two
weeks.  The deliverables include a project proposal, an interim report, a final
presentation, and a final report. The course project will be the
largest constituent of your final grade.

## 3. Midterms
Finally, there are two midterm exams, which are taken during class time.
The first is taken in the middle of the semester, and the second is taken
after all lectures have concluded.

Each midterm will cover content from a shortlisted selection of 10
of the required readings. There will be 9 questions on the midterm,
which covers 9 of the 10 papers, and you are only required to answer 7 of
the problems.

The course staff will also provide two past year exams to practice on, though
some of the readings may have changed since. 

It admittedly does seem quite daunting to have to study and be familiar with 10
papers spanning very different topics. I did not have time to actually re-read
all 10 papers to prepare for the midterm, and so the way I prepared was to go
through all the lecture slides again, re-read the most important sections of the
paper, and skim through the rest. Afterward, I attempted the past exams to fill
in any gaps that I may have missed. This strategy allowed me to do fairly well
on the exam.

# Workload
The class has a moderate workload for a systems class. Expect to spend 10-12 hours 
a week on the readings and paper summaries while lectures are ongoing, probably
a couple more hours once the projects get into motion midway through the
semester, and for it to consume a significant portion of your existence in the
last two weeks before the final presentations.

It is a far less demanding and stressful class than the legendary [15-410/605
Operating System Design and Implementation](https://www.cs.cmu.edu/~410/) class, 
so don't let the "advanced" in the course title scare you off from taking this class.
After all, most people taking this class are Ph.D. students who have their
own research to work on and can't exactly spend all their time on courses,
unlike undergraduates.


# Our Course Project, and Reflections
Our course project was on the automated optimal scheduling of data in dynamic neural
networks over heterogeneous GPUs for inference tasks in a pipeline-parallelism
fashion. This means that when a model is too large to fit on a single GPU
but instead has to be distributed across multiple GPUs, we aim to solve the problem
of finding the optimal way to perform this split in the presence of dynamism
in the network. In our case we focused on input dynamism, meaning that the
sizes of the inputs can vary, which can result in different execution times in different
segments of the network. We built a system called `DynPartition`, 
a reinforcement-learning based scheduler that uses Deep-Q Learning to learn
the optimal way of performing this split.

{% include figure.liquid 
    path="/assets/img/posts/adv_os/presentation.webp"
    width="400px"
    class="z-depth-1"
    caption="Giving our final project presentation in the Panther Hollow conference room at CIC"
%}

We had some positive empirical results on our benchmarks, but will require
additional future work to verify the generality of these results. Overall, I
thought it was a great experience working with PhD students and to learn from
their working styles and approach to solving problems. It was also really cool
to see the breadth and depth of projects presented by the other teams during the
final presentation, which was structured like a conference.

# Is This Class Suitable For Me?
I cannot recommend this course enough to anyone who has sufficient
background and have an interest in building systems, or systems research.

You should be sufficiently prepared for the class if you have taken
[15-410 Operating System Design and Implementation](https://www.cs.cmu.edu/~410/),
or any other equivalent rigorous operating systems design class in your undergraduate college.
Most papers draw heavily on low-level concepts from operating systems and assume
that the reader is familiar with them, and therefore familiarity with these ideas
is critical to understanding the papers.

I don't feel any of the other classes are as critical, as any new concepts can
be picked up relatively easily. For instance, a good grasp of considerations
involved in operating systems design means that it's not too hard to also
understand the challenges involved in filesystems or virtual machine design.
Having taken other classes would definitely still help to make the papers more
approachable though. For instance, the Pollux paper was not very approachable for
people who did not have prior exposure to machine learning systems, which led to the
course staff deciding not to include that as one of the papers tested for the
second midterm.

When I took the class, all the students were either Masters or Ph.D. students.
Strong undergraduates with sufficient background would also definitely do well
in the class.

# Why I Took This Class
I had to take a systems class this semester to fulfill my graduation
requirements for the MSCS program. I initially did include this class in my
shortlist of systems classes to take, but then thought it was just going to be a
paper reading class (not that I had been in one of such classes before, but it
just did not sound very interesting and felt like something I could do by myself
asynchronously after I graduate) and therefore was quite hesitant to take it.

As such, during registration week I settled on [15-618 Parallel Computer Architecture and
Programming](https://www.cs.cmu.edu/~418/),
since it included topics on GPU programming that aligned with my current
interests in machine learning. However, I did not feel like the class was
sufficiently challenging for me after the first lecture, as it was a bit too
slow-paced and simple for my liking as I already had exposure to most of the
topics from other system classes that I had taken. I decided to switch
to 15-712, and I knew immediately that it was the right class for me after the
first lecture. 

In a sense, this class was a hidden gem and I was really glad that I ended up
taking it.

# Acknowledgments
I would like to express my gratitude to [Albert
Gao](https://adbforlife.github.io/) for helping to proofread this article, who
took the class with me this semester.