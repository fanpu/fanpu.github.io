---
layout: page
permalink: /courses/
title: CMU Course Reviews
description: Reviews and Recommendations for Computer Science and Math Courses at CMU
nav: true
nav_order: 3
toc:
  sidebar: left
giscus_comments: true
---

Here I share my thoughts on the courses that I took at Carnegie Mellon
University. Note that my experiences with some of these courses can be highly
dependent on the professors, or due to the impact of virtual classes during the
COVID-19 pandemic. Do also check out the [CMU Online page]( {% link
_pages/cmu-online.md %} ) for a curation of courses whose lectures
and course content are publicly available.

Course background: I graduated with a [B.S in Computer Science](http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/undergraduatecomputerscience/#bscurriculumtextcontainer) (2018-2022) with two concentrations:
[Algorithms and Complexity](http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/scsconcentrations/#algorithmsandcomplexitytext) and [Computer Systems](http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/scsconcentrations/#computersystemstextcontainer),
and a minor in [Mathematical Sciences](http://coursecatalog.web.cmu.edu/schools-colleges/melloncollegeofscience/departmentofmathematicalsciences/#minorstext).
I then did a [M.S in Computer Science](https://csd.cmu.edu/academics/masters/ms-in-computer-science) (2022-2023),
with a mostly ML, theory, and stats-heavy courseload.

Courses with a &#11088; are those that I found transformative,
and those with a &#10084;&#65039; are those that I enjoyed a lot.

I don't write much about the more popular introductory or required
courses, since there's already a lot of word about it on the street and there
isn't much for me to add which already hasn't been said.

Feel free to ask any questions about the courses at CMU, including those that I
have not taken, in the comments section below.

Inspiration for this page came from [Wan Shen's CMU course review page](https://wanshenl.me/courses/reviews/).

{% assign n =  site.data.course_review_blogs.blogs | size %}
{% assign blogs = site.data.course_review_blogs.blogs  | sample: n %}
Other similar CMU course review pages, in randomized order each time this site is re-built and deployed: {% for blog in blogs %}[&#9733;]({{ blog }}) {% endfor %}

Do email me if you would also like your course review page to also be included here.

_This page was recently updated on 07/20/2023 as I work on finishing my course reviews for the Spring 2023 semester._

~~_This page was recently updated on 11/9/2022 to help students decide the classes
they may want to take ahead of course registration week._~~

## Categories

For ease of navigation, I've also categorized the classes that I've taken into
the areas that they fall under, so you can just access the ones that
you might be interested in directly.

### Computer Science Theory & Algorithms

| Computer Science Theory & Algorithms                                          |
| ----------------------------------------------------------------------------- |
| [15-859 CC Algorithms for Big Data](#course15859CC)                           |
| [15-859 OO Randomness in Computation](#course15859OO)                         |
| [15-855 Graduate Computational Complexity Theory](#course15855)               |
| [15-751 A Theorist's Toolkit](#course15751)                                   |
| [15-459 Quantum Computation](#course15459)                                    |
| [15-455 Undergraduate Computational Complexity Theory](#course15455)          |
| [15-451 Algorithm Design and Analysis](#course15451)                          |
| [15-356 Introduction to Cryptography](#course15356)                           |
| [15-354 Computation and Discrete Math](#course15354)                          |
| [15-210 Parallel and Sequential Data Structures and Algorithms](#course15210) |
| [15-251 Great Ideas in Theoretical Computer Science](#course15251)            |

{: .table .table-bordered .table-hover .table-sm }

<br>

### Machine Learning, Statistics, and Artificial Intelligence

| Machine Learning & AI                                                       |
| --------------------------------------------------------------------------- |
| [10-716 Advanced Machine Learning: Theory and Methods](#course10716)        |
| [10-708 Probablistic Graphical Models](#course10708)                        |
| [36-709 Advanced Statistical Theory I](#course36709)                        |
| [36-708 The ABCDE of Statistical Methods in Machine Learning](#course36708) |
| [10-703 Deep Reinforcement Learning and Control](#course10703)              |
| [10-725 Convex Optimization](#course10725)                                  |
| [10-617 Intermediate Deep Learning](#course10617)                           |
| [10-701 Introduction to Machine Learning](#course10701)                     |
| [15-784 Foundations of Cooperative AI](#course15784)                        |
| [16-385 Computer Vision](#course16385)                                      |

{: .table .table-bordered .table-hover .table-sm }

<br>

### Computer Systems

| Computer Systems                                                          |
| ------------------------------------------------------------------------- |
| [15-712 Advanced Operating Systems and Distributed Systems](#course15712) |
| [15-410 Operating Systems Design and Implementation](#course15410)        |
| [15-411 Compiler Design](#course15411)                                    |
| [15-641 Computer Networks](#course15641)                                  |
| [15-445 Database Systems](#course15445)                                   |
| [18-746 Storage Systems](#course18746)                                    |
| [15-319 Cloud Computing](#course15319)                                    |
| [15-440 Distributed Systems](#course15440)                                |
| [15-330 Introduction to Computer Security](#course15330)                  |
| [15-213 Introduction to Computer Systems](#course15213)                   |

{: .table .table-bordered .table-hover .table-sm }

<br>

### Programming Language Theory

| Programming Language Theory                                           |
| --------------------------------------------------------------------- |
| [15-819 Advanced Topics in Programming Language Theory](#course15819) |
| [15-312 Foundations of Programming Languages](#course15312)           |
| [15-317 Constructive Logic](#course15317)                             |
| [15-150 Principles of Functional Programming](#course15150)           |
| [98-317 StuCo: Type Theory (Hype for Types)](#course98317)            |

{: .table .table-bordered .table-hover .table-sm }

<br>

### Mathematics

| Mathematics                                                          |
| -------------------------------------------------------------------- |
| [21-651 General Topology](#course21651)                              |
| [21-355 Principles of Real Analysis I](#course21355)                 |
| [21-329 Set Theory](#course21329)                                    |
| [21-484 Graph Theory](#course21484)                                  |
| [21-301 Combinatorics](#course21301)                                 |
| [15-259 Probability and Computing](#course15259)                     |
| [15-260 Statistics and Computing](#course15260)                      |
| [21-268 Multi-dimensional Calculus](#course21268)                    |
| [21-241 Matrices and Linear Transformations](#course21241)           |
| [21-122 Integration and Approximation](#course21122)                 |
| [21-120 Differential and Integral Calculus](#course21120)            |
| [15-151 Mathematical Foundations for Computer Science](#course15151) |

{: .table .table-bordered .table-hover .table-sm }

<br>

### Computer Science - Others

| Computer Science - Other                                           |
| ------------------------------------------------------------------ |
| [15-462 Computer Graphics](#course15462)                           |
| [15-300 Research and Innovation in Computer Science](#course15300) |
| [17-604 Communication for Software Leaders II](#course17604)       |
| [17-603 Communication for Software Leaders I](#course17603)        |
| [15-122 Principles of Imperative Computation\*](#course15122)      |
| [15-604 Immigration Course](#course15604)                          |
| [07-128 First Year Immigration Course](#course07128)               |
| [07-131 Great Practical Ideas in Computer Science](#course07131)   |

{: .table .table-bordered .table-hover .table-sm }

<br>

### General Education, StuCos, Others

| General Education, StuCos, Others                                                     |
| ------------------------------------------------------------------------------------- |
| [33-228 Electronics I](#course33228)                                                  |
| [70-350 Acting for Business](#course70350)                                            |
| [84-380 US Grand Strategy](#course84380)                                              |
| [85-102 Introduction to Psychology](#course85102)                                     |
| [79-396 Music and Society in 19th and 20th Century Europe and the U.S.](#course79396) |
| [80-100 Introduction to Philosophy](#course80100)                                     |
| [76-107 Writing about Data](#course76106)                                             |
| [76-106 Writing about Literature and Art](#course76106)                               |
| [69-151 Introduction to Yoga](#course69151)                                           |
| [98-242 StuCo: Introduction to Esoteric Programming Languages](#course98242)          |
| [98-212 StuCo: Competitive Computer Security](#course98212)                           |
| [98-349 StuCo: Design and Analysis of Logic Puzzle Games](#course98349)               |

{: .table .table-bordered .table-hover .table-sm }

---

## CMU Courses

### Spring 2023

{: .first-course-item #course10716 }

- &#11088; 10-716 &nbsp; **[Advanced Machine Learning: Theory and Methods](https://www.cs.cmu.edu/~pradeepr/716/)**, [ Pradeep Ravikumar](http://www.cs.cmu.edu/~pradeepr/)

  This was my favorite class this semester. It felt like a natural
  continuation of [10-708 Probabilistic Graphical Models](#course10708),
  with a strong focus on understanding the theory of how many non-parametric
  modern machine learning techniques work. It is very much a math and theory
  course and can get pretty dense at times, but it is also very rewarding.

  Since there does not appear to be much information online about the specific
  content covered, I'll try to summarize them here.

  1. Statistical decision theory: this was mostly a quick recap,
     since it was covered in the pre-requisite class 36-705.
     If you did not take 36-705, it'll be helpful to learn/review
     Bayesian estimators, and Bayesian and minimax risk as it'll
     be used throughout the course.
  2. Nonparametric Bayesian methods: the Dirichlet process for CDF estimation,
     Dirichlet process mixture for density estimation,
     and the Gaussian process for estimating a regression function
  3. Nonparametric Density Estimation: histograms, kernel density estimators,
     series estimators. A key result is how the kernel estimator is minimax-optimal
     over many classes of loss functions and function spaces.
  4. Nonparametric Regression: partition estimators, spline estimators,
     basis/dictionary series estimators, k-NN regression,
     smoothing kernel regression estimators, Reproducing Kernel Hilbert Space (RKHS)/Mercer kernel regression estimators, wavelets
  5. Nonparametric Classification: contrasting
     classification error between parametric vs nonparametric models,
     minimax rates of convergence for classification vs regression
     for different function classes of distributions
  6. Nonparametric Greedy & Boosting: Orthogonal Greedy Algorithm,
     Greedy Coordinate Descent (i.e boosting), Adaboost,
     functional gradient descent
  7. Optimal Transport: Monge assignments and the Kantorovich relaxation to
     motivate Wasserstein distance, the Kantorovich dual, integral probability
     metrics, applications to statistical estimation and Wasserstein GAN
  8. Deep Density Estimation: variational auto-encoders, normalizing flows, autoregressive flows, destructive distribution learning
  9. Deep Representation Learning and Kernels: RKHS kernel regression, RKHS in
     relation to representation learning, random features, randomly wired DNNs and
     its relation to Gaussian Processes
  10. Dimensionality Reduction & Manifolds: PCA, multi-dimensional scaling
      (MDS), kernel PCA, local linear embeddings (LLE), Laplacian eigenmaps,
      diffusion maps, Johnson-Lindenstrauss Lemma and random projections
  11. Clustering: k-means as vector quantization, mixture models and local non-identifiability,
      level set clustering, hierarchical clustering, spectral clustering
  12. Learning & Games: online learning, Follow the Leader (FTL), Follow the Regularized Leader (FTRL),
      regret bounds for FTRL on convex and non-convex action domains and loss functions, two-player games and Nash equilibrium
  13. Causality: adjusting for confounding, causal graphs and structural equations
  14. Random Forests and Kernels: Bagging, layered nearest neighbor (LNN), kernel-based view of random forests

  Pradeep is a great lecturer and he takes great effort to answer every student's questions in detail.
  However, while I often find myself being able to follow the current micro-level derivations and explanations,
  I often feel somewhat lost about how it fits in with the bigger picture and its relation with other techniques,
  possibly due to a lack of prior exposure to many of these topics.

  There is significant variance in the difficulty of homework problems in a
  single homework, which I thought made the scoring of the homework problems
  somewhat nonsensical as the points were more or less evenly distributed. I
  recall there were a couple of problems that demanded a fair bit of thought and
  insight but had relatively short solutions, and therefore only netted a
  moderate amount of points compared to the effort required.

  I thought the exams were relatively easy. It is proof-based and will ask you
  to perform derivations of results or techniques related to what is seen in class,
  which sounds much scarier than it actually is because the questions are relatively guided.

  One thing that happened this semester (and also apparently for prior semesters
  according to some people that I spoke to) was the steep drop in attendance as the semester
  went on. During the first lecture, there were barely enough seats for everyone
  and some people had to stand, but by the mid-way point the average attendance was
  just around 5 people. However, when it came to the midterms, the classroom filled up again
  and I even heard the instructor for the previous class remark how she didn't
  realize we had so many students in this class as she was leaving.

  This was probably not such a bad thing for the people who did come to lectures, since
  it meant more personalized attention from the instructor, more opportunities
  to ask your own questions, and a better view of the board.

  I initially thought this phenomenon was because the other students (who were
  mostly MLD Ph.Ds and MSML/MSDS students) were already very knowledgeable and didn't see the need to
  come to lecture, but only learned much later that it was actually because many
  of them also found the material challenging and found it difficult to follow the lecture.

  I think the takeaway here is that this will be a very difficult class (in my opinion likely
  the hardest class offered in MLD), so try to go in with friends and don't shy away
  from asking questions since it is likely that many people may also be confused.

- &#11088; 15-712 &nbsp; **[Advanced Operating Systems and Distributed Systems](https://www.cs.cmu.edu/~15712/)**, [Phil Gibbons](http://www.cs.cmu.edu/~gibbons/)
  {: .course-item #course15712 }

  I really liked this class and felt that there wasn't enough information about
  this course online by people who have taken it, so I wrote a dedicated
  post for it: [CMU 15-712 Advanced Operating Systems and Distributed Systems Course Review]({% post_url 2023-06-09-advanced-operating-systems-course-review %})

- &#10084;&#65039; 36-709 &nbsp; **Advanced Statistical Theory I**, [Matey Neykov](https://mateyneykov.com/)
  {: .course-item #course36709 }

  This course largely follows [High-Dimensional Statistics: A Non-Asymptotic Viewpoint](https://www.cambridge.org/core/books/highdimensional-statistics/8A91ECEEC38F46DAB53E9FF8757C7A4E) by [Wainwright](https://people.eecs.berkeley.edu/~wainwrig/),
  with the last portion branching off towards topics in the professor's own research interests.
  This is a core course in theoretical statistics that all stats Ph.D students
  must take, and most students in the course came from this demographic. The
  focus of the course is on high-dimensional statistical models, and
  non-parametric statistical models.

  There were 4 homeworks, with around 2-3 weeks between each one. They are all proof-based questions, with many
  of them coming from Wainwright's book. I found the homeworks quite challenging, and had to collaborate with
  a few stats Ph.D students to get through some of them (thank you for
  introducing me to the stats Ph.D. lounge!). There were some tools required
  for solving some of the problems that I have never seen or had to use before
  in any of my previous CS or math classes, but which may (?) be standard fare in
  statistical literature, that may have contributed to my difficulty.
  Fortunately, I still managed to solve _almost_ all the homework problems.

  Each student also had to scribe a lecture, which is then posted on Canvas as
  reference for all other students. There is also a project on deeply understanding a
  recent advanced theoretical paper in statistics or machine learning, which involves
  both a paper writeup and a presentation of its contents to the class.
  Thankfully there are no exams.

  Topics covered included concentration inequalities (sub-Gaussian,
  sub-exponential random variables), maximal inequalities, bounded differences,
  covering and packing, Gaussian and Rademacher complexity, chaining and
  Dudley's entropy integral bound, comparison inequalities (Slepian and
  Sudakov-Fernique) and lower bounds, high-dimensional and sparse PCA,
  Davis-Kahan theorem, LASSO in relation to prediction/support recovery/debiasing,
  covariance matrix estimation, non-parametric least squares, minimax lower
  bounds (Le Cam's method and Fano's method), Gaussian sequence model minimax rates.

  Overall I really enjoyed this class as it introduced me to many concepts in
  modern statistical analysis, which is really helpful in understanding
  statistical machine learning papers.

- 36-708 &nbsp; **[The ABCDE of Statistical Methods in Machine Learning](https://36708.github.io/)**, [Aaditya Ramdas](https://www.stat.cmu.edu/~aramdas/)
  {: .course-item #course36708 }

  I initially did not notice this course, as I thought ABCDE meant just the basics.
  However, what the course is actually about is a journey through various
  methods in statistical machine learning, viewed from the following lens:

  - **A**lgorithm design principles,
  - **B**ias-variance trade-off,
  - **C**omputational and memory considerations, **C**alibration, **C**onformal prediction,
  - **D**ata analysis,
  - **E**xplainability and interpretability.

  Aaditya is really clear, and he will re-iterate the important
  points many times throughout the class, sometimes too many times in my opinion,
  which was fine but it would also be good to learn more content instead. I
  also heard from some other students that he is one of the best lecturers in
  the stats department.

  Most of the class is on non-parametric methods, covering
  techniques like nearest neighbor methods, distribution-free predictive inference,
  calibration, decision trees, bagging, random forests, stacking, boosting,
  reproducing kernel Hilbert spaces (RKHS), kernel methods like kernel
  ridge regression and kernel PCA, Shapley values, spectral PCA,
  and some basic deep learning topics like deep PCA.

  I found the class very practical and helpful, and learned a lot as someone
  with no data science and practical machine learning background. This was
  especially from the many discussions on which techniques were suitable for
  which contexts, according to the ABCDE methodology.

  A lot of the topics in this course were familiar to other students who came
  from a traditional stats background, and I feel like you might get bored in
  this class if you already have a pretty solid stats foundation. Otherwise,
  there is a lot you'll learn.

  The midterms and homework were both relatively chill. Homework included both
  theoretical derivations, and also experiments on datasets using methods
  learned in class.

- 21-329 &nbsp; **Set Theory**, Benjamin Siskind
  {: .course-item #course36708 }

  I initially got interested in this class as quite a couple of theorems from my
  other classes appealed to important results from set theory, such as Zorn's
  Lemma used for proving Tychonoff's theorem. Since set theory is such a foundational
  topic in mathematics, I wanted to take it to satisfy my curiosity.

  This course covered all the chapters in [A Course on Set Theory](https://www.cambridge.org/core/books/course-on-set-theory/9E65D5D9CA561CA2D87F91B21B0D117D)
  apart from 6, i.e ZFC, order, cardinality, trees, and filters and ideals.
  It starts fairly slowly, and speeds up towards the latter of the second
  half. I could see some parallels between set theory and type theory
  initially in the more constructive concepts, but they soon began to differ
  greatly due to the heavy use of non-constructive methods
  in the proofs of set theory.

  The most interesting topic for me was the development of trees and Baire
  spaces, which led to the study of games and determinacy,
  with the consequence that sets which are determined have certain nice
  properties.

  The course was taught by Benny, a postdoctoral associate teaching the class
  for the first time, so it was expected that there would be a few rough edges.
  Benny cares about the class, is passionate about the course content (he
  does research in model theory and descriptive set theory), and responds to
  emails and clarifications fairly promptly.

  I think the course could have been paced faster at the start. It was also
  challenging to read the lecturer's handwriting sometimes. I felt there was
  significant time pressure for the exams (similar to [21-301
  Combinatorics](#course21301)), though the homeworks were quite reasonable.

- 17-604 &nbsp; **Communications for Software Leaders II**, [Dominick (Nick) Frollini](https://www.linkedin.com/in/frollini/)
  {: .course-item #course17604 }

  The highlight of this follow-up class to
  [17-603 Communication for Software Leaders I](#course17603) are the mock
  negotiation in-class exercises that we had to prepare and role-play for.
  Prior to each mock negotiation, we were provided with case
  materials developed by the
  [Kellogg](https://www.kellogg.northwestern.edu/) MBA program that contains
  important private information specific to our role, which must
  be kept secret.
  We then planned out our negotiation strategy before class, possibly with
  other people in the same faction so as to present a unified front
  against the other parties during negotiation.

  The mock negotiations were really exciting as they put us in the shoes
  of roles as diverse as CEOs, government officials, hardware designers,
  or even fishermen. You will have specific instructions on
  the kind of personality traits, negotiation style, and cultural
  customs that your role will have. I really indulged myself in this,
  especially since I took [70-350 Acting for Business](#course70350)
  not too long ago. The outcome of the negotiations will directly
  affect your grades, so these are really high-stakes, high-pressure
  situations that you will be put into. It was extremely fun and
  entertaining.

  Nick is a very passionate teacher with clear and well-paced lectures.
  It's obvious he has done this so many times that it's almost
  like a rehearsed performance. In my opinion, if you are already
  planning to take [17-603](#course17603), then you must take
  this course as well since the mock negotiations were so much fun.

Units: 60

This was the final semester (out of 2) of my Master's degree. My
goal for pursuing a M.S degree is to reach the frontiers of a field
and find a research interest, which I did in theoretical machine learning.
I mostly leave without regrets, except for one: not having been able to
work with any of the brilliant professors here on research, since I found
my direction so late.

I TA'd for [10-708 Probabilistic Graphical
Models](https://andrejristeski.github.io/10708-S23/) under [Andrej
Risteski](https://www.andrew.cmu.edu/user/aristesk/) this semester.
It was a fun experience and not as stressful as I had feared. Being
able to lead 2 of the recitations was one of the highlights of my semester.
I would like to thank [Andrej](https://www.andrew.cmu.edu/user/aristesk/) and my
fellow TAs (especially [Jennifer](https://scholar.google.com/citations?user=4i-T-5UAAAAJ), whose
office hours slot was right before mine and always stayed overtime
to help me with the queue) for making it so enjoyable. I am also
grateful to all my students for their great questions and being able to mentor
their impressive projects, and for giving me the opportunity to grow as a TA.

I also continued staying in an advisory role for
[Autolab](https://autolabproject.com/), being generally quite hands-off.
[Michelle](https://michellexliu.me/) was the project lead this semester and did
a terrific job. This being my last semester, I will dearly miss the team when I
graduate.

I did booth with [SSA](https://cmussa.org/) again this year, and contributed mainly
to the mechanical team and design teams. We built a Blitz booth themed after Singapore's
[Gardens by the Bay](https://www.gardensbythebay.com.sg/) and came in runner's up, after
a well-deserved win by [JSA](https://www.cmujsa.com/).

Academically, this semester was rather risky for me because I did not have the necessary
prerequisites for over half of my classes. I was missing
[36-705 Intermediate Statistics](https://www.stat.cmu.edu/~larry/=stat705/)
for [10-716](#course10716), [36-708](#course36708), [36-709](#course36709),
and missing [21-720 Measure Theory and Integration](https://www.math.cmu.edu/~gautam/sj/teaching/2022-23/720-measure/) for [21-640].
It was also a difficult and tiring process to get into [36-709](#course36709) as
the class was a required stats Ph.D. course that was generally very full, and
they had no reservations for CS majors. I ended up reading through some of the
lecture notes of [36-705](https://www.stat.cmu.edu/~larry/=stat705/)
over the winter break before the semester began to insure against the worst-case
scenario.

I ended up dropping 21-640 Functional Analysis halfway through the semester
as it became increasingly difficult to do (and even understand) the homework without
the necessary measure theory background.
While 21-720 was not listed as a prerequisite
on SIO for this course, it was announced as such (along with [21-640 General
Topology](#course21640)) by the instructor during the first class, so as to
motivate and develop many of the more interesting applications in the subject.

This concludes my wonderful time at CMU, and I leave with many beautiful
memories. Thank you to my professors who showed me the bountiful fields of their
intellectual gardens and sparked a lifelong joy and insatiable desire in this
student of wanting to explore more of it; to my friends who colored my life with
love and laughter through our journey of growth and humility, I hope we will
stay lifelong friends; and finally to my family for always being so
unconditionally supportive and for providing a safe harbor for me to turn to any
time.

### Fall 2022

{: .first-course-item #course15859CC }

- &#11088; 15-859 CC &nbsp; **[Algorithms for Big Data](https://www.cs.cmu.edu/~dwoodruf/teaching/15859-fall22/index.html)**, [David Woodruff](http://www.cs.cmu.edu/~dwoodruf/)

  Woodruff is one of the giants in sketching and numerical linear algebra, having developed many of its most important algorithms.
  There is even a `sklearn` function called the [Clarkson-Woodruff transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.clarkson_woodruff_transform.html) that is named after him.

  His teaching is extremely clear as he makes sure to justify and explain every step used in a proof. The analysis for many
  sketching algorithms is highly non-trivial, but Woodruff manages to pull off explaining it in a way that reads like a storybook.
  He cares deeply about the class and the student's learning, and one thing that still amazes me to this day is how he will respond to my
  Piazza questions on a weekend in 2 minutes consistently. I even made a [meme about it](https://t.co/zSg7KwmJkN).

  The homework problems are long but rewarding, and you will become intimately
  familiar with all sorts of linear algebra manipulations and properties.

  One caveat is that the weekly lectures are 3-hours with a 10-minute break in the middle. Given how dense the lectures are, this can be quite
  taxing, so bring snacks or caffeine if needed.

- &#10084;&#65039; 15-859 OO &nbsp; **[Randomness in Computation](https://www.cs.cmu.edu/~praveshk/randomness.html)**, [Pravesh Kothari](https://www.cs.cmu.edu/~praveshk/)
  {: .course-item #course15859OO }

  I had limited exposure to most of the topics in this course (mostly from
  Theorist's Toolkit 15-751, Graduate Complexity Theory 15-855, and Graph Theory
  21-484) such as spectral graph theory, expander graphs, derandomization, etc,
  and this course helped to solidify and reinforce my understanding.
  It also proved a lot of things that did not have time to be proved in those earlier classes.
  Overall I felt that Pravesh is a great lecturer, and the topics covered are very interesting
  and applicable. The course was offered for the first time this semester, so there
  were a few rough edges (i.e in the proof of Cheeger's inequality he initially
  did not want to prove it in terms of the Laplacian of the graph to avoid
  introducing unnecessary concepts and notation, but doing so ended up being more confusing
  than helpful), but overall it is an excellent class. The homework problems
  are reasonable and the workload is on the lighter side.

- &#11088; 10-708 &nbsp; **[Probablistic Graphical Models](https://andrejristeski.github.io/10708-F22/)**, [Andrej Risteski](https://www.andrew.cmu.edu/user/aristesk/)
  {: .course-item #course10708 }

  This class has a reputation of being one of the hardest ML classes, but I
  think it is actually an excellent class that is very well-taught, so I hope
  that this reputation does not discourage people interested in the content from
  taking it. The class can be categorized into three module: representation,
  inference, and learning. In the representation module, you will learn about
  how joint distributions of several variables can be represented efficiently by
  various models, taking into factors such as causal relationships. In the
  inference module, you will learn that sampling from such models is very hard
  in general (assuming $$\mathsf{P} \neq \mathsf{NP}$$), and develop probabilistic ways of
  sampling from them such as Monte-Carlo Markov-Chain and Variational Inference methods.
  In the final module, you will learn how such models can be fitted to training data.
  Graphical models form the backbone of many modern machine learning techniques like
  generative adversarial networks (GANs) and diffusion models, and the way that
  Andrej teaches all of these topics in a rigorous way to provide a solid
  mathematical understanding of how they work is essential for keeping up
  to date with the state-of-the-art in this field.

- 15-784 &nbsp; **[Foundations of Cooperative AI](https://www.cs.cmu.edu/~15784/)**, [Vincent Conitzer](https://www.cs.cmu.edu/~conitzer/), [Caspar Oesterheld](https://www.andrew.cmu.edu/user/coesterh/), [Tuomas Sandholm](https://www.cs.cmu.edu/~sandholm/)
  {: .course-item #course15784 }

  This course covered many topics in AI which are typically not covered in a
  machine learning course, such as normal and extensive form games, various
  forms of equilibriums in games, solving for such equilibriums, learning in
  games (regret matching), decision theories, and mechanism design.
  In fact, a lot of it comes from economic theory.

  The content was interesting, but I did not enjoy this course as much as I
  would have liked because the way that the content was presented was
  relatively hand-wavey, trading depth for breadth. That said, the course is
  being offered for the first time, so it will probably improve in subsequent
  iterations.

- 10-617 &nbsp; **[Intermediate Deep Learning](https://rsalakhucmu.github.io/10417-22/)**, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)
  {: .course-item #course10617 }

  Ruslan is one of the household names in the machine learning community (he
  invented the Dropout technique to prevent overfitting which is now standard in
  neural network architectures), and I was very excited to be able to take this
  class with such a legend in the field. I really enjoy his lectures, and he
  made many remarks about what was happening in the field when various
  techniques were being introduced as he was introducing them, which really
  gives you a sense of how the field has evolved over the last few decades from
  a man who has seen and been through it all.

  However, I think the course infrastructure requires more improvement. Some of
  the starter code for the assignments are quite poorly written and contain many
  inconsistencies and wrong/outdated documentation, which leads to a fair amount
  of frustration from students. One particularly annoying inconsistency was how
  the data formats of their starter code were transposed from Homework 1 to
  Homework 2. My guess is that someone tried to update the assignment but
  did not have time to fully go through to fix all the inconsistencies before
  it was released.

  Many people ask about whether they should take 11-485/785 (Introduction to
  Deep Learning) offered by the Language Technologies Institute (LTI), or this
  class offered by the Machine Learning Department (MLD). The main difference is
  that 11-485/785 is more hands-on and practical (most assignments are working
  on Kaggle datasets), whereas 10-417/617 is more
  theoretical.

- 10-703 &nbsp; **[Deep Reinforcement Learning and Control](https://cmudeeprl.github.io/703website_f22/)**, [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)
  {: .course-item #course10703 }

  The first half of the course follows the standard Sutton and Barto
  textbook pretty closely, but the second half discusses topics
  and techniques that are relatively state-of-the-art (think within the last
  3 years). As such, there is not really any reference material other
  than the papers that those techniques were based on.

  There are usually 2-4 papers that are compulsory readings assigned to be read
  before every lecture. Unfortunately I was a bad student and did not read them
  beforehand, and so after the middle of the semester once the content went
  beyond any standard textbooks, I found it pretty hard to focus and understand
  what is going on in the class, and got very little out of lecture. Eventually
  I had to rewatch them after reading through the papers again to be able to
  properly appreciate it. So if you taking this class, please avoid my mistake
  and do your readings before the lecture to save time in the long run!

  The homework for this class is really fun as you get to implement
  reinforcement learning algorithms for agents in various OpenAI Gym
  environments. All assignments are done in groups of up to 3, so remember to
  grab a friend or two if you're taking this class.

- 21-651 &nbsp; **General Topology**, [Florian Frick](https://www.math.cmu.edu/~ffrick/)
  {: .course-item #course21651 }

  This class generalizes many concepts that is taught in an undergraduate
  analysis course from metric spaces in $$\mathbb{R}^n$$ to arbitrary
  topological spaces. It took some time for me to un-learn some of the things
  that I implicitly assumed was just always true, i.e while in a metric space
  you learn that all sequences contains a convergent subsequences in compact
  sets, this is no longer true in arbitrary topological spaces. Much of the content
  have connections and parallels to other deeper areas of mathematics, which
  I found very beautiful.

- 17-603 &nbsp; **Communications for Software Leaders I**, [Dominick (Nick) Frollini](https://www.linkedin.com/in/frollini/)
  {: .course-item #course17603 }

  This course felt like an MBA class. It is a required class for Masters of Software
  Engineering (MSE) students, and one thing that I did not expect was how much
  the course was geared towards international students (i.e there was quite some
  emphasis on what is appropriate for US customs and norms), which is
  understandable as most of the MSE students taking the class are international.

  I found the segments about how to give oral presentations useful, especially
  the many tips and things to take note of when presenting.
  However, I don't think I gained as much from other topics, such as those
  concerning written communication.

- 15-604 &nbsp; **Immigration Course**, [Dave Eckhardt](https://www.cs.cmu.edu/~davide/)
  {: .course-item #course15604 }

  Every Monday night, the entire MSCS cohort will gather for this course, where
  Dave will talk about topics ranging from classes to grad school to things to
  do in Pittsburgh. It's usually pretty funny because Dave has a great sense of
  humor, but many of the sessions are also not critically useful so
  attendance does being tapering off in the middle of the semester once
  people start getting busy with school.

Units: 90

This was a really heavy semester for me, mainly because I had to juggle 4 group
projects (Deep Learning, Probabilistic Graphical Models, Cooperative AI,
Algorithms for Big Data) simultaneously starting from the midpoint of the
semester, and all of them were significant course undertakings which are
anywhere between 30-50% of the final course grade. This is still with
homeworks from all of these classes also being due concurrently,
with the exception of Algorithms for Big Data.

Fortunately, all the projects turned out relatively well and I was pretty happy
with them, but it definitely took a toll on my physical and mental health. I
ended up only mostly talking to my project groupmates over the last two weeks
of school when all the projects were due. It would probably be wise to learn
from my mistake and make sure that you don't have too many classes with
significant course projects on your schedule to avoid such a situation.

---

### Summer 2022

I decided to do another return internship with [Jane
Street](https://www.janestreet.com/), as I had a great experience with them the
previous year. Highlights of the internship included a two-week exchange at
their London office, where I managed to catch up with many high school friends
who are now either studying or working in the UK.

However, in hindsight I feel that it would have been better to diversify my
internship experiences and try interning at a significantly different company
(such as a ML-focused company). This is because internships are a relatively
low-cost 3-month investment for you to figure out what your interests are and to
learn about the problems faced by different industries. Since the internship
structure does not vary significantly between yeras, I ended up
attending a lot of similar talks and lessons for interns, which felt slightly
repetitive. Not to say that there's any fault with their internship, which is very
well-run and well thought-out as far as internships go - this is on purely
on the grounds of increasing my early career exposure to different types of companies.

### Spring 2022

{: .first-course-item #course10725 }

- &#11088; 10-725 &nbsp; **Convex Optimization**, [Yuanzhi
  Li](https://www.andrew.cmu.edu/user/yuanzhil/)

  I did not enjoy 10-701 as it covered a lot of content, but did not go into much detail about many topics,
  and I felt like there was no true understanding and everything was very hand-wavey. That
  experience made me hesistant to take any other ML classes.

  10-725 changed that for me, as Yuanzhi Li started from first principles and
  rigorously proved how many machine learning algorithms can converge in some
  amount of steps given various assumptions. For instance, just in the
  second lecture you will learn how gradient descent can converge to the optimum
  up to an epsilon error in a number of states which is proportionate to a
  chosen learning rate, given assumptions on the smoothness of a convex optimization
  landscape. This will then be extended to more complicated settings such as stochastic gradient
  descent, gradient descent with momentum (ADAM), distributed gradient descent, and even
  quantum optimization.

  Yuanzhi Li also understands that the students taking the class have very
  different learning objectives, and therefore the homework contains a mix of
  required and bonus problems, where the bonus problems are usually
  significantly more challenging than the required ones, but are tailored for
  people who really want to get deep into this stuff. The grading policy is
  extremely gentle and you essentially only have to score just half the points
  on the required problems to get an A-. The late day policy is also extremely
  generous (14 days), so it is quite a good class to take if you want some
  flexibility in your schedule.

- &#11088; 15-751 **[A Theorist's
  Toolkit](https://www.youtube.com/playlist?list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX)**,
  [Ryan O'Donnell](http://www.cs.cmu.edu/~odonnell/)
  {: .course-item #course15751 }

  This class aims to prepare students for doing theoretical computer science (TCS)
  research in the future. It covers a wide range of topics that frequently crop up in
  TCS research (see the playlist below for a taste), and the homework is challenging
  but fun.
  I have been previously exposed to a number of the topics covered in previous
  courses, but still found many of the topics new to me such as spectral graph
  theory and semidefinite relaxations incredibly cool. It is usually only
  offered once every few years, so if you have any interest in doing TCS-related
  research I would highly recommend that you take the class.

  Recordings from a past offering of the course can be found on this [Youtube
  playlist](https://www.youtube.com/playlist?list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX).
  It also includes recitation videos where Ryan goes through homework problems.

- &#10084;&#65039; 15-312 **Foundations of Programming Languages**, [Jan
  Hoffmann](https://www.cs.cmu.edu/~janh/)
  {: .course-item #course15312 }

  The programming portion of the homework is the most fun part of this class.
  You will get to write typecheckers and implement the dynamics of various
  well-specified languages, starting from the simple lambda calculus to
  concurrent Algol (which is Golang-like). This class is great for
  appreciating language design and understanding the pitfalls and
  antipatterns that pervades most programming languages used in industry.

- &#10084;&#65039; 21-355 **Principles of Real Analysis I**, [Robin
  Neumayer](https://www.math.cmu.edu/~rneumaye/)
  {: .course-item #course21355 }

  It is true that different sections of math classes taught by different
  professors are run like entirely different classes, so your experience with
  any math class highly depends on which professor you are with. I am really
  glad that I took the class under Robin, whose passion for teaching, and care
  and concern shown towards students (in one of the first few lectures she took
  some time to have everyone introduce themselves so she can know everyone on a
  more personal level), made me really enjoy the class and the subject matter.
  It also helped that she followed Rudin pretty closely, so anything that I was
  unclear about can be easily checked. Her weekly homework has reasonable
  difficulty and is quite fun.

  The pace of the course under Robin is faster than most standard analysis classes,
  where we covered up to Chapter 7 (Sequences and Series of Functions) in Rudin.
  I enjoyed the class so much that I wanted to take another analysis class in the
  future, which led me to take General Topology (21-651) in the subsequent semester.
  If you are interested in taking analysis, I cannot recommend doing it with Robin
  enough.

- &#10084;&#65039; 21-484 **Graph Theory**, [Wesley
  Pegden](https://www.math.cmu.edu/~wes/)
  {: .course-item #course21484 }

  The class was quite fast-paced and many times I felt quite lost during the
  longer proofs (some of them took multiple classes to prove). I definitely had
  to review the content after each lecture to get a better sense of the proof
  technique. However, this is more due to the complicated setup of many of the
  proofs. Wesley is a great teacher and I really appreciate how he would
  re-explain the key ideas many times in different methods so that we can
  understand them better. His exams are reasonable, and he will send out a
  review sheet between each midterm where you are only expected to remember the
  statements for some of the more complicated
  theorems, instead of being able to reproduce them fully.

  The textbook for the clas (Graph Theory by Diestel) is quite tersely written
  as it is intended for a graduate audience, so definitely do not miss lectures!

- 15-819 **Advanced Topics in Programming Language Theory**, [Robert
  Harper](http://www.cs.cmu.edu/~rwh/)
  {: .course-item #course15819 }

  The class covered many advanced topics in PL theory where there isn't much
  publicly available resources on, so when I got confused during lecture
  it was quite hard to find other materials online to supplement my learning.
  Very often I would end up on the nLab wiki, and I'm pretty sure
  anyone who has visited that knows that it's not entirely the best place
  for beginners to understand anything.

  The class is offered pass/fail to reduce stress, and homeworks are assigned weekly.
  You will be asked to re-do questions that you get wrong as Bob really wants
  to make sure that you understand the material. I think I generally had
  a ok-understanding of what was going on in the class until the segment on
  logical frameworks, where it became very abstract. I still hope to
  rectify this gap in understanding someday when I have time, perhaps by playing
  around with Twelf.

- 16-385 **[Computer Vision](http://16385.courses.cs.cmu.edu/spring2022/)**,
  [Matthew O'Toole](http://www.cs.cmu.edu/~motoole2/)
  {: .course-item #course16385 }

  I took this class as a follow-up to Computer Graphics (15-462). Many computer
  vision techniques covered will abuse linear algebra heavily, so if you were
  not already an expert in SVD decompositions and making use of
  eigenvalues/eigenvectors for engineering, this course will definitely make you
  one. You will also get very comfortable with convolutions and some elementary
  Fourier analysis. A major part of the course also covers convolutional neural
  networks, which is the mainstay of many modern computer vision methods.

- 18-358 **[Introduction to Amateur Radio](https://tomzajdel.com/ece358/)**,
  [Tom Zajdel (AI6CU)](https://tomzajdel.com/)
  {: .course-item #course16385 }

  This is a pass/fail mini-course, where the only requirement for passing the
  class is to obtain at least a Technician amateur radio license.

  To be very honest I was a pretty bad student for this course. Given the
  workload of my other classes, I skipped more than a few lectures and only
  studied extensively for the FCC licensing exam, a

  It was definitely a cool class, and we were also provided handheld radios and
  taught how to operate them to listen in on Buggy Net over [Spring
  Carnival](https://www.cmu.edu/engage/alumni/events/campus/spring-carnival/index.html)
  pretty early on. I was manning SSA's booth for a few hours during then, and
  used the chance to hear the Buggy Net operators announcing the start of each
  race, and live updates on when which buggy crossed a certain junction.

- 70-350 **Acting for Business**, Evelyn Pierce
  {: .course-item #course70350 }

  This class is similar to 54-191 Acting for Non-Majors, except with the individual
  and duet performances reversed. The class was very enjoyable (though somewhat
  slow-paced at times) and the workload is low. You get to perform skits like
  monologues, improv, and duets in front of the class, as well as watch your
  fellow classmates perform and give feedback.

  The class made me acutely more aware of things like body language, stage
  presence, and voice projection. It gave me the observational lens required to
  better critique plays and acting by others, and increased self-awareness of
  how I present myself to others.

  Even though the class is called Acting for Business, there isn't really a huge
  explicit business component to it, which I appreciated. I think the key
  takeaway is to understand how you present yourself, to increase your
  confidence and interpersonal communication skills, and to improve your
  personal charisma.

Units: 93

This was the final semester of my undergrad, and so I made sure to spend enough
time with my friends and cherish our remaining time together on top of just
academics. We went to Las Vegas during Spring Break together, and the crazy
things we did together were truly memorable.

I worked on Spring Carnival [booth](https://www.springcarnival.org/booth.shtml)
with [SSA](https://cmussa.org/), and we built a blitz booth with a Changi
Airport theme. This was the first time booth has been held since the start of
the pandemic in 2020, and we lost a lot of technical know-how and tribal
knowledge about booth building from our seniors who also used to work on booth,
but have graduated since. I mainly worked on the mechanical team, doing tasks
like cutting wood, screwing frames together, and construction of the booth.
It was a really fun experience and I managed to get to know many of my juniors
in SSA much better. To put the cherry on top, our booth came in first place in our
Blitz category!

On the academic side of things, I tried to make it more manageable by
watching the lectures for 10-725 and 16-385 asynchronously, which helped to
reduce the in-person fatigue levels.
Well actually, I was forced to watch it asynchronously as otherwise I had
no time to eat lunch at all, having back-to-back classes from 10am till 4.30pm daily.

However, there were still several pretty crazy weeks, especially when midterms
from multiple classes lined up in a single week. Thankfully, I really enjoyed
the content from all my classes, which energized and motivated me every day.

It felt incredible how quickly 4 years went by. It didn't feel like too long
ago when I was still a freshman, wondering what kind of path I would take,
and what sort of impact I would make at CMU. I guess I know the answer now!

---

### Fall 2021

{: .first-course-item #course15459 }

- &#10084;&#65039; 15-459 [**Quantum
  Computation**](https://youtube.com/playlist?list=PLm3J0oaFux3YL5qLskC6xQ24JpMwOAeJz),
  [Ryan
  O'Donnell](http://www.cs.cmu.edu/~odonnell/)

  This course covers the theoretical aspects of quantum computing, starting from
  the Elitzur-Vaidman bomb algorithm (I bet this already got your attention) to
  classical results like Shor's and Grover's algorithm. The class starts off initially
  relatively slowly, but then ramps up in the middle half.

  A curious trait of this class is that all weekly homeworks are optional, but you can turn
  in two selected problems per week to offset the weightage of your midterms and final exam.
  As a result, it is possible to score worse in the class if you turn in garbage for
  your homework but do well on the exam. My personal recommendation and what I
  did is to always do all the problems and turn in all the selected problems so that
  you can receive feedback. I only missed turning in homework for one week due to
  many overlapping deadlines from other classes. This approach helps you stay on
  track with the class and improves your confidence of your abilities during the midterms.

  Recordings from a past offering of the course can be found on this [Youtube
  playlist](https://youtube.com/playlist?list=PLm3J0oaFux3YL5qLskC6xQ24JpMwOAeJz).

- &#11088; 15-641 [**Computer Networks**](https://computer-networks.github.io/fa21/), [Peter Steenkiste](http://www.cs.cmu.edu/~prs/)
  {: .course-item #course15641 }

  I dedicated an [entire post]({% post_url
  2022-08-15-cmu-computer-networks-review %}) to review this course. Trust me,
  it's worth a read! But the TLDR is this course is very different from other
  systems courses, and you will likely stand to gain a lot from it even if you
  already have a lot of systems experience.

- 15-317 [**Constructive Logic**](http://www.cs.cmu.edu/~crary/317-f21/), [Karl Crary](http://www.cs.cmu.edu/~crary/)
  {: .course-item #course15317 }

  It can be hard to see how this class is relevant without subsequently taking
  other PL classes. I found the theoretical foundations gained from this course
  immensely helpful for preparing for 15-312 and 15-819. However, if you are
  only planning to take a single PL class, I would recommend another course that
  is more immediately applicable.

- 15-356 [**Introduction to Cryptography**](https://www.cs.cmu.edu/~15356-f21/), [Elaine Shi](http://elaineshi.com/)
  {: .course-item #course15356 }

  I found the class relatively easy as I had already taken 15-330 prior to this, and so I only started seeing
  new content near the middle of the course. The last part of the course covers
  blockchains and distributed consensus protocols, which is one of Elaine's major research areas.
  There were some pretty cool content like oblivious data structures,
  multi-party computation, and zero-knowledge proofs, but personally I wished
  that it went faster and covered more content like lattice-based and post-quantum cryptography.

- 15-462 [**Computer Graphics**](http://15462.courses.cs.cmu.edu/fall2021/), [Keenan Crane](http://www.cs.cmu.edu/~kmcrane/)
  {: .course-item #course15462 }

  The course assignments are pretty fun. and there's many moments of instant
  gratification, such as when you see your ray tracer paint a beautiful image
  of light refracting through glass balls. The course staff cares deeply about
  the class and there's a very active Discord server for the class to ask questions,
  get help, and post memes. While I think I am pretty competent at systems debugging,
  I found that debugging graphics code required a completely different set of
  skills since the visible effect of bugs are usually the result of many
  different matrix operations, which can be quite unintuitive to trace back to
  the source. It was a great learning experience for me.

  Recordings from a past offering of the course can be found on this [Youtube
  playlist](https://youtube.com/playlist?list=PL9_jI1bdZmz2emSh0UQ5iOdT2xRHFHL7E).
  In fact, when I took this course we were told to watch the lectures from this
  playlist asynchronously before class, and lecture time was reserved for
  in-class activities and Q&A. So together with the publicly available homework
  materials from [the course website](http://15462.courses.cs.cmu.edu/fall2021/)
  you can effectively self-study the class.

- 18-746 **[Storage Systems](https://course.ece.cmu.edu/~ece746/)**, [Greg
  Ganger](http://users.ece.cmu.edu/~ganger/), [George
  Amvrosiadis](http://users.ece.cmu.edu/~gamvrosi/), [Huaicheng
  Li](https://huaicheng.github.io/)
  {: .course-item #course18746 }

  During the first lecture, the professor talked about NAND flash alll the way
  down to the gory details about how it is implemented with MOSFET cells, and
  how reads, writes, and erases are performed at the voltage control level. I
  knew nothing about ECE, had no idea what was going on, and thought that if
  the entire course was going to be so low-level (since it is from the ECE
  department after all) I was probably going to drop the class.

  Fortunately, that was not necessary as most of the subsequent lectures stayed
  at the layer of abstraction of the software level, and therefore don't
  be afraid to take this class if you are not a ECE major.

  There are two big assignments in the class. In the first project, you will
  implement a Flash Translation Layer (FTL) for a SSD, and optimize your
  implementation such that it preserves the lifespan of the SSD for as long as
  possible for different workloads. This is because NAND flash can only tolerate a
  certain number of erases before it fails. In the second project, you will
  implement a FUSE-based filesystem (cloudFS) that automatically saves large objects
  to the cloud (i.e AWS S3), performs optimizations like de-deduplicating
  identical segments of stored files, and extend it to be able to
  read/write large files that can fit on the cloud but not on disk
  by performing segment-based read/write operations.

  Overall I thought it was a good class that provides you with a good sense of
  where the bottlenecks of various I/O-based workloads could lie (i.e database
  queries) for different filesystem designs, which is quite useful for
  optimizing the performance of many real-world systems you might work on in the future.
  It also gives you a good understanding of how many distributed filesystems (NFS, AFS, Google Filesystem)
  and storage systems (Google BigTable) that big tech companies use works.

- 84-380 **US Grand Strategy**, Andrew Stravers
  {: .course-item #course84380 }

  As someone who is interested in history and geopolitics I really liked this
  class. It offers a critical and balanced examination of past US foreign policy
  and grand strategy under various administrations. It was refreshing to see
  opinions which are not just variants of US/West is good and
  Russia/China/non-democracies are bad which permeates online discourse.
  Classes are structured such that the first hour is spent on lecturing, and the
  remaining 20 minutes is spent on group discussions.

Units: 75

---

### Summer 2021

I spent my summer as a Linux engineer intern in [Jane Street](https://www.janestreet.com/)
to work on exciting systems-level problems, and had a lot of fun exploring NYC
with other interns. I learned the following summer that the VM manager that I wrote
for my first project is now the standard way that VMs are provisioned by
different teams in the firm, which made me feel that my work was very impactful
and valuable.

---

### Spring 2021

{: .first-course-item #course15855 }

- &#11088; 15-855 [**Graduate Computational Complexity
  Theory**](https://youtube.com/playlist?list=PLm3J0oaFux3b8Gg1DdaJOzYNsaXYLAOKH),
  [Ryan O'Donnell](http://www.cs.cmu.edu/~odonnell/)

  This was my first graduate theory class, and needless to say I struggled a lot
  with the homework. Interpreting the homework problem was oftentimes
  a challenge in itself, and there were problems that I was stuck on for days,
  which at the time was really depressing and led to me questioning my intelligence
  and aptitude. However, in hindsight I grew a lot from this experience,
  and the euphoria of finally solving something that you have been struggling
  with for ages is indescribable. I gained a lot more confidence in my
  mathematical abilities, and became better in engaging in long periods
  of deep thought about a problem.

  Recordings from a past offering of the course can be found on this [Youtube
  playlist](https://youtube.com/playlist?list=PLm3J0oaFux3b8Gg1DdaJOzYNsaXYLAOKH).

- &#10084;&#65039; 21-301 [**Combinatorics**](https://www.math.cmu.edu/~jmackey/math301/syll.html),
  [John Mackey](https://scholar.google.com/citations?user=uOCVq1cAAAAJ&hl=en)
  {: .course-item #course21301 }

  I loved 15-151 (Concepts in Mathematics) and after that class I swore that if
  Mackey ever taught another class, I will take it for sure. Mackey is just as
  great and fun in Combinatorics as he was in Concepts. The homework is
  challenging, and the exams even more so. After the first midterm (where the
  variance was really high) he explained that he had to make the course
  challenging because there were many people who already came in with a lot of
  math competition background, the bulk of which involves combinatorics, and he
  did not want them to be bored. Unfortunately I did not have such a background
  and struggled with solving all the problems during the stressful time-crunch
  of the 50 minute midterm (I can't really think creatively when I am stressed),
  and this class ended up being the only B that I got in my undergraduate
  career (all others A). That said, I don't really care about my GPA and my main
  goal is to optimize for learning, so I am still very glad I took this class.
  Not having a 4.0 also ironically gave me more courage to take a more challenging
  course schedule in the future, as I no longer had to worry about losing my
  perfect GPA.

- &#10084;&#65039; 85-102 **Introduction to Psychology**, Kody Manke
  {: .course-item #course85102 }

  My view of psychology was pretty misguided (think Sigmund Freud, Carl Jung)
  before coming into this class, and it ended up being one of my favorite geneds.
  Kody is a very energetic and fun lecturer, and the class helped to open my eyes
  to many social phenomenom that I encounter in daily life, such as stereotype
  threat and priming.

- 10-701 [**Introduction to Machine Learning**](https://www.cs.cmu.edu/~aarti/Class/10701_Spring21/index.html),
  [Geoff Gordon](http://www.cs.cmu.edu/~ggordon/), [Aarti
  Singh](http://www.cs.cmu.edu/~aarti/)
  {: .course-item #course10701 }

  I did not really enjoy this class, and felt that it tried to cover too much
  too quickly, which led to a very shallow and surface-level understanding of
  many topics. Maybe this is the whole point of an introductory survey class,
  and maybe my lack of background in the area also led to me feeling oftentimes
  very confused about what is going on. I don't feel like I remembered
  a lot from this class, and none of the homework assignments were memorable.

  Unfortunately, there are not many great options for introductory ML classes.  
   People I knew who took the undergraduate Introduction to ML class (10-315)
  complained about it being hand-wavey and not rigorous, and 10-601 is sort of
  an in-between. As such, if you are a motivated and mathematically mature
  student, 10-701 would still probably be your best option for an ML class.
  Another choice would be 10-715, which is geared towards Ph.D students in the
  Machine Learning Department (MLD) and is the fastest-paced and most
  challenging option.

- 15-354 [**Computation and Discrete Math**](https://www.cs.cmu.edu/~sutner/CDM/index.html), [Klaus Sutner](https://www.cs.cmu.edu/~sutner/index.html)
  {: .course-item #course15354 }

  Klaus has a reputation of trying to add abstract algebra in everything that he
  teaches. This was definitely true in his trademark class Computation and
  Discrete Math (CDM), where he condensed a semester's worth of material from 21-373
  Algebraic Structures covering groups, rings, and fields into two lectures, and
  for good measure also threw in some discussion about Galois theory. This
  culminated in a formal introduction of Polya counting, algebraic feedback
  shift registers, and many other algebraic applications.

  There was also a lot of discussion about advanced concepts in automata theory
  in the first part of the course, building upon the basics introduced in 15-251
  Great Theoretical Ideas in Computer Science. This was his research area and
  was therefore unsurprising.

- 15-440 [**Distributed Systems**](https://www.andrew.cmu.edu/course/15-440/),
  [Mahadev Satyanarayanan](http://www.cs.cmu.edu/~satya/), [Padmanabhan
  Pillai](https://www.andrew.cmu.edu/user/pspillai/)
  {: .course-item #course15440 }

  Having already taken 15-410 (Operating Systems) which covers many concurrency
  and cache coherence concepts, 15-319 (Cloud Computing) which has an entire
  module on distributed consensus protocols, 15-445 (Database Systems) which
  talks about distributed databases, I did not really get much out of this
  course and wish I had taken something else. The projects were fairly
  straightforward compared to the other systems classes I have already taken.
  This course is usually considered the easiest systems class and I would only
  recommened it if you are new to systems or if this is the only systems class
  that you are planning to take.

- 98-242 **StuCo: Introduction to Esoteric Programming Languages**, [Ari Cohn](https://wafflecohn.github.io/), Jonathan Loungani
  {: .course-item #course98242 }

Units: 69

---

### Fall 2020

- &#10084;&#65039; 15-411 [**Compiler
  Design**](https://www.cs.cmu.edu/afs/cs/academic/class/15411-f20/www/), [Seth
  Goldstein](http://www.cs.cmu.edu/~seth/)
  {: .first-course-item #course15411 }

  Compilers is the other "star" systems class in CMU apart from OS. The best
  part of the course were the labs by far. They are done in groups of 2, and
  you can choose to use OCaml (best-supported, most people use this), C++ (ok
  support), or Rust (very limited support, maybe better now?). Why
  not be the next legend to write it in JavaScript? (please don't: reasoning will be given later). There are 6 labs
  in total, with each lab depending on previous labs. They are as follows:

  1. Lab 1: You will implement lexing, parsing, typechecking,
     register allocation, and code generation in x86
     for a very simple language that only performs arithmetic
     for integers.

  2. Lab 2: You extend the language that your compiler
     can compile to boolean
     expressions (including integer comparisons that results in booleans),
     conditional expressions, while and for loops.

  3. Lab 3: The language that your compiler supports now includes
     function prototypes, function declarations, typedefs.
     It will be necessary to perform some tail-call optimizations to
     avoid stack overflows on some of the test cases that performs
     recursive calls.

  4. Lab 4: Your compiler must now be able to allocate/free memory
     on the heap, deal with pointers & deferencing, support structs
     and field dereferencing/selection, support allocation
     and de-allocation of fixed-sized arrays of a given type.

  5. Lab 5: In my opinion, this lab was the most fun. The goal is to implement
     various compiler optimizations such as strength reduction, single static
     assignment (SSA), improved register allocation, aggressive deadcode
     elimination, partial redundancy elimination, etc. You need to beat `gcc -O1`
     in order to receive full credit, and it is possible to earn bonus points if
     you can beat it further. There is a scoreboard as well, which encourages teams
     to be competitive and creative with their optimizations. <br><br>
     The running joke about Lab 5 is that many teams spend most of it implementing
     SSA, but then run out of time to actually build optimizations on top of it.
     This is because SSA is just a mechanism for transforming your intermediate
     representation into a form that is more suitable for implementing many other
     globla optimizations, but implementing it just by itself without optimizations
     does not speed things up and instead makes your code slower since code in SSA
     form is longer. <br><br>
     This lab allows you to go as crazy as you want, and you can work on whatever
     optimizations that interests you the most personally. For instance,
     in the past there was a group that wrote an optimization to vectorize
     operations using SIMD extensions. It did not really improve the
     performance of the binary by a lot on most of the test cases, since
     most of them are testing performance on other cases like deadcode elimination
     or loop optimizations, but it was a cool project nonetheless.

  6. Lab 6: In the last lab, you are asked to write your own proposal
     for extending your compiler, and then to implement it. My group
     chose to add object-oriented features to the language, including inheritance,
     access modifiers (i.e `private/public/protected`), and runtime polymorphism via vtables.

  Ok, so I mentioned previously that you shouldn't use some weird language
  for compilers unless you really know what you are doing. This is because
  in Lab 5, the amount of time given to your compiler to compile each
  file is limited. Some of the more advanced optimizations are actually
  very computationally expensive, and therefore you want to make sure
  that the language that you use is fast, or otherwise you know how to
  make it fast. There are many teams using OCaml that lost points on Lab 5
  because their compiler could not compile many of the test cases in time (this
  can be avoided by optimizing your compiler code, even if it's heap-based like
  OCaml, although it might be trickier since people are generally less familiar
  with how to write performant OCaml code). We also faced compiler runtime
  issues with a few test cases initially even though we were on C++, but we were
  able to resolve all of them by performing some optimizations.

  Another fun part about this course is that on top of the test cases provided
  to you for each lab, you will also have to write at least 20 programs to be
  submitted as test cases, which will be added to the common pool of test cases
  that everyone will be evaluated on. This creates some sort of cat-and-mouse
  game where over the years the tests get increasingly harder and catch even
  more edge cases, and your compiler must be faster and be more robust. Some of
  the test cases even revealed bugs in the reference compiler. Of course, there
  is a limit to how adversarial the programs can be - the reference compiler
  must be able to compile them within the time limit. Our team did something
  quite evil by binary searching for the parameters determining the difficulty
  of our programs that barely passes the threshold of what the reference
  compiler can accept, which tripped up quite a number of other teams.

  Another piece of advice for this class is that you cannot afford to fall behind on any of the
  labs. This is because each lab depends on the successful implementation of
  previous labs, and your compiler will be run against test cases in previous
  labs. In other words, if you are failing 5 test cases in Lab 2, then you will
  at minimum continue to fail all 5 of them in Lab 3 because they will
  re-appear. This makes it critically important to get everything right for each
  lab when it is due, if not the amount of points that you lose will continue to
  compound to subsequent labs.

- &#11088; 15-455 **Undergraduate Complexity Theory**, [Venkatesan Guruswami
  ](https://people.eecs.berkeley.edu/~venkatg/), [Jonathan Mosheiff](https://www.cs.bgu.ac.il/~mosheiff/)
  {: .course-item #course15455 }

  I really enjoyed 15-251 Great Theoretical Ideas in Computer Science under
  Venkat, and decided to take this class with him as well. Undergraduate
  Complexity Theory (UCT) is considered to be the follow-up class to 15-251.
  The lecture pacing was quite fast, and he covered a lot of advanced topics
  which would usually only be touched upon in a graduate class (i.e parameterized complexity, pseudorandomness), but all for the
  better. The homeworks and exams are notoriously difficult and I ended up
  spending much more time on this class than I expected to (~20 hours/week), but
  it was a great preparation for the follow-up 15-855 Graduate Complexity Theory
  course that I took next semester. Also, don't worry too much
  about doing extremely well on his exams because they are always generously
  curved at the end.

  Unfortunately, Venkat is now no longer at CMU, but has moved to Berkeley,
  which in my opinion is a very great loss for CMU's theory community.
  He is truly an amazing lecturer.

  Recordings from a past offering of the course by Ryan O'Donnell can be found
  on this [Youtube
  playlist](https://www.youtube.com/playlist?list=PLm3J0oaFux3YL5vLXpzOyJiLtqLp6dCW2).

- &#10084;&#65039; 15-445 [**Database Systems**](https://15445.courses.cs.cmu.edu/), [Andy Pavlo](http://www.cs.cmu.edu/~pavlo/)
  {: .course-item #course15445 }

  Andy's lectures are a whole lot of fun, and it is hard to not be fired up by
  his enthusiasm and energy for databases. You will get to build the major
  components of a database through the class projects. Did I also mention that
  Andy really loves Wu-Tang Clan, and [once gave an extra credit question worth
  the entire final if you can list all members of Wu-Tang
  Clan](https://twitter.com/andy_pavlo/status/1204470956141547523)?

  Recordings from a past offering of the course can be found on this [Youtube
  playlist](https://youtube.com/playlist?list=PLSE8ODhjZXjbohkNBWQs_otTrBTrjyohi).

- 15-300 **Research and Innovation in Computer Science**, [Bogdan
  Vasilescu](https://bvasiles.github.io/), [Jonathan
  Aldrich](https://www.cs.cmu.edu/~./aldrich/)
  {: .course-item #course15300 }

  The goal of this class is to encourage undergraduates to contact professors
  and get into research. However, from what I saw people who already started
  doing research before the class simply continued with their existing topics,
  and those that were not engaged in research previously also did not continue
  after the class. In other words, don't expect this class to suddenly turn you
  into a researcher - the motivation must come from within.

- 98-212 StuCo: [**Competitive Computer Security**]({% link
  /assets/notes/98_212_Stuco_Syllabus.pdf %}), [Chris
  Lambert](https://github.com/cslamber)
  {: .course-item #course98212 }

  This was a great class on popular cyber-security Capture-The-Flag (CTF)
  topics. The depth of the topics covered go beyond introductory pretty quickly,
  so even CTF veterans will take away a lot from this course. Unfortunately, it
  is unclear if/when the class will be offered again.

Units: 48

I dropped 15-312 Foundations of Programming Languages in the middle of the semester, as UCT ended up taking much more time than I anticipated, and so that I could spend more time on compilers.

---

### Summer 2020

Due to the COVID-19 pandemic, I ended up doing my internship with [The Company
Formerly Called Facebook](https://about.meta.com/) on the [Tupperware
Scheduler
team](https://engineering.fb.com/2019/06/06/data-center-engineering/twine/) (now
called Twine) remotely from Pittsburgh. Twine is like Kubernetes, but can
orchestrate containers on up to millions of servers, instead of just a few
hundred. We were facing very interesting scaling engineering problems at the
time, due to the huge surge in traffic on Facebook products driven by the
pandemic. As part of the efforts to address this issue, I was tasked with
writing a large-scale load testing framework to see if Twine was battle-ready to
withstand the increasing load placed by their billions of users.

Since there was not much to do outside during the pandemic, my [roommate](https://adbforlife.github.io/)
and I ended up playing and solving CTF challenges every day after work, and the highlight
was playing DEF CON 28 together with [PPP](https://pwning.net/), where we came in second.

---

### Spring 2020

{: .first-course-item #course15410 }

- &#11088; 15-410 [**Operating Systems Design and Implementation**](https://www.cs.cmu.edu/~410/), [Dave Eckhardt](http://www.cs.cmu.edu/~davide/), [Brian Railing](http://www.cs.cmu.edu/~bpr/), [Dave O'Hallaron](http://www.cs.cmu.edu/~droh/)

  This is probably the most "famous" systems class at CMU.
  The [slides from the first
  lecture](https://www.cs.cmu.edu/~410/lectures/L01a_OS.pdf) will probably give
  you a good sense of how the class is like.

  A few pieces of advice:

  1. Start early and aim to finish 2-3 days before the deadline. It is hard to debug a complicated kernel under stress, and you will always find a few
     more bugs just when you thought you were done.
  2. Choose your partner wisely. Preferably they will be someone that
     you have worked with before. Avoid choosing your significant other
     as your partner - this class is very stressful and it will introduce
     many fault-lines into your relationship.

  I will probably write a dedicated, separate post for this class someday.

- 15-259 [**Probability and Computing**](https://www.cs.cmu.edu/~harchol/PnC/class.html)
  {: .course-item #course15259 }

  Mor is absolutely amazing at teaching and in maintaining student's interest
  during classes. She will give out chocolates in class if you answer a
  question, leading to very high in-class participation. People fight
  to get seated closer to the front, so that it is more likely that they
  will get called on to answer questions.

  Aside from that, her teaching is very clear and she has a wonderfully
  written textbook that accompanies the course. Homework problems
  are on the challenging side, but also very fun.

  If you are a CS major with a strong mathematical background, you should
  definitely take this class to satisfy your probability requirement instead of
  taking the one from the statistics department (which is not as rigorous). This
  course is tailored for CS students and covers many topics that are pervasive
  in CS research, such as concentration bounds including Chernoff and Hoeffding,
  classical randomized algorithms, and Markov Chains.

- 15-260 Statistics and Computing
  {: .course-item #course15260 }

  This class is offered as a mini, meant to accompany 15-259 Probability and
  Computing. It allows CS students to satisfy satistics requirement for some ML
  or graduate statistics classes without having to take another introductory
  statistic course.

- 15-451 [**Algorithm Design and Analysis**](http://www.cs.cmu.edu/~15451-s20/), [Danny Sleator](http://www.cs.cmu.edu/~sleator/), [David Woodruff](http://www.cs.cmu.edu/~dwoodruf/)
  {: .course-item #course15451 }

  This is the main algorithms class of CMU, and Danny Sleator and Splay Trees
  (which he invented) are the heart of it. Since students are expected to have
  already completed several other introductory algorithms and theory classes
  prior to taking this, the course covers many more advanced topics.

  One misconception that many people have is that they need to take 15-451
  before they know 'enough' algorithms to feel confident enough to apply for
  software engineering jobs and pass the interview. Most of the content covered
  in 15-451 are probably out of scope for most interviews (nobody is going to
  ask you to code max-flow, or prove LP duality, or write Rabin-Miller hashing
  from memory during a 30-minute interview). The only relevant topic for
  interview preparation is probably dynamic programming, but given that DP
  was already covered in 15-210, there is really no reason to wait for this
  class.

- 33-228 [**Electronics I**](http://www-meg.phys.cmu.edu/physics_33228/), [Jeffrey Peterson](https://www.cmu.edu/physics/people/faculty/peterson.html), [Ben Hunt](https://www.cmu.edu/physics/people/faculty/hunt.html), [Simranjeet Singh](https://www.cmu.edu/physics/people/faculty/singh.html)
  {: .course-item #course33228 }

  I took this class to satisfy my lab requirement. I personally thought
  that this is the best class for the lab requirement for CS majors, since it
  introduces you to the basics of circuits and electronics, which seems
  quite useful as there are no other CS classes that does anything comparable.
  One of the most memorable projects was in building a 555 timer from scratch
  (not using the IC) with resistors, transistors, capacitors, and voltage
  dividers. It made me realize that if debugging software was bad,
  debugging hardware is on a whole new level.

- 79-396 **Music and Society in 19th and 20th Century Europe and the U.S.**, [Naum Kats](https://www.cmu.edu/dietrich/modlang/about-us/filter/adjunct/naum-kats.html)
  {: .course-item #course79396 }

  This was a relatively low-stress and fun class, where each week you would
  watch a live play, musical, or orchestral concert, and also visit art
  galleries and museums. The weekly homework is just to write a one-page
  reflection of the experience, and apart from that there are only two
  book reviews to complete. It was quite a good class to take to destress
  from my other more demanding classes.

Units: 61

---

### Fall 2019

{: .first-course-item #course15319 }

- &#10084;&#65039; 15-319 **Cloud Computing**

  [This post by Lisa
  Hou](https://medium.com/@zhuojun.hou9270/how-is-cloud-computing-15619-15319-at-cmu-96893f5c2b0)
  discusses the projects of the course in detail.

  I personally enjoyed this class a lot and the amount of industry knowledge
  that I gained from the course was like having done 3 internships back-to-back,
  which happened to make it really great interview prep as well.

  I would recommend taking the class with friends as it is easy to get stuck in
  weird errors of a complex system that you just learned, and to go for the
  graduate 15-619 option if possible, as it includes a very transformative team
  project that will stretch your wits in finding ways of improving the
  throughput of a distributed web service in the cloud.

- 15-330 **Introduction to Computer Security**, [Bryan Parno](https://www.andrew.cmu.edu/user/bparno/)
  {: .course-item #course15330 }

  Bryan is a great lecturer, and the course is divided into 3 portions: binary
  exploitation, cryptography, and web security. I took this course because I
  wanted better knowledge of topics in computer security, so that I can get better at playing Capture-The-Flag (CTF) competitions with
  the [Plaid Parliament of Pwning (PPP)](https://pwning.net/).

  I personally think it is a form of gross negligence if you ever write critical
  software systems without having a good understanding of computer security,
  especially if you ever touch or deal with anything related to cryptography, or
  maintain critical and sensitive user-facing platforms like banking systems.
  The number of software engineers who still don't know the difference between
  private and public keys is stupefying, and you really don't want to cause your
  company to get hacked or lose customer data because of a lack of understanding
  of fundamental security concepts.

- 15-210 [**Parallel and Sequential Data Structures and Algorithms**](https://www.diderot.one/courses/44)
  {: .course-item #course15210 }

  I found the course okay, some portions were interesting but
  I don't remember being very excited by any topic in particular. I already
  knew quite a lot of the algorithms from doing competitive programming
  in high school, so it was relatively easy.

- 21-268 **Multi-dimensional Calculus**, [Jack Schaeffer](https://www.cmu.edu/math/people/faculty/schaeffer-j.html)
  {: .course-item #course21268 }

  This is the honors version of 21-259 Calculus in Three Dimensions, and I would
  recommend mathematically strong students take this (or 21-269 Vector
  Analysis if they are on the math honors track). It covers more content
  and is more rigorous.

- 69-151 **Introduction to Yoga**
  {: .course-item #course69151 }

  Took this for fun because I attended a lot of free yoga classes at the office while I was interning at Asana.

- 98-349 **StuCo: Design and Analysis of Logic Puzzle Games**
  {: .course-item #course98349 }

Units: 52

I dropped 21-373 Algebraic Structures in the last week of classes, as the
professor who was a visiting postdoc was really poor at teaching (i.e
forgetting proofs halfway, seems very unprepared for classes, homework and exam
questions drastically different from what was covered in class),
and I thought I was going to get a B in the class. In hindsight grades
don't really matter and I should have stuck through, but ensuring that you
learn under a good professor is also really important. The experience made me
avoid math classes for a while as I wanted to make sure any future math class I took
was taught by a good professor.

---

### Summer 2019

I spent a wonderful summer interning at [Asana](https://asana.com/) in San Francisco as a Software
Engineering intern on the Infrastructure Management Team. My mentor and I
became great friends, and we still regularly keep in touch many years on.

I ended up visiting many of the great national parks in California on weekend
trips during the internship, including
[Yosemite](https://www.nps.gov/yose/index.htm), [Lake
Tahoe](https://visitlaketahoe.com/), [Mt. Lassen](https://www.nps.gov/lavo/index.htm), and
[Muir Woods](https://www.nps.gov/muwo/index.htm). It was a fun time!

---

### Spring 2019

{: .first-course-item #course15251 }

- &#10084;&#65039; 15-251 [**Great Ideas in Theoretical Computer Science**](http://www.cs.cmu.edu/~15251/), [Venkatesan Guruswami](https://people.eecs.berkeley.edu/~venkatg/), [Vipul Goyal](https://www.cs.cmu.edu/~goyal/)

  The class was very well-taught and structured. Homeworks are done in the form
  of weekly writing sessions, where you have to reproduce proofs to selected
  homework problems independently in an exam-like setting. This helps to ensure
  that all students fully understand the proofs that they write up.

- 15-150 [**Principles of Functional Programming**](http://www.cs.cmu.edu/~15150/)
  {: .course-item #course15150 }

  Functions are values!

- 15-213 [**Introduction to Computer Systems**](https://www.cs.cmu.edu/~213/), [Seth Goldstein](http://www.cs.cmu.edu/~seth/), [Brandon Lucia](https://brandonlucia.com/)
  {: .course-item #course15213 }

  This class unlocks all other systems classes, and many classes also have this
  as a requirement to ensure that students have had experience implementing
  large projects. Much of what is covered will form the bread and butter of
  subsequent systems classes, and many of the topics will also be extremely
  useful in your future career as a software engineer.

- 21-122 **Integration and Approximation**, [Eric Baer](http://www.stat.uchicago.edu/~ebaer/)
  {: .course-item #course21122 }

  I liked his lectures, although Eric is now at UChicago.

- 21-241 **Matrices and Linear Transformations**, [Irina Gheorghiciuc
  ](https://www.cmu.edu/math/people/faculty/gheorghiciuc.html)
  {: .course-item #course21241 }

  I personally liked Irina's to-the-point, no-nonsense teaching style.
  Her lectures are quite clear and interesting.

- 98-317 [**StuCo: Type Theory (Hype for Types)**](https://hypefortypes.github.io/S19/), Jeanne Luning Prak, [Vijay Ramamurthy](https://vijayramamurthy.com/), [Chris Grossack](https://grossack.site/), [Charles Yuan](https://people.csail.mit.edu/chenhuiy/)
  {: .course-item #course98317 }

  We were given a crash course of SML in the first week, as everything else
  in the course depended on that, and it was required for the weekly homework. I
  would say that taking this concurrently with 15-150 definitely made 15-150
  much easier, but at the time I was also pretty lost during many of the
  lectures in this class. Somehow I was able to complete all the programming
  assignments despite understanding very little from the lecture, which goes to
  show how much a strongly typed functional language can inform and guide your
  code. It was only a few years later after having taken both 15-312
  Foundations of Programming Languages and 15-317 Constructive Logic that I
  could fully appreciate much of the content in hindsight.

Units: 57

---

### Fall 2018

{: .first-course-item #course15151 }

- &#10084;&#65039; 15-151 [**Mathematical Foundations for Computer Science**](https://www.math.cmu.edu/~jmackey/151_128/welcome.html), [John Mackey](https://www.cmu.edu/math/people/faculty/mackey.html)

  It is said that the best predictor of a student's success is how well
  they did in 15-151. Having been through the whole process,
  I can testify that it is indeed true. Much of computer science depends
  on math, and you need a strong and firm foundation in discrete
  mathematics and proof writing in order to succeed in all subsequent classes.

  Seriousness aside, Mackey is an amazing lecturer. In fact, I would go so far
  as to say his lectures are more like a performance, full of wit and humor. I
  will always remember how after the first lecture everyone stood up and gave
  him a standing ovation. I believe that was the first and only time I have ever
  seen 200 students in a packed lecture hall give a standing ovation after a
  lecture.

- 15-122 [**Principles of Imperative Computation**](https://www.cs.cmu.edu/~15122/)
  {: .course-item #course15122 }

  I was quite bored during lecture as I already knew most of the content
  due to my competitive programming background. The assignments were still pretty cool though.

- 21-120 **Differential and Integral Calculus**
  {: .course-item #course21120 }

  I could not transfer AP credit from A-Level H2 Mathematics as the SEAB course code
  changed while I was serving National Service as they modified the syllabus and
  split it into H2 Further Mathematics and H2 Mathematics, and was therefore
  forced to take the class by my first-year advisor. It was honestly largely a
  waste of time, but it was also the only class that I took at CMU where I was
  classmates with a lot of college football players.
  It was quite funny because I was the
  only Asian kid in the class, and since I already knew all the content, I would
  always go up to the blackboard and write up the answers to all the TA's
  questions during recitation, to the applause of all the other kids. The whole
  scene is quite stereotypical but it's also funny, and having this experience
  is perhaps the only silver lining of taking this class.

- 76-106 **Writing about Literature and Art**
  {: .course-item #course76106 }

  I enjoyed the assigned readings of literature from different cultures,
  and the class discussions were also quite engaging.

- 76-107 **Writing about Data**
  {: .course-item #course76107 }

  Found this class relatively dry, and most of the content seemed obvious and boring.

- 80-100 **Introduction to Philosophy**
  {: .course-item #course80100 }

  I had some prior interest and experience with philosophy (read a lot of
  philosophy books during my time in the army), and found this a
  really interesting class that covered a good amount of topics. However, the
  workload was quite high for a gen-ed (2 sets of assignments and readings per
  week), and I don't think I could have tolerated the workload if I had taken it
  beyond freshmen year.

- 07-131 **Great Practical Ideas in Computer Science**
  {: .course-item #course07131 }

  Teaches you the tools of the trade like Git, Vim, etc.
  You'll be able to work much more effectively and save a lot of time
  if you master these skills early.

- 15-051 Discrete Math Primer
  {: .course-item #course15051 }

  This is a course that CS majors take over the summer to gain some foundations
  in discrete math before starting college to prepare them for 15-151.

- 99-101 Computing at Carnegie Mellon
  {: .course-item #course99101 }

  This is another online course that can be taken over the summer before the
  semester, to familiarize yourself with CMU's computing resources and tools,
  and how to avoid getting into trouble such as torrenting movies on campus.

- 07-128 First Year Immigration Course
  {: .course-item #course07128 }

  A weekly session during the first semester where our first-year advisor
  [Jacobo Carrasquel](http://www.cs.cmu.edu/~jxc/home.html) and the
  Associate Dean for Undergraduates [Tom Cortina](https://www.cs.cmu.edu/~tcortina/)
  talks to us about the various things happening at CMU SCS, and the opportunities
  available to us. One of the most useful parts of this class is how they got
  professors from each of the 7 departments within the School of Computer Science
  to talk to us about their research areas, and the problems that they work on daily.
  It gives you a pretty good overview of the research that is being done across
  the broad, and where to look if you have specific interests.

Units: 55

Most of the last few courses are just orientation courses that are not real classes with actual workload.

---

## Some CMU-Specific Things

This FAQ is mostly geared towards people who are unfamiliar with CMU.

### How does the course numbering work?

From [a note regarding course numbers](http://coursecatalog.web.cmu.edu/previous/2017-2018/schoolofcomputerscience/courses/):

> Each Carnegie Mellon course number begins with a two-digit prefix which
> designates the department offering the course (76-xxx courses are offered by the
> Department of English, etc.). Although each department maintains its own course
> numbering practices, typically the first digit after the prefix indicates the
> class level: xx-1xx courses are freshmen-level, xx-2xx courses are sophomore
> level, etc. xx-6xx courses may be either undergraduate senior-level or
> graduate-level, depending on the department. xx-7xx courses and higher are
> graduate-level. Please consult the Schedule of Classes each semester for course
> offerings and for any necessary pre-requisites or co-requisites.

Here are some of the common ones that you will see as a CS major:

- 15-xxx: Computer Science
- 21-xxx: Mathematical Sciences
- 02-xxx: Computational Biology
- 05-xxx: Human-Computer Interaction
- 10-xxx: Machine Learning
- 11-xxx: Language Technologies Institute
- 16-xxx: Robotics
- 17-xxx: Institute for Software Research
- 18-xxx: Electrical & Computer Engineering
- 98-xxx: Student Led Courses

### Overload Policy

In the _School of Computer Science_, students have a unit cap of 54. Unit overload requests must be submitted to your academic advisor, contingent on past academic
performance.

First-year students are not permitted to overload.

### What are StuCos?

_Student Taught Courses_ (StuCos) are low-stress classes taught by other fellow students. They are pass/fail, and are always 3 units each. These are usually fun and interesting topics, such as lockpicking, stand-up comedy, esoteric programming languages, etc.

### Special Topics Courses

The Computer Science Department offers Special Topics in Theory courses,
numbered 15-859. The section numbers differentiate the specific course.
For instance, 15-859 CC is Algorithms for Big Data.

These are irregularly-offered courses targeted at people intending to pursue
research in the field.
