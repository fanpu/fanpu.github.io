---
title: "A Concise Proof of the Central Limit Theorem, and Its Actually Useful Version, the Berry-Esseen Theorem"
layout: post
tags: [math, machine-learning]
cover: maine.webp
cover_preview: maine.webp
caption: Acadia National Park, Maine, USA
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
    The Central Limit Theorem is widely used in statistics and machine learning,
    as it allows us to assume that given enough samples, the mean of the samples
    will follow a normal distribution. This holds even if the samples come
    from a distribution that is not normally distributed.
    In this post, we prove the Central Limit Theorem, and then take a look
    at the Berry-Esseen Theorem, which actually provides a quantitative bound
    on the convergence of the distribution and can therefore be actually used in
    deriving theoretical bounds.
---

The Central Limit Theorem is widely used in statistics and machine learning,
as it allows us to assume that given enough samples, the mean of the samples
will follow a normal distribution. This holds even if the samples come
from a distribution that is not normally distributed.
In this post, we prove the Central Limit Theorem, and then take a look
at the Berry-Esseen Theorem, which actually provides a quantitative bound
on the convergence of the distribution and can therefore be actually used in
deriving theoretical bounds.

### The Central Limit Theorem

The Central Limit states that the mean of an appropriately transformed random variable
converges in distribution to a standard normal. We first need to introduce the 
definition of convergence of probability distributions:


{% include theorem.md 
  type="definition"
  name="Convergence in Distribution"
  statement="
  Let \( F_{X_n} \) and \( F_{X} \) denotes the cumulative density functions (CDF) of 
  \( X_n \) and \( X \) respectively.

  A sequence \( X_n \) converges to \( X \) in distribution if
  $$ \lim_{n \to \infty } F_{X_n}(t) = F_X (t)$$
  
  for all points \( t \) where \( F_X \) is continuous.
  "
%}

Note that the requirement that it only holds for points of continuity is not superfluous, as there
can be distributions that converge but disagree in value at points of discontinuities
(i.e take $$ X_n = N(0, 1/n) $$ and $$ X $$ to be the point mass at 0, they converge but their CDF take different values at $$ t=0 $$).

The Central Limit Theorem can then be stated in the following form (there are many other equivalent statements):

{% include theorem.md 
  type="theorem"
  name="Central Limit Theorem"
  statement="
  Let \( X_1, X_2, \dots, X_n \) be a sequence of independent random variables with mean \( \mu \) and variance \( \sigma^2 \).
  Assume that the moment generating function \( \mathbb{E} \left[ \exp(t X_i) \right] \) is finite for \( t \) in a neighborhood around zero.
  Let \( \overline{X}_n = \frac{1}{n} \sum\limits_{i=1}^n X_i \). Let
  
  $$ Z_n = \frac{\sqrt{n} \left( \overline{X}_n - \mu \right)}{\sigma}. $$
  
  Then \( Z_n \) converges in distribution to \( Z \sim N(0, 1) \).
  "
%}

### Proof of the Central Limit Theorem

There are several ways of proving the Central Limit Theorem. 
The proof that we will explore today relies on the methods of moments.
An alternative measure-theoretic version of the proof relies on Lévy's
Continuity Theorem, and makes use of convolutions and Fourier transforms.

Our goal is to show that $$ Z_n $$ converges in distribution to $$ Z \sim N(0,
1) $$. To do so, we will show that all the moments of $$ Z_n $$ converges to
the respective moments of $$ Z $$. 

#### Moment Generating Functions
The moments of a random variable can be obtained from its moment-generating function (MGF),
defined as follows:

{% include theorem.md 
  type="definition"
  name="Moment Generating Function"
  statement="
  The moment generating function of a random variable \( X \) is given by
  
  $$ M_X(t) = \mathbb{E} \left[ e^{tX} \right].$$
  "
%}

It is called a moment generating function since the $$k$$th moment of $$X$$,
i.e $$\mathbb{E} \left[X^k \right] $$, can be obtained by taking the
$$k$$th derivative of its moment-generating function (MGF) at 0:

$$ \mathbb{E} \left[X^k \right]  = M^{(k)}(0). $$

This is not too hard to see by induction on the fact that
$$M_X^k(t) = \mathbb{E} \left[ X^k e^{tX} \right]$$. The base case is trivial. 
For the inductive case,

$$
\begin{align*}
    M_X^{(k)}(t) & = \frac{d^k}{dt^k} \mathbb{E} \left[ e^{tX} \right] \\ 
               & = \frac{d}{dt} \mathbb{E} \left[ X^{k-1} e^{tX} \right] & \text{(by IH)}\\
               & = \frac{d}{dt} \int f(x) x^{k-1} e^{tx} \; dx \\ 
               & = \int \frac{d}{dt} f(x) x^{k-1} e^{tx} \; dx \\
               & = \int f(x) x^{k} e^{tx} \; dx \\
               & = \mathbb{E} \left[ X^{k} e^{tX} \right].
\end{align*}
$$

Substituting $$t=0$$ gives us the desired result.


#### Normal Distribution is Determined by its Moments
Distributions are determined uniquely by its moments under certain conditions. This is made precise
in the following theorem:

{% include theorem.md 
  type="theorem"
  name="Sufficient Condition for Distribution to be Determined by Moments"
  statement="
  Let \( s_0 > 0 \), and let \( X \) be a random variable with moment generating
  function \( M_X(s) \) which is finite for \( |s| < s_0 \). Then \( f_X \)
  is determined by its moments (and also by \( M_X(s)\)).
  "
%}

In words, it means that for some open interval around 0 we have that all moments are finite,
then the moments determine the distribution. This is true for the normal distribution,
where it can be shown that the following recurrence holds for the $$k$$th moment:

$$M^k(t) = \mu M^{k-1}(t) + (k-1) \sigma^2 M^{k-2}(t).$$

This is also not hard to show by induction, and the proof is omitted for brevity. Since the
first two moments of the standard normal distribution are 1 and 0 respectively which are both finite,
and our mean and standard deviation are both finite, then all our moments generated by the
recurrence must also be finite. So our standard normal is determined by its moments.

#### Method of Moments
Now cue the theorem that ties things together:

{% include theorem.md 
  type="theorem"
  name="Method of Moments"
  statement="
  Suppose that \( X \) is determined by its moments. Let \( X_n \) be a sequence of
  distributions, such that \( \int f_{X_n}(x) x^k \; dx \) is finite for all \( n, k \in \N \),
  and such that \( \lim_{n \to \infty} \int f_{X_n}(x) x^k \; dx = \int f_{X}(x) x^k \; dx \)
  for each \( k \in \N \). Then \( X_n \) converges in distribution to \( X \).
  "
%}

In words, it states that if the $$k$$th moment of $$X_n$$ is finite and converges to the $$k$$th moment
of $$X$$ in the limit of $$n$$, then $$X_n$$ converges to $$X$$.

This is great, since now we just have to show that all the moments of 
$$Z_n = \frac{\sqrt{n} \left( \overline{X}_n - \mu \right)}{\sigma}$$ converges to
the moments of the standard normal $$Z$$.

### Moment Generating Function of $$Z$$
Let's first find the moment generating function of $$Z$$:

$$
\begin{align*}
    M_{Z} & = \mathbb{E} \left[ e^{tZ} \right]                                                                                                \\
          & = \int f_Z(x) e^{tx} \; dx                                                                                                \\
          & = \int \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}x^2} e^{tx} \; dx                 & \text{(subst. pdf of standard Gaussian)} \\
          & = \int \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}x^2 + tx} \; dx                                                              \\
          & = \int \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}(x - t)^2 + \frac{1}{2}t^2} \; dx & \text{(completing the square)}           \\
          & = e^{\frac{1}{2}t^2} \int \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}(x - t)^2 } \; dx & \text{($e^{\frac{1}{2}t^2}$ does not depend on $x$)}           \\
          & = e^{\frac{1}{2}t^2} \cdot  1 \\
          & = e^{\frac{1}{2}t^2},
\end{align*}
$$

where the second last step comes from the fact that
$$\frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}(x - t)^2 }$$ is a probability distribution of a Gaussian with mean $$t$$ and variance 1, 
and therefore the integral integrates to 1.


### Moment Generating Function of $$Z_n$$

Now we find the moment generating function of $$Z_n$$. 
To simplify notation, define
$$ A_i = \frac{X_i - \mu}{\sigma}$$,
and see that we can write $$Z_n = \frac{1}{\sqrt{n}} \sum\limits_{i=1}^n A_i$$, since

$$
\begin{align*}
    \frac{1}{\sqrt{n}} \sum\limits_{i=1}^n A_i
    &= \frac{1}{\sqrt{n}} \sum\limits_{i=1}^n \frac{X_i - \mu}{\sigma} \\
    &= \sqrt{n} \sum\limits_{i=1}^n \frac{X_i - \mu}{ n \sigma} \\
    &= \sqrt{n} \frac{\overline{X}_n - \mu}{ \sigma} \\
    &= Z_n.
\end{align*}
$$

See that $$\mathbb{E}[A_i] = 0$$, and $$\mathbf{Var}(A_i) = 1$$.

Then starting from the definition of the moment generating function of $$Z_n$$,

$$
\begin{align*}
    M_{Z_n}(t) & = \mathbb{E} \left[ e^{t Z_n} \right]                                                                                                   \\
               & = \mathbb{E} \left[ \exp\left(t \frac{1}{\sqrt{n}} \sum\limits_{i=1}^n A_i \right) \right] & \text{(by equivalent definition of $Z_n$)} \\
               & = \prod_{i=1}^n \mathbb{E} \left[ \exp\left( \frac{t}{\sqrt{n}} A_i \right) \right]        & \text{(by independence of $A_i$'s)}        \\
               & = \prod_{i=1}^n M_{A_i}(t/\sqrt{n})                                                & \text{(definition of $M_{A_i}$)}           \\
               & = M_{A_i}(t/\sqrt{n} )^n.
\end{align*}
$$

Let's analyze each individual term $$M_{A_i}(t / \sqrt{n})$$ by performing a Taylor expansion around 0.
Recall that the Taylor expansion of a function $$f(x)$$ about a point $$a$$ is
given by 
$$ f(x)= \sum\limits_{n=0}^\infty \frac{f^{(n)(a)}}{n!}(x-a)^n.$$. We will expand up to the
second order term, which requires us to find the first three moments of the MGF.

These are:

$$
\begin{align*}
    M_{A_i}(0)                 & = \mathbb{E} \left[ e^{t A_i} \right] \Big|_{t=0}                                                                                                                 \\
                               & = \mathbb{E} \left[ 1 \right]                                                                                                                                     \\
                               & = 1,                                                                                                                                                      \\
    M_{A_i}^\prime(0)          & = \mathbb{E} \left[ A_i \right]                                                       & \text{(by the $k$th moment property proved previously)}                   \\
                               & = 0,                                                                                                                                                      \\
    M_{A_i}^{\prime \prime}(0) & = \mathbb{E} \left[ A_i^2 \right]                                                     & \text{(by the $k$th moment property proved previously)}                   \\
                               & = \mathbb{E} \left[ A_i^2 \right] - \mathbb{E} \left[ A_i \right]^2 + \mathbb{E} \left[ A_i \right]^2                                                                             \\
                               & = \mathbf{Var}(A_i) + \mathbb{E} \left[ A_i \right]^2                                         & \text{($\mathbf{Var}(A_i) = \mathbb{E} \left[ A_i^2 \right] - \mathbb{E} \left[ A_i \right]^2 $)} \\
                               & = 1 + 0 \\
                               & = 1.
\end{align*}
$$

Taking all terms up to the second order Taylor expansion allows us to approximate $$M_{A_i}$$ as

$$
\begin{align*}
    M_{A_i}(t/\sqrt{n}) & \approx M_{A_i}(0) + M_{A_i}^\prime(0) + M_{A_i}^{\prime \prime}(0) \frac{t^2}{2n} \\
                        & = 1 + 0 + \frac{t^2}{2n}                                                           \\
                        & = 1 + \frac{t^2}{2n}.
\end{align*}
$$

Then now we can write the limit of the MGF of $$Z_n$$ as the following:

$$
\begin{align*}
    M_{Z_n}(t) & = M_{A_i}(t/\sqrt{n})^n       \\
               & \approx \left( 1 + \frac{t^2}{2n}  \right)^n \\
               & \to e^{t^2/2}, & \text{(by identity $\lim_{n \to \infty} (1 + x/n)^n \to e^x$)}
\end{align*}
$$

which shows that it converges to the MGF of $$Z$$, as desired. Hooray!

### An Uncomfortable Feeling
However, there is one thing in this proof that might have bothered you.
Our result came from making use of the Taylor approximation and taking limits, 
but there is no bound on how large $$n$$ must be for the distributions to converge
up to a maximum amount of error. This makes it unsuitable for much theoretical analysis,
since usually we would like to know that $$n$$ does not have to be too large
for us to obtain a sufficiently good approximation to the standard normal.

### The Useful Form of the Central Limit Theorem: The Berry-Esseen Theorem

The Berry-Esseen theorem solves this limitation by also providing explicit error bounds. 
This was proved independently by Andrew Berry and Carl-Gustav Esseen in the 40s,
and the statement goes as follows:

{% include theorem.md 
  type="theorem"
  name="Berry-Esseen"
  statement="
    Let \( X_1, \dots, X_n \) be independent random variables.

    Assume \( \mathbb{E} [X_i] = 0 \; \forall i \).

    Write \( \sigma_i^2 = \mathbf{Var} [ X_i] = \mathbb{E}[X_i^2] - \mathbb{E}[X_i]^2 = \mathbb{E}[X_i^2] \).

    Assume \( \sum\limits_{i=1}^n \sigma_i^2 = 1 \). 

    Let \( S = \sum\limits_{i=1}^n X_i \). Then \( \forall u \in \mathbb{R}\),
    
    $$
        \lvert \Pr \left[ S \leq u \right] - \Pr \left[ Z \leq u \right] \rvert
        \leq \mbox{const} \cdot \beta,
    $$

    where the exact constant depends on the proof, with the best known constant
    being \(.5600\) proven by Shevtsova in 2010, and 
    \(\beta = \sum\limits_{i=1}^n \mathbb{E} \left[ \lvert X_i \rvert^3 \right]\).

  "
%}

In words, the theorem says that the difference between the CDF of the sum of
the mean-0 random variables and the CDF of the standard normal is bounded by a
value proportionate to the third moment. This then becomes useful as a tool in
proving high probability statements if we can show that the third moment is
inversely polynomially small, i.e $$\beta = 1/\text{poly}(n)$$.

Another thing to note is that the theorem only provides an absolute bound for all values of $$u$$.
Therefore, when $$u$$ is very negative and $$\Pr [Z \leq u ] = \Phi(u)$$ is very small, the
relative error is actually very large, and therefore is not as helpful.

I hope this article has been helpful!

*I would like to express my thanks to my friend [Albert Gao](https://adbforlife.github.io/)
for reviewing this article and for providing valuable suggestions*.

### References
- Rosenthal, J. S. (2016). A first look at rigorous probability theory. World Scientific. 
- Larry Wasserman, CMU 36-705 Intermediate Statistics Lecture Notes. URL: [https://www.stat.cmu.edu/~larry/=stat705/](https://www.stat.cmu.edu/~larry/=stat705/)
- Ryan O'Donnell, CMU 15-751 A Theorist's Toolkit. URL: [https://www.youtube.com/watch?v=Ig5TuZauhW4](https://www.youtube.com/watch?v=Ig5TuZauhW4)