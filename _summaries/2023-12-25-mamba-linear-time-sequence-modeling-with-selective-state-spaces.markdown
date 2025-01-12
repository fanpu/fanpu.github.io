---
layout: summary
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
giscus_comments: true
bib_id: 2312.00752v1
published: false
---

### Three Important Things

I was really excited for this paper - there was quite a buzz about it
on X right after it was announced, and the authors were also of the
S4 paper fame which was a really promising step towards a new
sequence modeling architecture that could match the performance of Transformers,
which I [previously wrote about]({% link _summaries/2023-08-25-efficiently-modeling-long-sequences-with-structured-state-spaces.markdown %}).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Quadratic attention has been indispensable for information-dense modalities such as language... until now.<br><br>Announcing Mamba: a new SSM arch. that has linear-time scaling, ultra long context, and most importantly--outperforms Transformers everywhere we&#39;ve tried.<br><br>With <a href="https://twitter.com/tri_dao?ref_src=twsrc%5Etfw">@tri_dao</a> 1/ <a href="https://t.co/vXumZqJsdb">pic.twitter.com/vXumZqJsdb</a></p>&mdash; Albert Gu (@_albertgu) <a href="https://twitter.com/_albertgu/status/1731727672286294400?ref_src=twsrc%5Etfw">December 4, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

#### 1. The Secret Sauce: Selection Mechanism

From one perspective, Transformers and recurrent models are polar
opposites in terms of compression: Transformers do not compress
anything at all and consider everything in its context window, while RNNs
compress everything down into its hidden state that is passed on recurrently.
However, Transformers pay a price for this choice in terms of having slow
linear-time inference and requiring quadratic-time training.

To understand how state space models fare, let's first recap how they
work. We'll focus on [structured state space sequence (S4) models]({% link _summaries/2023-08-25-efficiently-modeling-long-sequences-with-structured-state-spaces.markdown %}).

S4 models are parameterized by $$\Delta$$, the discretization timestep,
and learnt matrices $$\bA, \bB, \bC$$.

As a system that evolves continuously with latent state $$h(t)$$, 
it can be formulated as the following differential equation:

$$
\begin{aligned}
 h^{\prime}(t)&=\boldsymbol{A} h(t)+\boldsymbol{B} x(t) \\
 y(t)&=\bC h(t)
\end{aligned}
$$

In practice, we discretize it with timestep $$\Delta$$. Taking this approach,
we obtain the following recurrence:

$$
\begin{aligned}
h_t &= \overline{\boldsymbol{A}} h_{t-1}+\overline{\bB} x_t \\
y_t &= \bC h_t
\end{aligned}
$$

If we unroll the recurrence, then we realize that we can actually
solve this using a convolution with kernel $$\overline{\bK}$$,
that gives us the response at each timestep:

$$
\begin{aligned}
 \overline{\boldsymbol{K}} &=\left(\boldsymbol{C} \overline{\boldsymbol{B}}, \boldsymbol{C} \overline{\boldsymbol{A B}}, \ldots, C\overline{\boldsymbol{A}}^k \overline{\boldsymbol{B}}, \ldots\right) \\
 y &= x * \overline{\mathbf{K}} \\
\end{aligned}
$$

The fact that this convolution can be efficiently computed when placing
some assumptions on the structure of $$\bA$$ was the focus of the S4 paper.

<!-- Recall that matrix exponentials are given by the following:

$$
\exp(\bX) = \sum_{k=0}^\infty \frac{1}{k!} \bX^k
$$ -->



<!-- State space models suffer from the same issues as recurrent models.
However, the authors  -->

{% include figure.liquid
    path="/assets/img/summaries/mamba_alg.webp"
    width="600px"
    class="z-depth-1"
%}




#### 2. Hardware-Aware Algorithm

Associative means that the grouping of operations
does not matter: $(A \times B) \times C = A \times (B \times C)$.

Prefix sums are referred to as scan in the functional programming/parallel
programming community.


 
#### 3. Baz

### Most Glaring Deficiency

### Conclusions for Future Work
