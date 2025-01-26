---
title: "NTK"
layout: post
tags: [machine-learning, math]
cover: furano.webp
cover_preview: furano.webp
caption: Lavender Fields in Biei, Kamikawa Subprefecture, Hokkaido, Japan
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  TODO
---

Outline:

Suppose we have a collection of input-output pairs $(x_1, y_1), \cdots, (x_n, y_n) \in \mathcal{X} \times \mathcal{Y}$, and we would like to learn relationships between them so that given a test point $x$, we can predict its corresponding label $y$.

If the data is linearly related, then this becomes standard least-squares linear regression:

{% include figure.liquid
    path="/assets/img/posts/kernel-methods/linear-regression.webp"
    width="500px"
    class="z-depth-1"
    num=1
    caption="
        Example of a linear regression best-fit solution. Figure from <a href='https://www.cambridge.org/core/books/kernel-methods-for-pattern-analysis/811462F4D6CD6A536A05127319A8935A'>Kernel Methods for Pattern Analysis</a>.
    "
%}

What if the data is non-linear? Then it turns out that we can still use our
algorithmic toolbox for optimizing linear functions
using what are called _kernel methods_.

Before we get into kernels, let's first understand what feature mappings are.
Consider the following classification problem between data that is initially not linearly
separable, but becomes so after applying a feature mapping $\phi$:

{% include figure.liquid
    path="/assets/img/posts/kernel-methods/linear-regression-with-feature-mapping.webp"
    width="500px"
    class="z-depth-1"
    num=1
    caption="
        Non-linear data that becomes linearly separable after applying a feature map $\phi$ to each input. Figure from <a href='https://www.cambridge.org/core/books/kernel-methods-for-pattern-analysis/811462F4D6CD6A536A05127319A8935A'>Kernel Methods for Pattern Analysis</a>.
    "
%}

$\phi$ could for instance map the input into a polynomial. 

{% include theorem.md 
  type="definition"
  name="Kernel Function"
  statement="
  A kernel is a function $\kappa$ that for all $x, x' \in \mathcal{X}$ satisfies:

  $$\kappa(x, x') = \langle \phi(x), \phi(x') \rangle$$

  where $\phi$ is a mapping from $\mathcal{X}$ to an (inner product) feature space $F$

  $$\phi : x \mapsto \phi(x) \in F$$
  "
%}

The beauty of kernels is that there are many applications where one can compute 
$\kappa(x, x')$ efficiently, even though $\phi(x)$ itself
could be a vector with an exponential or infinite number of dimensions.


In the regression setting, 
