---
title: "A Friendly Tutorial to Variational Inference: Introduction and "
layout: post
tags: [code, general]
cover: ubin_mangrove.webp
cover_preview: ubin_mangrove.webp
caption: Chek Jawa Wetlands, Pulau Ubin, Singapore
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
published: false
---

In Bayesian inference, a common task is to approximate the posterior of a
distribution, given some observations. This can be framed as trying to learn
about the latent variable of the distribution, where latent variables are
variables that cannot be directly observed, but can be inferred indirectly from
observation of other quantities. 

For example, we might notice some symptoms in a patient, and try to deduce what
the underlying disease could be. Then our prior density $$p(\mbox{Disease})$$
is what we think the probability of each disease is, and our posterior
distribution after observing the symptoms is $$p(\mbox{Disease} \mid
\mbox{Symptoms})$$. 

It is easy to sample from the joint density of latent variables and observations in Bayesian networks
by an application of the definition of conditional probability:

$$ p(\mbox{Disease}, \mbox{Symptoms}) = \underbrace{p(\mbox{Disease})}_{\text{prior}} \cdot \underbrace{p(\mbox{Symptoms} \mid \mbox{Disease})}_{\text{likelihood based on model}} $$

On the other hand, computing the posterior is hard. By Baye's rule,

$$
\begin{align*}
    p(\mbox{Disease} \mid \mbox{Symptoms})
     & = \frac{p(\mbox{Disease}, \mbox{Symptoms})}{p(\mbox{Symptoms})} \\
     & = \frac{p(\mbox{Disease}, \mbox{Symptoms})}{\sum_{\mbox{Disease}^\prime \in \mbox{Diseases}}p(\mbox{Disease}^\prime, \mbox{Symptoms})},
\end{align*}
$$

where in the second line we had to marginalize over all possible diseases. The summation
term is called the partition function, and also referred to as the evidence. Computing
the partition function exactly is often computationally intractable as there can be an
exponential number of configurations. 

For instance, consider an Ising model, where we have $$n$$ nodes $$\bx = x_1, \dots,
x_n$$, and each node takes on a binary value $$x_i \in \left\{ \pm 1
\right\}$$. The model is parameterized by $$\theta$$, with $$\theta_{ij}$$
denoting the strength of interactions between nodes $$i$$ and $$j$$,
and $$\theta_i$$ denoting the self-relationship for node $$i$$. The
joint probability is given by

$$
\begin{align*}
    \frac{1}{\mathcal{Z}(\theta)}\exp \left( \sum_{\substack{i, j \in [n] \\ i \neq j}} x_i x_j \theta_{ij} + \sum_{i \in [n]} x_i \theta_i \right),
\end{align*}
$$

with the partition function $$\mathcal{Z}(\theta)$$ that ensures that the probability
distribution sums to 1 given by 

$$
\begin{align*}
    \mathcal{Z}(\theta) = \sum_{\bx \in \left\{ \pm 1 \right\}^n} p(\theta, \bx).
\end{align*}
$$

Computing $$\mathcal{Z}(\theta)$$ is therefore the main problem here since 
the number of configurations of $$\bx$$ is exponential in $$n$$.
Indeed, computing partition functions is proven to be
$$\#\mathsf{P}$$-hard in general (this is stricly harder than being $$\mathsf{NP}$$-hard, which to our
current knowledge only has exponential time algorithms).

Therefore, several families of methods have been developed to try to approximate
the posterior distribution in a computationally feasible manner. In this series
of posts, we will discuss using variational inference techniques, which reduces
it to an optimization problem. Another family of techniques
is called Markov chain Monte Carlo (MCMC), which makes use of Markov chains to
sample from a stationary distribution to approximate the posterior.

### Variational Inference
The main idea of variational inference is to estimate the partition function
by minimizing the distance between our distribution $$P$$
and some easier to compute distribution $$Q$$, by modifying the 

The name variational inference comes from the calculus of variations, which uses
small perturbations to find maxima/minimas of functionals.

{% include theorem.md 
  type="theorem"
  name="Gibbs Variational Principle"
  statement="
    We say that an (undirected) graph \(H\) is a minor of \(G\) if \(H\) can be
    obtained from \(G\) by deleting edges and vertices, and edge contractions.
  "
%}






Consider a latent-variable model, which means 

In this post, we will see how we can apply variational methods to perform inference
and learning in latent variable models.