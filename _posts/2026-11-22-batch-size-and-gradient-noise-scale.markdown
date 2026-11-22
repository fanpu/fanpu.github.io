---
title: "Understanding How Batch Size Relates to Gradient Noise Scale"
published: false
layout: distill
tags: [machine-learning]
cover: olancha_milky_way.webp
cover_preview: olancha_milky_way.webp
caption: Olancha, California, USA
class: post-template
author: fanpu
giscus_comments: true
description: "For a tuned learning rate, the best batch size from a goodput perspective depends on how noisy the minibatch gradients are. If the batch size is small relative to the gradient noise, the standard error of the gradient estimate is larger than the gradient itself, so the learning rate has to be small and convergence takes many steps. On the other hand, if the batch size is already large relative to the gradient noise, FLOPs and datapoints are wasted as the gradient estimates are already accurate. In this post, we derive the optimal learning rate for SGD with minibatch noise, and the gradient noise scale that separates these two regimes."
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
toc: true
---

In SGD, how should we set the learning rate and the batch size? The answer
depends on the gradient noise scale $$\mathcal{B}_{\text{noise}}$$. These notes
largely come from the paper
[An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162)
by McCandlish et al. They derive the best step size for a noisy minibatch gradient,

$$
\epsilon_{\text{opt}} = \frac{\epsilon_{\max}}{1 + \mathcal{B}_{\text{noise}} / B},
$$

where $$B$$ is the batch size and $$\epsilon_{\max}$$ is the best step size for the
true gradient.

## The minibatch gradient is a noisy estimate

In SGD we estimate the gradient from a minibatch:

$$
G_{\text{est}}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta L_{x_i}(\theta),
\qquad x_i \sim p.
$$

Here $$G$$ is the gradient, $$B$$ is the batch size, $$L_x(\theta)$$ is the loss
function on example $$x$$, and $$p$$ is the data distribution.

The expectation of the estimated gradient is its true value:

$$
\mathbb{E}_{x_{1 \ldots B} \sim p}\left[G_{\text{est}}(\theta)\right] = G(\theta).
$$

Its variance scales inversely with the batch size:

$$
\operatorname{cov}_{x_{1 \ldots B} \sim p}\left(G_{\text{est}}(\theta)\right)
= \frac{1}{B} \Sigma(\theta).
$$

We use the covariance here since gradients are vectors. $$\Sigma(\theta)$$ is the
covariance of the gradient of a single example, given by

$$
\begin{aligned}
\Sigma(\theta)
&\equiv \mathbb{E}_x\Big[
  \big(\nabla_\theta L_x(\theta) - G(\theta)\big) \\
&\qquad\quad \big(\nabla_\theta L_x(\theta) - G(\theta)\big)^T
\Big] \\
&= \mathbb{E}_x\left[\nabla_\theta L_x(\theta) \nabla_\theta L_x(\theta)^T\right] \\
&\qquad - 2\, G(\theta) G(\theta)^T + G(\theta) G(\theta)^T \\
&= \mathbb{E}_x\left[\nabla_\theta L_x(\theta) \nabla_\theta L_x(\theta)^T\right]
  - G(\theta) G(\theta)^T.
\end{aligned}
$$

So the minibatch gradient gives noisy estimates of the gradient, and larger
batch sizes give better estimates.

## The best step size with the true gradient

Let's consider this with a second order curvature analysis of the loss. Let $$G$$
be the true gradient at $$\theta$$ and $$H$$ be the Hessian at $$\theta$$. Perturbing
the parameters along some vector $$V$$ with step size $$\epsilon$$ gives the second
order expansion

$$
L(\theta - \epsilon V) \approx L(\theta) - \epsilon\, G^T V
+ \frac{1}{2} \epsilon^2\, V^T H V.
$$

What is the minimizer for $$\epsilon$$ if we had the true gradient $$G$$? Set $$V = G$$ and differentiate with respect to $$\epsilon$$:

$$
\begin{aligned}
\frac{d}{d\epsilon} L(\theta - \epsilon V)
&= -G^T V + \epsilon\, V^T H V \\
&= -G^T G + \epsilon\, G^T H G = 0,
\end{aligned}
$$

$$
\implies \epsilon = \frac{G^T G}{G^T H G} = \frac{\lVert G \rVert^2}{G^T H G}.
$$

What does this mean geometrically? $$H$$ holds the derivative of the gradient $$G$$, so $$HG$$ gives the change in the gradient as you move along $$G$$. Then $$G^T (HG)$$ is the projection of this change back onto $$G$$, which is the rate at
which the gradient changes as we move along $$G$$.

So

$$
\epsilon_{\max} = \frac{\lVert G \rVert^2}{G^T H G}
$$

can be thought of as saying that the best step size is the current slope
divided by the rate of change of the slope along $$G$$.

## The best step size with a noisy gradient

In practice we don't have the true gradient, so we have to minimize the
expected loss after a step along $$G_{\text{est}}$$:

$$
\begin{aligned}
\mathbb{E}\left[L(\theta - \epsilon\, G_{\text{est}})\right]
&= \mathbb{E}\Big[
  L(\theta) - \epsilon\, G^T G_{\text{est}} \\
&\qquad\quad + \frac{1}{2} \epsilon^2\, G_{\text{est}}^T H G_{\text{est}}
\Big] \\
&= L(\theta) - \epsilon \lVert G \rVert^2 \\
&\qquad + \frac{1}{2} \epsilon^2\, \mathbb{E}\left[G_{\text{est}}^T H G_{\text{est}}\right].
\end{aligned}
$$

The middle term uses $$\mathbb{E}[G_{\text{est}}] = G$$.

We model $$G_{\text{est}} = G + \xi$$. The noise $$\xi$$ has zero mean, $$\mathbb{E}[\xi] = 0$$, and the covariance in the gradient estimates comes from
the noise:

$$
\operatorname{cov}(\xi) = \operatorname{cov}(G_{\text{est}}) = \frac{\Sigma}{B}.
$$

Then

$$
\begin{aligned}
\mathbb{E}\left[G_{\text{est}}^T H G_{\text{est}}\right]
&= \mathbb{E}\left[(G + \xi)^T H (G + \xi)\right] \\
&= \mathbb{E}\big[G^T H G + G^T H \xi \\
&\qquad\quad + \xi^T H G + \xi^T H \xi\big] \\
&= G^T H G + \underbrace{\mathbb{E}\left[2\, G^T H \xi\right]}_{=\,0} \\
&\qquad + \mathbb{E}\left[\xi^T H \xi\right].
\end{aligned}
$$

Let's analyze the last term:

$$
\mathbb{E}\left[\xi^T H \xi\right]
= \sum_{i,j} H_{ij}\, \mathbb{E}\left[\xi_i \xi_j\right]
= \sum_{i,j} H_{ij} \frac{\Sigma_{ij}}{B},
$$

by the definition of covariance. Next we use the property that

$$
\operatorname{tr}(AB) = \sum_i \sum_j A_{ij} B_{ji}.
$$

To prove the property, we know that $$(AB)_{ik} = \sum_j A_{ij} B_{jk}$$. By the
definition of the trace,

$$
\operatorname{tr}(AB) = \sum_i (AB)_{ii} = \sum_i \sum_j A_{ij} B_{ji}.
$$

Since $$\Sigma$$ is symmetric, $$\Sigma_{ij} = \Sigma_{ji}$$, so

$$
\mathbb{E}\left[\xi^T H \xi\right] = \frac{\operatorname{tr}(H \Sigma)}{B}.
$$

Putting everything together,

$$
\begin{aligned}
\mathbb{E}\left[L(\theta - \epsilon\, G_{\text{est}})\right]
&= L(\theta) - \epsilon \lVert G \rVert^2 \\
&\quad + \frac{1}{2} \epsilon^2 \left(G^T H G + \frac{\operatorname{tr}(H \Sigma)}{B}\right).
\end{aligned}
$$

To find the optimal step size $$\epsilon_{\text{opt}}$$, we take the derivative
with respect to $$\epsilon$$ and set it to zero:

$$
-\lVert G \rVert^2
+ \epsilon \left(G^T H G + \frac{\operatorname{tr}(H \Sigma)}{B}\right) = 0,
$$

$$
\epsilon_{\text{opt}}
= \frac{\lVert G \rVert^2}{G^T H G + \dfrac{\operatorname{tr}(H \Sigma)}{B}}
= \frac{\dfrac{\lVert G \rVert^2}{G^T H G}}
       {1 + \dfrac{\operatorname{tr}(H \Sigma)}{B\, G^T H G}}.
$$

The numerator is $$\epsilon_{\max}$$, our solution for the noiseless gradient
from before. Defining the noise scale as

$$
\mathcal{B}_{\text{noise}} = \frac{\operatorname{tr}(H \Sigma)}{G^T H G},
$$

this can be rewritten as

$$
\epsilon_{\text{opt}} = \frac{\epsilon_{\max}}{1 + \dfrac{\mathcal{B}_{\text{noise}}}{B}}.
$$

### Observations

- With noise, $$\epsilon_{\text{opt}} < \epsilon_{\max}$$, so we want to take
  smaller steps when the gradient is noisy.
- If the batch size is much smaller than the noise scale
  ($$B \ll \mathcal{B}_{\text{noise}}$$), then the $$\mathcal{B}_{\text{noise}} / B$$ term dominates the denominator, so
  increasing the batch size helps speed up convergence linearly.
- If $$B \gg \mathcal{B}_{\text{noise}}$$, the denominator is dominated by its
  first term, the $$1$$, so increasing the batch size has little effect on
  speeding up training.

## Why is this a reasonable definition of the noise scale?

The noise scale is defined as

$$
\mathcal{B}_{\text{noise}} = \frac{\operatorname{tr}(H \Sigma)}{G^T H G}.
$$

Why is this a reasonable definition? Let's consider the simpler 1D setting,
where $$H = h$$, $$\Sigma = \sigma^2$$ and $$G = g$$ are scalars. Then

$$
\mathcal{B}_{\text{noise}} = \frac{h \sigma^2}{h g^2} = \frac{\sigma^2}{g^2}
= \left(\frac{\sigma}{g}\right)^2.
$$

If $$\sigma$$ is the standard deviation of a single example's gradient, the
standard error of averaging $$B$$ samples is $$\frac{\sigma}{\sqrt{B}}$$. If we
want this error to be as large as the gradient itself, we set $$\frac{\sigma}{\sqrt{B}} = \lvert g \rvert$$, which rearranges to give precisely

$$
B = \frac{\sigma^2}{g^2}.
$$

So the takeaway is that $$\mathcal{B}_{\text{noise}}$$ is where the error bar on
the gradient is as large as the gradient itself. If the batch size is too small
relative to the noise scale, the error bars are too big and the batch gradient
is mostly sampling error. On the other hand, if the batch size already exceeds $$\mathcal{B}_{\text{noise}}$$, you are just slightly polishing the gradient and
wasting compute.
