---
title: "Understanding Positional Embeddings: From Sinusoidal Embeddings to RoPE"
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
published: false
---

Positional embeddings enable the Transformer architecture to perform well on
sequential data, as the self-attention mechanism itself (and all other
components of the Transformer other than causal masking) are position-agnostic.

In this post, we'll learn why positional embeddings are important, how the
original positional embeddings from the [Attention Is All You
Need](https://arxiv.org/abs/1706.03762) paper works and intuition
behind how it was designed, and finally derive a state-of-the-art
technique called rotary position embeddings (RoPE) which performs better
and is used in almost all open-source large language models today (and believed
to be used in proprietary models as well).

{% include figure.liquid
    path="/assets/img/posts/positional_embeddings/rope_models.webp"
    width="500px"
    class="z-depth-1"
    num=1
    caption="
        Open source models predominantly use RoPE for positional embeddings.
        Image from <a href='https://arxiv.org/abs/2402.00838'>OLMo: Accelerating the Science of Language Models</a>.
    "
%}

## Attention and Self-Attention

To understand why positional embeddings are important, let's review how the
attention mechanism works.

There are three components to the attention mechanism: queries, keys, and values.
Keys and values come together in pairs, and queries and keys are comparable
against each other for some similarity measure.

This terminology is inspired by search engines, where given a particular query $$q$$,
you compare it against all keys $$k_1, \cdots, k_m$$ for some similarity
function $$\alpha(q, k_i)$$, and then weigh each of the corresponding values
$$v_1, \cdots, v_m$$ based on how similar each corresponding key was to the
query, i.e $$\sum_{i=1}^m \alpha (q, k_i) v_i$$.

In practice, the choice of similarity function is the dot product (otherwise known as dot-product attention),
scaled by $$1/\sqrt{d}$$ where $$d$$ is the dimension of $$q$$ and the $$k_i$$'s to reduce variance of the results.
A softmax is then applied to make the weights for each $$v_i$$ sum to 1.

If we block this with queries $$q_1, \cdots, q_n$$ with $$\mathbf{Q}$$,
and similarly $$\mathbf{K}, \mathbf{V}$$ for the keys and values,
this gives that the attention is given by

$$
\textrm{Attention}(\mathbf{Q}, \mathbf{K}) = \sum_{i=1}^m \frac{\exp \left( \frac{q^T k_i}{\sqrt{d}} \right)}
{\sum_{j=1}^m \exp \left( \frac{q^T k_j}{\sqrt{d}} \right) } v_i = \textrm{softmax} \left( \frac{\mathbf{QK}^T}{\sqrt{d}} \right) \mathbf{V}
$$

As you can see, the formulation of the attention mechanism is agnostic to the ordering of the queries, keys, and values.
In other words, you can use any permutation of the indices of the keys and
values, and you will still obtain the same result for each query.

However, positional information is clearly very important in text.

## How Positional Encoding in "Attention Is All You Need" Works

Position embedding is a way to give the Transformer information about
the ordering of each token in the input sequence, since the attention
mechanism itself is order-agnostic.

To get the query, key, and values for the embeddings $$\bsx_m$$ of input tokens,
where we are considering the $$m$$th query and $$n$$th keys and values,
abstractly we can model self-attention as performing:

$$\begin{aligned} \boldsymbol{q}_m & =f_q\left(\boldsymbol{x}_m, m\right) \\ \boldsymbol{k}_n & =f_k\left(\boldsymbol{x}_n, n\right) \\ \boldsymbol{v}_n & =f_v\left(\boldsymbol{x}_n, n\right).\end{aligned}$$

For instance, in the original Attention Is All You Need paper,
this is given as

$$f_{t: t \in\{q, k, v\}}\left(\boldsymbol{x}_i, i\right):=\boldsymbol{W}_{t: t \in\{q, k, v\}}\left(\boldsymbol{x}_i+\boldsymbol{p}_i\right),$$

where the position encoding is added onto the embeddings and multiplied by the respective query/key/value matrices, with
the $$2t$$th and $$2t+1$$th element of the $$d-$$dimensional vector
$$\bsp_i$$ being given in a sinusoidal form

$$\begin{cases}\boldsymbol{p}_{i, 2 t} & =\sin \left(\frac{i }{ 10000^{2 t / d}}\right) \\ \boldsymbol{p}_{i, 2 t+1} & =\cos \left(\frac{i}{ 10000^{2 t / d}}\right)\end{cases}.$$

We'll go in deeply to understand this formulation in particular.

The intuition they had for formulating it using sinusoids is due to the fact that
a positional encoding can be transformed from one offset into another offset
using only linear operations. Let's see this in action.

To get from $$\bsp_{i, 2t}, \bsp_{i, 2t+1}$$ to $$\bsp_{i+k, 2t},
\bsp_{i+k, 2t+1}$$ respectively, we can write

$$
a = \frac{i }{ 10000^{2 t / d}}, \qquad
b = \frac{k }{ 10000^{2 t / d}}.
$$

Then

$$
\begin{align*}
    \bsp_{i, 2t} &= \sin \left( a \right) \\
    \bsp_{i, 2t+1} &= \cos \left( a \right) \\
    \bsp_{i+k, 2t} &= \sin \left( a+b \right) \\
    \bsp_{i+k, 2t+1} &= \cos \left( a+b \right). \\
\end{align*}
$$

Recall the identities

$$\begin{aligned} & \sin (a+b)=\sin (a) \cos (b)+\cos (a) \sin (b) \\ & \cos (a+b)=\cos (a) \cos (b)-\sin (a) \sin (b)\end{aligned}.$$

This means we can use the following rotation matrix, which gives the desired
transformation:

$$
\begin{bmatrix}
    \cos(b) & \sin(b) \\
    -\sin(b) & \cos(b)\\
\end{bmatrix}
\begin{bmatrix}
    \sin(a) \\
    \cos(a) \\
\end{bmatrix}
 =
\begin{bmatrix}
    \sin(a+b) \\
    \cos(a+b) \\
\end{bmatrix}.
$$

By the standard dot-product attention, we can expand and obtain

$$\boldsymbol{q}_m^{\top} \boldsymbol{k}_n=\boldsymbol{x}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{x}_n+\boldsymbol{x}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{p}_n+\boldsymbol{p}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{x}_n+\boldsymbol{p}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{p}_n.$$

Now we see that the full story of how the upstream weight matrices can work with
these positional encodings is really much more complicated,
and we'll see how RoPE will derive a new way of formulating encodings.

#### 2. Rotary Position Embeddings

The goal is for us to choose some encoding function $$f_q, f_k$$
such that their inner product is defined as some function $$g$$,
for query on the $$m$$th token and key on the $$n$$th token:

$$
\boldsymbol{q}_m^{\top} \boldsymbol{k}_n =
\left\langle f_q\left(\boldsymbol{x}_m, m\right), f_k\left(\boldsymbol{x}_n, n\right)\right\rangle=g\left(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n\right).
$$

$$g$$ hence allows us to go from absolute to relative position information.

We also want the initial condition to be such that the absolute position
of 0 gives the original vector, i.e:

$$\begin{aligned} & \boldsymbol{q}=f_q\left(\boldsymbol{x}_q, 0\right) \\ & \boldsymbol{k}=f_k\left(\boldsymbol{x}_k, 0\right)\end{aligned}$$

The authors then try to guess for values of $$f_q, f_k$$. Here, we consider
just the 2-dimensional case (we'll see how to easily extend this to arbitrary
$$d$$, where $$d$$ is even afterward).

We can re-interpret the 2d points in the complex plane, and obtain
the following formulation:

$$\begin{aligned} f_q\left(\boldsymbol{x}_q, m\right) & =R_q\left(\boldsymbol{x}_q, m\right) e^{i \Theta_q\left(\boldsymbol{x}_q, m\right)} \\ f_k\left(\boldsymbol{x}_k, n\right) & =R_k\left(\boldsymbol{x}_k, n\right) e^{i \Theta_k\left(\boldsymbol{x}_k, n\right)} \\ g\left(\boldsymbol{x}_q, \boldsymbol{x}_k, n-m\right) & =R_g\left(\boldsymbol{x}_q, \boldsymbol{x}_k, n-m\right) e^{i \Theta_g\left(\boldsymbol{x}_q, \boldsymbol{x}_k, n-m\right)}\end{aligned}$$

where $$R_q, R_k$$ gives the magnitude, and $$\Theta$$ gives the angular component.

For the initial conditions to hold, we have $$n=m=0$$, so

#### 3. Baz

### Most Glaring Deficiency

### Conclusions for Future Work
