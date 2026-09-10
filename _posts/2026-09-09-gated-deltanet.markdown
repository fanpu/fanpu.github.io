---
title: "Concise Introduction to Gated DeltaNets"
layout: post
tags: [machine-learning, linear-attention]
cover: whitney_morning.webp
cover_preview: whitney_morning.webp
caption: Mt. Tumanguya (Whitney), Sierra Nevada, California, U.S
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  A concise introduction to understanding Gated DeltaNets (used in Qwen3, Kimi K3, Olmo Hybrid)
---

LLM inference workloads have been moving towards longer context lengths,
stemming from usage from long-running agents and requiring working in large
codebases/contexts. This results in two problems: KV cache size grows as $O(T)$ with
the length of the sequence $T$, and each decoded token costs $O(T)$.

Linear attention methods have been gaining prominence recently as it avoids both problems entirely. 
This post is a short introduction to understanding how a recent linear attention
technique, [Gated DeltaNets](https://arxiv.org/abs/2412.06464) (GDN) from Songlin Yang et al. works.

<!-- ## Linear Attention

Recall that in softmax attention, we have: -->

# Linear Attention

Recall that in softmax attention:

$$
o_t = \sum_{i \le t} \operatorname{softmax}_i(q_t \cdot k_i)\, v_i
$$

With this, we must keep all $k_i, v_i$ around (KV cache). This grows linearly with sequence length.

## Removing the softmax

Suppose we remove the softmax:

$$
o_t = \sum_{i \le t} (q_t \cdot k_i)\, v_i
$$

Then we can rewrite:

$$
\begin{aligned}
(q_t \cdot k_i)\, v_i &= v_i\, (q_t \cdot k_i) \\
&= v_i\, (k_i \cdot q_t) \\
&= v_i\, (k_i^\top q_t) \\
&= (v_i k_i^\top)\, q_t
\end{aligned}
$$

where $v_i k_i^\top$ is a $D_v \times D_k$ outer product. 

This gives:

$$
o_t = \underbrace{\left( \sum_{i \le t} v_i k_i^\top \right)}_{S_t} q_t
$$

where $S_t$ is the state, of shape $D_v \times D_k$. 

Now our state recurrence becomes

$$
S_t = S_{t-1} + v_t k_t^\top,
$$

which is a $O(1)$ update, and we also no longer have to carry the KV cache around.
This is known as [linear attention](https://arxiv.org/abs/2006.16236).

## Associative Memory

$S$ behaves like an associative memory: if you assume all keys added to the state are orthogonal unit vectors, then for query $q = k_j$:

$$
\begin{aligned}
S_t q &= \left( \sum_{i \le t} v_i k_i^\top \right) k_j \\
&= v_j
\end{aligned}
$$

Of course, since you are limited to $D_k$ dimensions, storing more than $D_k$ keys leads to overlap, and there is interference in the contribution from the wrong entries when read.

## Problems with linear attention

### 1. No forgetting

$$
S_t = S_{t-1} + v_t k_t^\top
$$

The state keeps accumulating. After $n$ tokens, you end up with a sum of $n$ outer products and end up reading noise.

**Fix:** add a gate to decay the old state:

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top
$$

Note that $\alpha_t$ can depend on the current token.

### 2. Writes are blind

Suppose we write key $k$ with value $v_{\text{old}}$, and now we want to update it to $v_{\text{new}}$.

The update causes $k$ to become: $v_{\text{old}} + v_{\text{new}}$ (or $\text{decay} \cdot v_{\text{old}} + v_{\text{new}}$), i.e an interpolation between the new and old values.

What if we want to be able to revise the value directly?

**Fix:** what is known as the delta rule.

First, query what the memory currently stores about $k_t$:

$$
v_{\text{old}} = \tilde{S} k_t, \quad \text{where } \tilde{S} = \alpha_t S_{t-1}
$$

Then update by the error between $v_t$ and $v_{\text{old}}$:

$$
S_t = \tilde{S} + \beta_t (v_t - v_{\text{old}})\, k_t^\top
$$

where $\beta_t$ is another gating factor that can be input-dependent.

## Correctness: why is this update good?

### 1. Test-time regression

Define the loss to be "how badly do I recall $v_t$ from $k_t$":

$$
\mathcal{L}(S) = \frac{1}{2} \lVert S k_t - v_t \rVert^2,
\qquad
\nabla_S \mathcal{L} = (S k_t - v_t)\, k_t^\top
$$

Then we can see that the update corresponds to a gradient step with learning rate $\beta_t$:

$$
S \leftarrow S - \beta_t (S k_t - v_t)\, k_t^\top
$$

So this is like running online gradient descent on this regression objective in the forward pass of the state update, with $\beta_t$ as a per-token learning rate, i.e. "test-time regression".

### 2. Erase-then-write

See that the update can be rewritten as:
$$
\begin{aligned}
S_t &= \tilde{S} + \beta_t (v_t - \tilde{S} k_t)\, k_t^\top \\
&= \tilde{S} - \beta_t \tilde{S} k_t k_t^\top + \beta_t v_t k_t^\top \\
&= \tilde{S} (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
&= \tilde{S} (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
\end{aligned}
$$

Suppose keys are $\ell_2$-normalized, so $\lVert k_t \rVert = 1$.

**Claim:** $(I - \beta k k^\top)$ is a matrix that scales the $k$ direction by a factor of $1 - \beta$.

**Proof.** For some vector $x$, decompose it into directions parallel to and perpendicular to $k$, where $k$ has unit norm:

$$
x = \underbrace{(k^\top x)\, k}_{x_\parallel} + x_\perp
$$

Then

$$
k k^\top x = (k^\top x)\, k = x_\parallel
$$

so

$$
\begin{aligned}
(I - \beta k k^\top)\, x &= x - \beta k k^\top x \\
&= x - \beta (k^\top x)\, k \\
&= x - \beta x_\parallel \\
&= (1 - \beta)\, x_\parallel + x_\perp
\end{aligned}
$$

$\blacksquare$

(Note: this is a Householder-type transform.)

This means that querying the new state with $k_t$ returns

$$
S_t k_t = (1-\beta_t)\,\tilde{S} k_t + \beta_t v_t = (1-\beta_t)\, v_{\text{old}} + \beta_t v_t,
$$

So, the update uses $\beta_t \in [0, 1]$, to control how much of the old readout is erased, whilst adding the new value.

# Gated DeltaNet
Putting everything together, here's our update:

1. **Decay old state:**

$$
\tilde{S} = \alpha_t S_{t-1}
$$

2. **Read what memory says about $k_t$:**

$$
v_{\text{old}} = \tilde{S} k_t
$$

3. **Correction term:**

$$
S_t = \tilde{S} + \beta_t (v_t - v_{\text{old}})\, k_t^\top
$$

4. **Read out with query:**

$$
o_t = \frac{1}{\sqrt{D_k}}\, S_t q_t
$$

**Overall update:**

$$
S_t = \alpha_t S_{t-1} \left( I - \beta_t k_t k_t^\top \right) + \beta_t v_t k_t^\top
$$
