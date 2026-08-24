---
title: "Exercises for Understanding the Frontier of LLM Training"
layout: post
tags: [machine-learning, ml-systems]
cover: aoraki.webp
cover_preview: aoraki.webp
caption: Mt. Aoraki (Mt. Cook), South Island, New Zealand
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  TODO
published: false
---

I frequently get questions from friends and internet strangers on good ways to
get started with understanding how LLMs work or are trained. 

This series of blog posts of exercises is a pedagogical attempt to 
let one quickly figure out what they don't know, so it is easier to find
out what to focus time on learning.

For resources, I highly recommend: [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook), and [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).


## Fundamentals

We'll use the following config for subsequent exercises on this post.

| Quantity | Value |
|---|---|
| Params N | 70e9 (dense) |
| Layers L | 80 |
| d_model | 8192 |
| d_ff | 28672 (SwiGLU) |
| Heads | 64 query, 8 KV (GQA), head_dim 128 |
| Vocab | 128k |
| Seq len s | 8192 |
| Cluster | 512× H100 SXM (64 nodes × 8) |
| H100 | 989 TFLOP/s bf16 dense, 80 GB HBM @ 3.35 TB/s |
| Interconnect | NVLink intra-node; 400 Gb/s InfiniBand per GPU inter-node |
| Global batch | 4.19e6 tokens (2²²) |
{: .table .table-bordered .table-sm }

## A1. Parameter, gradient, and optimizer state memory

{% include theorem.md
  type="exercise"
  name="Training memory usage"
  statement="
    <p>Standard mixed-precision AdamW: bf16 params for compute, bf16 gradients,
    fp32 master weights, fp32 Adam m and v.</p>
    <ol type='a'>
      <li>Give the per-parameter byte cost, itemized.</li>
      <li>Total for the 70B, unsharded. Does it fit on one 8×H100 node
          (640 GB aggregate)?</li>
      <li>Under ZeRO-3 across all 512 GPUs, what is the per-GPU cost of
          params + grads + optimizer state?</li>
      <li>Given (c), what actually dominates per-GPU memory at this scale?</li>
    </ol>
  "
%}

{% answer %}
**(a)** Every parameter carries five tensors:

| Tensor | Precision | Bytes/param |
|---|---|---|
| Compute weights | bf16 | 2 |
| Gradients | bf16 | 2 |
| Master weights | fp32 | 4 |
| Adam $$m$$ | fp32 | 4 |
| Adam $$v$$ | fp32 | 4 |
| **Total** | | **16** |
{: .table .table-bordered .table-sm }

Note that the 16 assumes bf16 gradients. Keeping gradients in fp32, as some
frameworks do by default, makes it 18.

**(b)** $$16 \times 7\times 10^{10} = 1.12\times 10^{12}$$ bytes, or 1.12 TB. A
node holds 640 GB, so no: the state alone is $$1.75\times$$ the node's aggregate
HBM, before a single activation.

**(c)** ZeRO-3 shards params, gradients and optimizer state (fp32 master weights
included) across the data-parallel group, so all 16 bytes divide by 512:

$$
\frac{16 \times 7\times 10^{10}}{512} = 2.19\times 10^{9}\ \text{bytes}
$$

which is 2.19 GB per GPU resident. The transient all-gather buffers for whichever
layers are in flight sit on top of that, about 1.71 GB per layer of bf16 weights
here.

**(d)** Activations. 2.19 GB is under 3% of an 80 GB H100, so once the state is
sharded it stops being the constraint and activation memory takes over. This is
the argument for setting the sharding degree from how much memory you actually
need rather than maximizing it, i.e HSDP (shard within a node, replicate across
nodes) rather than flat ZeRO-3 at this scale.
{% endanswer %}

## A2. Activations and recomputation

{% include theorem.md
  type="exercise"
  name="The cost of not remembering"
  statement="
    <p>Global batch is 4.19M tokens (2^22). Activation checkpointing at
    layer-boundary granularity: you store each transformer layer's input and
    recompute everything inside the layer during backward.</p>
    <ol type='a'>
      <li>Tokens per GPU per step?</li>
      <li>Estimate per-GPU activation memory for the stored checkpoints.
          State your formula before you plug in numbers.</li>
      <li>What did your estimate in (b) omit, and roughly how much does that
          add?</li>
      <li>What is the FLOPs overhead of this checkpointing scheme, as a fraction
          of the un-checkpointed training FLOPs? Derive it, do not recall it.</li>
    </ol>
  "
%}

{% answer %}
**(a)** $$2^{22} / 2^{9} = 2^{13} = 8192$$ tokens per GPU, which is exactly one
sequence.

**(b)** One checkpoint per layer, each the layer's input of shape
$$[t, d_{\text{model}}]$$ in bf16, with $$t$$ the tokens per GPU:

$$
\text{bytes} = L \times t \times d_{\text{model}} \times \text{bytes/elem}
$$

$$
80 \times 8192 \times 8192 \times 2 = 1.07\times 10^{10}\ \text{bytes}
\approx 10.7\ \text{GB}
$$

or 134 MB per layer.

**(c)** The largest omission is the vocabulary projection. The LM head produces
$$[8192, 131072] \approx 1.07\times 10^{9}$$ logits, i.e 2.15 GB in bf16 and 4.29
GB once they are upcast to fp32 for the cross-entropy, and
$$\partial L / \partial \text{logits}$$ is a tensor of the same size. Between
precision and fusion choices that is roughly 4 to 9 GB sitting outside any layer,
which is what fused linear-cross-entropy kernels (Liger, cut cross-entropy) are
for: they never materialize the full logit tensor.

The estimate also describes steady state rather than peak. During backward one
layer's interior activations are re-materialized and held on top of every stored
checkpoint, the attention scores and the $$d_{ff}$$-wide SwiGLU intermediates, so
the high-water mark is a layer above the 10.7 GB. Communication buffers, kernel
workspace and allocator fragmentation account for the rest.

Running total is around 20 GB out of 80, so roughly 60 GB of HBM is doing nothing
in this configuration.

**(d)** Training compute per token is $$2N$$ for the forward pass and $$4N$$ for
the backward, the $$6N$$ that sits inside $$6ND$$. Layer-boundary checkpointing
replays the entire forward during backward, adding another $$2N$$:

$$
\frac{2N}{6N} = \frac{1}{3}
$$

so +33% of the un-checkpointed training FLOPs.
{% endanswer %}

## A3. MFU, HFU, and where the time went

{% include theorem.md
  type="exercise"
  name="12 seconds a step"
  statement="
    <p>You run this config on all 512 GPUs and measure 12.0 seconds per step at
    the 4.19M-token global batch.</p>
    <ol type='a'>
      <li>Compute total model FLOPs per step. State which term you are using
          and why.</li>
      <li>Compute the attention contribution separately. What fraction of the
          total is it, and at what sequence length would it start to dominate?</li>
      <li>Compute MFU.</li>
      <li>Compute HFU for the checkpointing scheme in A2, and explain in one
          sentence why the two numbers differ and which one you would report to
          a VP.</li>
      <li>12 s/step is bad. Give a prioritized hypothesis list, ordered by
          (probability × cost to test), for where the missing throughput
          went, and what evidence is sufficient to confirm or reject it.</li>
    </ol>
  "
%}

{% answer %}
**(a)** The $$6ND$$ term, which counts the dense GEMM work over every parameter,
$$2N$$ forward and $$4N$$ backward:

$$
6ND = 6 \times 7.0\times 10^{10} \times 4.194\times 10^{6}
  = 1.761\times 10^{18}\ \text{FLOPs}
$$

It excludes recompute, since that is work the hardware did rather than work the
model required, and it excludes the $$s^2$$ attention term, which is additive and
computed in (b). Adding that gives $$2.03\times 10^{18}$$ FLOPs per step. At
$$s = 8192$$ attention is not negligible.

**(b)** Attention FLOPs are $$2s^2 d$$ for $$QK^\top$$ and $$2s^2 d$$ for
$$PV$$ per layer per sequence, tripled for the backward pass and divided by
$$s$$ to get a per-token figure:

$$
12\,L\,s\,d_{\text{model}} \times D
= 12 \times 80 \times 8192 \times 8192 \times 4.194\times 10^{6}
= 2.70\times 10^{17}\ \text{FLOPs/step}
$$

which is 13.3% of the total. Note that this counts the full $$s \times s$$ score
matrix with no factor of two for causal masking and ignores the softmax, matching
the accounting convention that $$6ND$$ comes from; a causal-only convention
halves it. GQA does not reduce any of this, it only reduces KV-cache memory at
inference time. For the crossover, set the two terms equal:

$$
12\,L\,s\,d = 6N
\quad\Longrightarrow\quad
s^{*} = \frac{N}{2Ld} = \frac{7.0\times 10^{10}}{2 \times 80 \times 8192}
\approx 5.4\times 10^{4}
$$

**(c)** Cluster peak is $$512 \times 989\times 10^{12} = 5.06\times 10^{17}$$
FLOP/s, so

$$
\text{MFU} = \frac{2.03\times 10^{18}}{12.0 \times 5.06\times 10^{17}} = 33.4\%
$$

**(d)** HFU counts everything the hardware retired, including the $$4/3$$ from
A2(d):

$$
\text{HFU} = \frac{2.70\times 10^{18}}{6.08\times 10^{18}} = 44.4\%
$$

HFU is the engineer's number, how close the kernels run to silicon peak, and MFU
is the organization's number, useful model FLOPs per hardware-second and
comparable across implementations. Report MFU externally: HFU can be inflated by
adding wasteful recomputation, so a rising HFU does not mean a better system. The
11-point gap here is the recomputation tax.

**(e)** Llama 3 405B reported 38 to 43% overall bf16 MFU, so 33.4% is mediocre
rather than catastrophic, maybe 5 to 10 points below a good production run.

Before enumerating causes it helps to bound the gap, by decomposing the step time
into a floor and a remainder:

$$
\begin{align*}
\text{ideal floor (model FLOPs only)} &= \frac{2.03\times 10^{18}}{5.06\times 10^{17}} = 4.0\ \text{s} \\
\text{compute floor (with recompute)} &= \frac{2.70\times 10^{18}}{5.06\times 10^{17}} = 5.33\ \text{s} \\
\text{measured} &= 12.0\ \text{s} \\
\text{unexplained} &= 12.0 - 5.33 = 6.7\ \text{s}
\end{align*}
$$

The two buckets have different remedies. The first is the recompute tax, which
lifts the floor from 4.0 s to 5.33 s and caps MFU at 75% before any inefficiency
at all, and it is not a hypothesis: A2 already established that a third of the
FLOPs go to recompute while some 60 GB of HBM sits idle. Selective recomputation,
retaining the cheap-to-store activations and recomputing only the attention
interior, buys most of that back. `torch.cuda.max_memory_allocated()` tells you
how much headroom there is to spend.

The second is the 6.7 s in which the GPUs are not doing useful math, and any
hypothesis here has to claim a share of those 6.7 seconds.

1. ZeRO-3 is sharded flat across all 512 GPUs, so every per-layer collective
   traverses the 400 Gb/s inter-node tier and waits on the slowest of 512
   participants. HSDP keeps the gather on NVLink. Check the profiler's
   communication share and a per-rank histogram of collective times.
2. Communication that is not overlapped with compute. Compare collective time
   against the overlap windows in a trace.
3. Stragglers, i.e a thermally-limited GPU or a slow node pacing a synchronous
   step while looking fine in every aggregate metric. Check per-rank CUDA event
   timings and NCCL diagnostics.
4. Then memory-bound elementwise work, and host-side dataloader overhead.
{% endanswer %}

## D1. GQA

Implement GQA attention forward:

{% highlight python linenos %}
{% raw %}
def gqa_attention(x, Wq, Wk, Wv, Wo, n_q_heads, n_kv_heads):
    """
    x:  [B, S, D]  float32
    Wq: [D, n_q_heads  * head_dim]
    Wk: [D, n_kv_heads * head_dim]
    Wv: [D, n_kv_heads * head_dim]
    Wo: [n_q_heads * head_dim, D]
    Causal masking. Returns [B, S, D].
    """
{% endraw %}
{% endhighlight %}

Hints:
- Heads are grouped contiguously: first G query heads share kv head 0, next G share kv head 1, etc (same as Llama)
- You shouldn't need to copy K or V anywhere

{% answer Grader %}
Expanding the KV heads and calling `scaled_dot_product_attention` gives a
reference in three lines, so the grader is that, compared against yours:

{% highlight python linenos %}
{% raw %}
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

def grade_gqa_attention(fn, B=2, S=16, D=64, n_q=8, n_kv=2, seed=0):
    torch.manual_seed(seed)
    hd, G = D // n_q, n_q // n_kv
    x = torch.randn(B, S, D)
    Wq = torch.randn(D, n_q * hd) / D**0.5
    Wk = torch.randn(D, n_kv * hd) / D**0.5
    Wv = torch.randn(D, n_kv * hd) / D**0.5
    Wo = torch.randn(n_q * hd, D) / D**0.5

    q = (x @ Wq).view(B, S, n_q, hd).transpose(1, 2)
    k = (x @ Wk).view(B, S, n_kv, hd).transpose(1, 2).repeat_interleave(G, 1)
    v = (x @ Wv).view(B, S, n_kv, hd).transpose(1, 2).repeat_interleave(G, 1)
    ref = sdpa(q, k, v, is_causal=True)
    ref = ref.transpose(1, 2).reshape(B, S, n_q * hd) @ Wo

    got = fn(x, Wq, Wk, Wv, Wo, n_q, n_kv)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    err = (got - ref).abs().max().item()
    print("ok" if err < 1e-4 else "FAIL", "(max abs err %.2e)" % err)
{% endraw %}
{% endhighlight %}

The `repeat_interleave` is the grouping convention from the hint written out
explicitly, and it is also the copy that the second hint asks you to avoid: it
materializes K and V at $$64$$ heads instead of $$8$$.
{% endanswer %}

## D2. KV-cache decoding

Implement the KV-cache decode step:

{% highlight python linenos %}
{% raw %}
def gqa_decode_step(x, Wq, Wk, Wv, Wo, n_q_heads, n_kv_heads, cache, pos):
    """
    Single-token decode step with a KV cache.

    x:     [B, 1, D]      the one new token's hidden state
    Wq:    [D, n_q_heads  * head_dim]
    Wk:    [D, n_kv_heads * head_dim]
    Wv:    [D, n_kv_heads * head_dim]
    Wo:    [n_q_heads * head_dim, D]
    cache: (k_cache, v_cache) from the previous step, or None on the first call
    pos:   int, index of the new token (0-based); number of cached tokens

    Returns: (out, cache)
        out:   [B, 1, D]
        cache: updated (k_cache, v_cache)
    """
{% endraw %}
{% endhighlight %}

Questions to ponder before writing any code:

- What shape do you store the cache in, and is it pre-allocated to max_seq_len or grown by concatenation? Name the cost of each choice.
- Which parts of the prefill code disappear entirely, and why? 
- For the reference config (80 layers, 8 KV heads, head_dim=128, bf16) at 8192 context, B=1: how many bytes is the full KV cache across the entire model?

{% answer Grader %}
Decoding a sequence one token at a time has to give the same thing as running the
whole sequence through D1 in one shot, so the grader needs no reference of its
own:

{% highlight python linenos %}
{% raw %}
import torch

def grade_gqa_decode(decode, prefill, B=2, S=16, D=64, n_q=8, n_kv=2, seed=0):
    torch.manual_seed(seed)
    hd = D // n_q
    x = torch.randn(B, S, D)
    W = [torch.randn(a, b) / a**0.5
         for a, b in [(D, n_q*hd), (D, n_kv*hd), (D, n_kv*hd), (n_q*hd, D)]]

    ref = prefill(x, *W, n_q, n_kv)

    cache, outs = None, []
    for t in range(S):
        out, cache = decode(x[:, t:t+1], *W, n_q, n_kv, cache, t)
        outs.append(out)
    got = torch.cat(outs, dim=1)

    assert got.shape == ref.shape, (got.shape, ref.shape)
    err = (got - ref).abs().max().item()
    print("ok" if err < 1e-4 else "FAIL", "(max abs err %.2e)" % err)
{% endraw %}
{% endhighlight %}

Note that this checks your decode against your own prefill rather than against
the truth, so a bug shared by both passes it. Run the D1 grader first.
{% endanswer %}
