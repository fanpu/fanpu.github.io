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
| Intra-node | NVLink 4.0 over NVSwitch, 450 GB/s per direction per GPU |
| Inter-node | 400 Gb/s InfiniBand per GPU, i.e 50 GB/s per direction |
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
$$[8192, 128000] \approx 1.05\times 10^{9}$$ logits, i.e 2.10 GB in bf16 and 4.19
GB once they are upcast to fp32 for the cross-entropy, and
$$\partial L / \partial \text{logits}$$ is a tensor of the same size. Between
precision and fusion choices that is roughly 4 to 8 GB sitting outside any layer,
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

## B1. Communication volume

{% include theorem.md
  type="exercise"
  name="Bytes per parameter"
  statement="
    <p>Per optimizer step, per GPU, express the communication in bytes per
    parameter, i.e normalize by N so the answer is a small number, and state the
    dtype assumption that makes it so.</p>
    <ol type='a'>
      <li>Plain DDP with a ring all-reduce over 512 GPUs. Derive the 2(P-1)/P
          factor rather than quoting it.</li>
      <li>ZeRO-3 / FSDP across all 512. Itemize each collective in the step and
          what it moves.</li>
      <li>The ratio of (b) to (a), and the one-sentence reason ZeRO-3 is not free
          memory savings.</li>
      <li>Now add gradient accumulation of 4 micro-steps. What happens to (a) and
          to (b), per step and per token? State the mechanism, not the
          conclusion.</li>
    </ol>
  "
%}

{% answer %}
**(a)** DDP replicates every parameter, so the only traffic in the step is the
gradient all-reduce, and with bf16 gradients that buffer is 2 bytes per
parameter.

A ring all-reduce is a reduce-scatter followed by an all-gather. The
reduce-scatter cuts the buffer into $$P$$ shards and leaves each rank owning the
reduced version of one of them, which takes $$P-1$$ hops of $$1/P$$ of the buffer
each, i.e $$(P-1)/P$$ of the buffer sent per rank. The all-gather is the same
thing in reverse, so

$$
2 \times \frac{P-1}{P} \times 2 \ \text{bytes/param} = \frac{4(P-1)}{P} \ \text{bytes/param}
$$

which at $$P = 512$$ is 3.99, i.e 4 bytes per parameter for any $$P$$ large
enough to be worth the ring.

**(b)** ZeRO-3 shards params, gradients and optimizer state, so a rank holds
$$1/P$$ of every layer and has to gather the rest of it just before it is used:

1. Forward: all-gather each layer's bf16 params before its GEMMs and free them
   again straight after, 2 bytes/param.
2. Backward: all-gather them a second time, since they were freed. Same 2
   bytes/param, and this is the gather that also serves the recompute if you are
   checkpointing.
3. Backward: reduce-scatter the bf16 gradients, leaving each rank holding only
   the shard it owns, 2 bytes/param.

The optimizer step itself moves nothing, as the fp32 master weights and Adam
moments never leave the rank that owns their shard. Costing each collective the
same $$(P-1)/P$$ per rank as in (a),

$$
3 \times \frac{P-1}{P} \times 2 \ \text{bytes/param} = \frac{6(P-1)}{P} \approx 6 \ \text{bytes/param}
$$

**(c)** $$6/4 = 1.5$$, a 3:2 ratio. ZeRO-3 buys the memory by converting it into
extra parameter all-gathers that sit on the forward and backward critical path,
so unless they are perfectly overlapped with compute, you pay for the memory in
step time.

**(d)** Read the 4 micro-steps as putting 4x the tokens into one optimizer step,
which is what makes the per-step and per-token numbers come apart.

Under DDP the all-reduce is only needed once, at the end of the accumulation: the
micro-steps sum into the same local gradient buffer, and that summed buffer is
exactly what the all-reduce would have reduced anyway. Per step it stays at 4
bytes/param, and per token it drops 4x, to the equivalent of 1 byte/param.

Under ZeRO-3 nothing comes off the micro-step. The param all-gathers are needed
on every forward and backward because the shards are freed after each use, and
the reduce-scatter has to run every micro-step as well, since skipping it means
accumulating into a full unsharded gradient buffer, i.e paying back the memory
that ZeRO-3 was there to save (FSDP's `no_sync` is exactly that trade). So per
step this is $$4 \times 6 = 24$$ bytes/param, and per token it is unchanged at 6.
{% endanswer %}

## B2. Layout design

{% include theorem.md
  type="exercise"
  name="Spending 512 GPUs"
  statement="
    <p>Design a parallelism layout for the config above on the 512-GPU
    cluster.</p>
    <ol type='a'>
      <li>Give the full assignment (TP / PP / CP / DP-or-FSDP degrees, product
          512), and for each axis state which bandwidth tier it lives on and why
          it has to live there.</li>
      <li>For your layout, give the per-GPU memory ledger (params, grads,
          optimizer state, activations, transients) and whether it fits in 80 GB.
          Show the slack.</li>
      <li>Your layout is determined by a small number of binding constraints,
          and most of the givens in this config have slack. Name the one given
          your layout is least robust to. State which direction hurts, the
          approximate value at which your layout stops fitting or stops being
          efficient, which specific constraint binds first at that point, and the
          layout you would move to past it.</li>
      <li>Your ledger in (b) leaves some HBM unallocated. Name three distinct
          ways to spend it, each using a different mechanism to convert memory
          into throughput. Pick one. Estimate the MFU change with its mechanism,
          and state what you give up. Also say how much of the slack you would
          actually spend, and why not all of it.</li>
    </ol>
  "
%}

{% answer %}
**(a)** TP is the axis that pins down everything else. Megatron-style TP
all-reduces the activation tensor twice per transformer layer, once after
attention and once after the FFN, and both sit between two GEMMs that need the
result, so there is nothing to overlap them against. That forces TP onto the
fastest tier and caps it at the NVLink domain, i.e 8 here. PP is the opposite:
one activation boundary per stage, point-to-point, and the send for micro-batch
$$i$$ overlaps the compute of micro-batch $$i+1$$, so it is the axis you exile to
the slowest links. FSDP and DP sit in between, moving parameters and gradients
rather than activations, which can be prefetched a layer ahead and amortized over
gradient accumulation.

The degrees then follow from A1: model state is 16 bytes/param, i.e 1120 GB
unsharded against 80 GB cards, so the layout has to shard it at least 14-fold
before a single activation exists. TP already supplies 8 of that. Taking FSDP
$$= 4$$ brings it to 32, i.e 35 GB of resident state per rank, and DP is whatever
is left, $$512 / (8 \times 4) = 16$$.

| Axis | Degree | Tier | Why there |
|---|---|---|---|
| TP | 8 | NVLink, within one node | two unoverlappable activation all-reduces per layer |
| FSDP | 4 | InfiniBand | param all-gathers prefetch a layer ahead of the GEMMs that need them |
| DP | 16 | InfiniBand | one gradient all-reduce per optimizer step, amortized over the micro-steps |
| CP | 1 | | activations already fit |
| PP | 1 | | the memory is not needed and the bubble is a pure loss |
{: .table .table-bordered .table-sm }

TP fills the node, so every other axis is inter-node whatever we do with it, and
we shard only as far as the memory actually requires: each extra FSDP participant
is one more rank in an InfiniBand collective sitting on the critical path.

The data-parallel width is $$512/8 = 64$$, and the 4.19M-token global batch is 512
sequences, so a rank sees 8 sequences per optimizer step, which we run as 8
micro-steps of one sequence each. The ledger for that comes to 57 GB of the 80 in
(b), and the headroom is why CP is 1: CP would split the 10.7 GB of layer
checkpoints, which we are not short of, and charge a ring exchange of K and V in
every layer for it. PP is 1 for the same reason. Note that it is not the bubble
that rules PP out here, since $$m$$ grows as the pipeline deepens: at $$p = 8$$
the data-parallel width
falls to 8, each rank group sees 64 sequences per step, and the bubble is
$$(p-1)/m = 7/64 \approx 11\%$$. However, 11% of throughput is a strange thing
to pay for memory we already have.

**(b)** State is sharded 32-fold by $$\text{TP} \times \text{FSDP}$$, and one
sequence of 8192 tokens is in flight per rank:

| Item | Size | Per GPU |
|---|---|---|
| bf16 params | $$2N/32$$ | 4.4 GB |
| bf16 gradients | $$2N/32$$ | 4.4 GB |
| fp32 master weights, Adam $$m$$ and $$v$$ | $$12N/32$$ | 26.3 GB |
| Layer checkpoints | $$L \times t \times d_{\text{model}} \times 2$$ | 10.7 GB |
| Logits, bf16 plus fp32 logits and $$\partial L/\partial \text{logits}$$ | $$3 \times [8192, 128000]$$ | 10.5 GB |
| Recompute transients, one layer | $$3 \times [t, d_{\text{model}}]$$ plus $$4 \times [t, d_{ff}/8]$$ and the attention pieces | 0.7 GB |
| FSDP gather and reduce-scatter buffers | 2 layers of a TP shard, in flight | 0.2 GB |
| **Total** | | **57.2 GB** |
{: .table .table-bordered .table-sm }

so it fits, with 22.8 GB of slack. Note that the logit tensors cost about as much
as all 80 layer checkpoints put together, and they do so only because we assumed
vanilla TP, which replicates the boundary tensors and the vocabulary rather than
sharding either. The transients are small for the same reason the logits are not,
i.e $$d_{ff}$$ is column-parallel under TP so each rank only ever materializes
$$[8192, 3584]$$ of the SwiGLU intermediates, and FlashAttention never
materializes an $$S \times S$$ score matrix at all, which leaves the three
replicated $$[t, d_{\text{model}}]$$ tensors of the layer as the larger half of
that 0.7 GB. Allocator fragmentation and NCCL workspace eat into the slack by a
couple of GB, so call it 20 GB of genuinely free HBM.

**(c)** The given we are least robust to is the sequence length, going up. The
model state is fixed at $$70\times 10^{9}/32 = 2.19\times 10^{9}$$ params
resident at 16 bytes each, i.e 35 GB, plus 0.2 GB of FSDP buffers, and everything
else in the ledger is linear in $$s$$: 10.7 GB of layer checkpoints, 10.5 GB of
logit tensors and 0.7 GB of recompute transients, i.e 21.9 GB at $$s = 8192$$, or 2.7 MB per token. Keeping
8 GB (10% of HBM) back for fragmentation and NCCL workspace,

$$
35.2 + 2.7\times 10^{-3} s \leq 72
\quad\Longrightarrow\quad
s \leq 1.4\times 10^{4}
$$

so the layout stops fitting a little short of 14k tokens, and what binds is
activation memory rather than anything on the wires. It binds first because the
micro-batch is already a single sequence, i.e there is nothing smaller left to
make smaller before the sequence itself has to be split.

Two cheaper rungs come before a new parallelism axis. Vocab-parallel
cross-entropy keeps the logits sharded across the TP group and never
materializes $$[s, V]$$ on any one rank, taking 10.5 GB to 1.3 GB. Sequence
parallelism shards the layer checkpoints by 8 as well, 10.7 GB to 1.3 GB, and
cuts the replicated part of the transient from 402 MB to 50 MB, though recompute
then has to re-gather the two $$[s, d_{\text{model}}]$$ inputs to QKV and the
MLP, which puts 268 MB back and lands the transient near 0.6 GB. That is 3.2 GB
of $$s$$-dependent memory at 8192, i.e 0.39 MB per token, and

$$
35.2 + 0.39\times 10^{-3} s \leq 72
\quad\Longrightarrow\quad
s \leq 9.4\times 10^{4}
$$

However, a second constraint binds well before that, and it is not memory. There
are $$512/8 = 64$$ data-parallel workers against a $$2^{22}$$-token global batch,
so a worker sees $$2^{22}/(64 s)$$ sequences per step: 8 at $$s = 8192$$, 1 at
$$s = 65536$$, and half of one at $$s = 131072$$. 64k is the real wall for this
layout even after both rungs, since past it some workers have no sequence at all
and the global batch can only be held where it is by narrowing the data-parallel
width.

Past it the degree comes out of DP and goes to context parallelism. At
$$s = 131072$$ that is TP = 8, CP = 2, FSDP = 4, DP = 8, product 512, and
$$32 \times 131072 = 2^{22}$$ tokens per step. CP costs a K and V exchange in
every layer, which is cheap here: with GQA at TP = 8 each rank holds a single KV
head, so at CP = 2 each rank's share of K is $$[65536, 128]$$ in bf16, i.e 16.8
MB, and the same again for V. At that size the all-gather CP that Llama 3 uses
beats ring attention.

**(d)** Vocab-parallel cross-entropy from (c) is free memory rather than a trade,
so it goes on first: the logits drop from 10.5 GB to 1.3 GB, the ledger to 48 GB,
and one sequence of activations to $$10.7 + 1.3 + 0.7 = 12.7$$ GB. That leaves 32
GB free, of which 8 GB is the headroom above, so about 24 GB is spendable.

A3 gives the compute side, $$2.70\times 10^{18}$$ FLOPs per step against
$$5.06\times 10^{17}$$ FLOP/s of cluster peak, i.e a 5.33 s floor under the
measured 12.0 s and 33.4% MFU. The communication side is the FSDP traffic, for a
TP shard of $$70\times 10^{9}/8 = 8.75\times 10^{9}$$ params at 2 bytes each:

$$
\begin{align*}
\text{one all-gather over 4 ranks} &= \tfrac{3}{4} \times 17.5 = 13.1\ \text{GB} \\
\text{two all-gathers and a reduce-scatter} &= 39.4\ \text{GB} \\
\text{at 50 GB/s} &= 0.79\ \text{s per micro-step} \\
\text{over 8 micro-steps} &= 6.3\ \text{s per step}
\end{align*}
$$

so the parameter traffic is as long as the compute, and it is harmless only if
the overlap is close to perfect, which the 12.0 s step says it is not. Take half
of it as exposed, i.e 3.1 s of the 6.7 s that A3 could not account for.

The first way to spend the slack is to run fewer micro-steps, by doubling the
micro-batch to two sequences. The tokens per step do not change, so it is 4
micro-steps instead of 8 and the traffic halves to 3.1 s, of which 1.6 s is
exposed. The step goes 12.0 s to 10.5 s and MFU to
$$33.4 \times 12.0/10.5 \approx 38\%$$, for one more sequence of activations,
12.7 GB.

The second is to buy back recompute, keeping a layer's transients rather than
recomputing them. At 0.7 GB per layer the 24 GB covers 34 layers, 43% of the 80,
and recompute is $$2N$$ of the $$8N$$ per step, so this removes
$$0.43/4 = 10.7\%$$ of the step's FLOPs. The floor falls 5.33 s to 4.76 s, the
step to 11.4 s, and MFU to 35.1%. It is the smallest of the three because the
step is not compute-bound.

The third is to stop sharding the parameters, keeping the 17.5 GB TP shard
replicated within the FSDP group and sharding only gradients and optimizer
state. Both param all-gathers leave every micro-step, so the traffic is the
reduce-scatter alone, $$8 \times 13.1 = 105$$ GB, plus one 13.1 GB all-gather of
the updated params at the step boundary, i.e 118 GB and 2.4 s against 6.3 s. On
the same half-exposed assumption the step goes to 10.1 s and MFU to
$$33.4 \times 12.0/10.1 \approx 40\%$$, for $$17.5 - 4.4 = 13.1$$ GB.

We would take the third. It removes the most communication per gigabyte and
needs nothing else turned on first, and what it gives up is 13.1 GB permanently,
so the sequence length and the micro-batch are frozen where they are until we
shard the params again. That is 13 GB of the 32, and the rest stays unspent: 8 GB
of it was never really ours, and the 11 GB left over is not enough to also take
the first way, since two sequences want 12.7 GB. Every MFU figure above rests on
the guess that half the FSDP traffic is exposed, which is the first thing to
check in a profile before believing any of them.
{% endanswer %}

## B3. Perturbations

{% include theorem.md
  type="exercise"
  name="What breaks first"
  statement="
    <p>Each part below changes one thing about the layout you built in B2 and
    leaves the rest of the config alone.</p>
    <ol type='a'>
      <li>TP = 8 within a node, Megatron-style. Count the collectives in a
          single transformer layer, giving the type of each and the bytes it
          moves per micro-batch. Then turn on Megatron sequence parallelism:
          which collectives replace them, how many bytes do those move, and is
          the total larger, smaller, or the same?</li>
      <li>Derive the bubble fraction for a 1F1B pipeline of p stages fed m
          micro-batches. Then set PP = 8, taken out of the data-parallel
          width of your B2 layout: how large does m have to be to hold the
          bubble under 5%, and does your global batch have room for that
          many?</li>
      <li>Sequence length goes from 8k to 128k, with the global token batch held
          where it is. What breaks first, and which axis do you add for it? Give
          the activation-memory number the answer rests on.</li>
      <li>Swap the cluster for H800-class nodes, where NVLink is cut to roughly
          200 GB/s per direction. Which single decision from B2(a) does that
          reverse, and what do you put in its place?</li>
    </ol>
  "
%}

{% answer %}
**(b)** Write $$t_f$$ and $$t_b$$ for the forward and the backward of one
micro-batch on one stage. 1F1B runs in three phases: a warmup of $$p-1$$ forwards
while the pipeline fills, a steady phase where every stage alternates one forward
and one backward, and a cooldown of $$p-1$$ backwards while it drains. All $$m$$
micro-batches of real work happen in the steady phase, so the fill and the drain
are the whole of the waste:

$$
\begin{align*}
\text{useful} &= m(t_f + t_b) \\
\text{bubble} &= (p-1) t_f + (p-1) t_b = (p-1)(t_f + t_b) \\
\text{bubble fraction} &= \frac{(p-1)(t_f + t_b)}{m(t_f + t_b)} = \frac{p-1}{m}
\end{align*}
$$

which is the Megatron convention, bubble time over useful time. Note that as a
fraction of wall clock the same schedule gives $$(p-1)/(m+p-1)$$, which is 9.9%
where this one says 10.9%, so quote which of the two you mean. Note also that
1F1B does not shrink the bubble relative to GPipe at all, it only holds fewer
micro-batches of activations at once. To get under 5% at $$p = 8$$,

$$
\frac{7}{m} < 0.05 \quad\Longrightarrow\quad m > 140
$$

TP = 8 and PP = 8 leave 8 for FSDP and DP together, so there are
$$512/(8 \times 8) = 8$$ pipeline groups and each sees $$512/8 = 64$$ of the 512
sequences in the global batch. At one sequence per micro-batch that is $$m = 64$$
and a bubble of $$7/64 = 10.9\%$$, short of what 5% needs by more than a factor of
two.

Deepening the pipeline does not rescue it either, since $$m$$ and $$p$$ grow
together: at pipeline degree $$p$$ there are $$64/p$$ groups of $$8p$$ sequences
each, so $$m = 8p$$ and

$$
\text{bubble} = \frac{p-1}{8p} = \frac{1}{8}\left(1 - \frac{1}{p}\right)
$$

which increases with $$p$$ towards $$1/8$$ and is already 6.25% at $$p = 2$$. At
this global batch, 5% is out of reach for any pipeline at all.

Interleaving is the escape that leaves the batch alone. Interleaved 1F1B gives
each device $$v$$ non-contiguous chunks of the layer stack and divides the bubble
by $$v$$, so $$v = 2$$ takes 10.9% to 5.5% and $$v = 3$$ to 3.6%, at the price of
$$v$$ times as many point-to-point sends per micro-batch. The alternative is to
grow the global batch, and $$m > 140$$ across 8 groups is 1120 sequences, i.e 9.2M
tokens against the 4.19M we are given, which is a change to the optimization
rather than to the layout. The memory that PP is usually bought for is not on
offer here anyway: 1F1B keeps up to $$p$$ micro-batches in flight at the first
stage, each carrying $$L/p = 10$$ layer checkpoints, i.e the same 80 checkpoints
and the same 10.7 GB that B2 already pays with no pipeline at all.

**(c)** The batch breaks first. A $$2^{22}$$-token global batch at $$s = 2^{17}$$
is $$2^{22}/2^{17} = 32$$ sequences, so any data-parallel width above 32 leaves
ranks with nothing to run, and B2's $$\text{FSDP} \times \text{DP} = 4 \times 16
= 64$$ is twice too wide. The degree comes out of DP and goes to context
parallelism, the one axis that splits a single sequence: TP = 8, CP = 2, FSDP =
4, DP = 8, product 512, with 32 groups each holding one 128k sequence.

Memory breaks too, and CP by itself does not save it. A layer boundary at
$$\text{CP} = 1$$ is $$[1, s, d_{\text{model}}]$$ in bf16, i.e
$$131072 \times 8192 \times 2 = 2.15$$ GB, so the 80 checkpoints alone are 172
GB, and $$\text{CP} = 2$$ only brings that to 86 GB. What makes it fit is the two
rungs from B2(c). Sequence parallelism shards the layer boundaries across TP as
well, so a rank holds $$[1, s/(\text{TP} \cdot \text{CP}), d_{\text{model}}] =
[1, 8192, 8192]$$, i.e 134 MB per layer and 10.7 GB for all 80, which is the same
number as B2 because 128k tokens over 16 ranks is 8k tokens over one.
Vocab-parallel cross-entropy shards the logits over TP and CP splits the sequence
again, i.e $$[65536, 16000]$$ per rank, 2.1 GB in bf16 and 10.5 GB once the fp32
copy and its gradient are counted. The recompute transients scale the same way,
16 times B2(c)'s 0.6 GB at 8k and halved by CP, i.e 4.8 GB:

| Item | Per GPU |
|---|---|
| Model state, 16 bytes/param over $$\text{TP} \times \text{FSDP} = 32$$ | 35.0 GB |
| FSDP gather and reduce-scatter buffers | 0.2 GB |
| Layer checkpoints, $$80 \times [1, s/16, d_{\text{model}}]$$ | 10.7 GB |
| Logits, vocab-parallel and CP-split, bf16 plus fp32 and its gradient | 10.5 GB |
| Recompute transients | 4.8 GB |
| **Total** | **61.2 GB** |
{: .table .table-bordered .table-sm }

so 128k fits with 19 GB to spare, and the only new running cost is the K and V
exchange that CP adds to every layer, 16.8 MB per rank per tensor at these
degrees.
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

grade_gqa_attention(gqa_attention)
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

grade_gqa_decode(gqa_decode_step, gqa_attention)
{% endraw %}
{% endhighlight %}

Note that this checks your decode against your own prefill rather than against
the truth, so a bug shared by both passes it. Run the D1 grader first.
{% endanswer %}

## D3. Online softmax

Compute $$\text{out} = \text{softmax}(q K^\top / \sqrt{d}) V$$ for a single
query, seeing $$K$$ and $$V$$ one chunk of rows at a time, never allocating
anything $$O(S)$$.

Hint: carry three quantities across the chunks, the running max $$m$$, the
running denominator $$l$$, and the running output accumulator $$o$$.

**(a)** What are the online updates for $$m$$, $$l$$ and $$o$$? 

**(b)** Implement online softmax.

{% highlight python linenos %}
{% raw %}
def online_softmax_attention(q, K, V, chunk_size):
    """
    q: [d]        single query vector
    K: [S, d]
    V: [S, d]
    Returns: [d]
    Constraint: never allocate a tensor of size O(S).
    Process K and V in chunks of chunk_size rows.
    """
{% endraw %}
{% endhighlight %}

{% answer %}
**(a)** Write $$s_j = q \cdot k_j / \sqrt{d}$$. Reduce each chunk $$C$$ to a state
$$(m_C, l_C, o_C)$$, the first two scalars and the third of shape $$[d]$$:

$$
m_C = \max_{j \in C} s_j, \qquad
l_C = \sum_{j \in C} e^{s_j - m_C}, \qquad
o_C = \sum_{j \in C} e^{s_j - m_C} v_j.
$$

Two states over disjoint rows merge by moving both to a common reference:

$$
\begin{align*}
m &= \max(m_A, m_B) \\
l &= e^{m_A - m} l_A + e^{m_B - m} l_B \\
o &= e^{m_A - m} o_A + e^{m_B - m} o_B
\end{align*}
$$

which is exact since $$e^{s_j - m} = e^{s_j - m_A} e^{m_A - m}$$ and that factor
does not mention $$j$$, so it comes out of the sum. Fold the merge over the
chunks from $$(-\infty, 0, 0)$$ and return $$o / l$$.

Note that the merge is associative and symmetric, so a kernel can split the
sequence across workers and reduce pairwise, and $$m$$ never has to be the global
max since it cancels in $$o / l$$.
{% endanswer %}

{% answer Grader %}
For **(b)**, a full-sequence softmax is the reference, run once at ordinary scale
and once at a scale where a version that forgets to subtract the max overflows:

{% highlight python linenos %}
{% raw %}
import torch

def grade_online_softmax(fn, S=257, d=64, chunk=32, scale=1.0, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(d) * scale
    K = torch.randn(S, d)
    V = torch.randn(S, d)

    ref = torch.softmax(K @ q / d**0.5, dim=0) @ V
    got = fn(q, K, V, chunk)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    err = (got - ref).abs().max().item()
    print("ok" if err < 1e-4 else "FAIL", "(scale %g, max abs err %.2e)" % (scale, err))

grade_online_softmax(online_softmax_attention)
grade_online_softmax(online_softmax_attention, scale=50.0)
{% endraw %}
{% endhighlight %}

$$S = 257$$ against a chunk size of 32 leaves a ragged final chunk of one row,
where off-by-one slicing shows up. The $$O(S)$$ constraint is on you to check,
since concatenating the chunks and calling `torch.softmax` once also passes.
{% endanswer %}
