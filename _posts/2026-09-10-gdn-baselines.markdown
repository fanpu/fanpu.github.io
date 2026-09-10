---
title: "Day 1 - Verifying the flash-linear-attention Gated DeltaNet kernel on GB10 (DGX Spark)"
layout: distill
tags: [machine-learning, linear-attention]
cover: whitney_ascent.JPG
cover_preview: whitney_ascent.webp
caption: Mt. Tumanguya (Whitney), Sierra Nevada, California, U.S
class: post-template
author: fanpu
giscus_comments: true
description: "A concise introduction to understanding Gated DeltaNets (used in Qwen3, Kimi K3, Olmo Hybrid)"
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
toc: true
---

I'm doing a 30 day research sprint on linear attention. Today is day 1.

The rough research plan is to run small-scale controlled experiments on sequence mixers between 30M to 125M parameters on my GB10 DGX Spark. 

Before I am able to interpret the significance of any experiments, I need to understand the effect of precision differences, seed noise, and correctness of kernels on my hardware. So the first test is to check if the chunked GDN kernel agrees numerically with a naive looped implementation.

The GDN update:

```python
S <- alpha * S                        # forget
v_old = S^T k                         # read what's currently stored for this key
S <- S + beta * k (v - v_old)^T       # write the error, not the value
o = S^T q                             # look up with the query
```

## Experiments

Setup: fla 0.5.2, torch 2.14.0+cu130, on GB10.

Inputs are iid Gaussian q/v, and L2-normalized Gaussian k.

We measure the max relative error across all elements (across batch, timestep, head, hidden dim) of the state and output for the forward pass, as well as the gradients in the backward pass.

We do this across both fp32 and bf16.

## Results

#### `max_rel`, fp32 (TF32 matmuls), units of $10^{-3}$

| $T$ | $d_k$ | $d_v$ | fwd out | fwd state | dq | dk | dv | dg | dbeta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 64 | 1.477 | 0.975 | 1.414 | 1.743 | 1.814 | 1.739 | 1.970 |
| 256 | 64 | 128 | 1.487 | 1.136 | 1.507 | 1.836 | 1.894 | 1.647 | 1.721 |
| 256 | 128 | 128 | 1.535 | 0.804 | 1.447 | 1.473 | 2.102 | 1.403 | 1.753 |
| 1024 | 64 | 64 | 1.793 | 0.852 | 1.421 | 1.697 | 2.037 | 1.623 | 1.991 |
| 1024 | 64 | 128 | 1.521 | 0.925 | 1.578 | 1.529 | 1.930 | 1.531 | 2.217 |
| 1024 | 128 | 128 | 1.708 | 0.835 | 1.663 | 1.432 | 2.223 | 1.776 | 1.931 |
| 4096 | 64 | 64 | 1.704 | 0.907 | 1.381 | 1.835 | 2.207 | 1.789 | 1.807 |
| 4096 | 64 | 128 | 1.624 | 0.927 | 1.610 | 1.478 | 2.034 | 1.741 | 2.091 |
| 4096 | 128 | 128 | 1.611 | 0.904 | 1.635 | 1.598 | 1.916 | 1.897 | 2.175 |

#### `max_rel`, bf16, units of $10^{-3}$

| $T$ | $d_k$ | $d_v$ | fwd out | fwd state | dq | dk | dv | dg | dbeta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 64 | 5.748 | 4.601 | 6.061 | 7.042 | 6.711 | 4.540 | 3.758 |
| 256 | 64 | 128 | 4.466 | 3.779 | 4.630 | 4.525 | 6.410 | 4.546 | 3.150 |
| 256 | 128 | 128 | 4.603 | 3.877 | 4.739 | 5.587 | 4.566 | 3.069 | 3.047 |
| 1024 | 64 | 64 | 6.342 | 5.588 | 5.525 | 5.780 | 6.757 | 4.261 | 4.039 |
| 1024 | 64 | 128 | 6.558 | 4.326 | 4.184 | 5.282 | 6.135 | 3.682 | 3.717 |
| 1024 | 128 | 128 | 6.589 | 3.230 | 4.032 | 5.128 | 8.547 | 4.270 | 4.424 |
| 4096 | 64 | 64 | 6.603 | 5.751 | 5.208 | 5.556 | 4.695 | 4.305 | 4.740 |
| 4096 | 64 | 128 | 6.858 | 4.398 | 3.704 | 6.803 | 4.484 | 3.984 | 3.162 |
| 4096 | 128 | 128 | 7.239 | 4.240 | 3.650 | 5.291 | 7.519 | 4.682 | 2.873 |

To understand how good or bad this is, we consider error in bf16 and fp32 representation respectively.

### bf16
bf16 has 7 mantissa bits, so the relative gap between each evenly spaced value with the
same exponent is $2^{-7} \approx 0.0078$. Then the roundoff error is half of that, i.e 3.9e-3.

So max relative error under bf16 is within a few multiples of bf16 roundoff error.

### fp32
bf16 has 23 mantissa bits, so the relative gap between each evenly spaced value with the
same exponent is $2^{-23} = 1.19e-7$. Then the roundoff error is 5.96e-8.

This is where things become extremely suspicious. The relative errors are around 4 orders of magnitude larger than the fp32 roundoff errors. It is likely that the kernel was not actually running in fp32 under the hood (which makes sense, since you can't make use of tensor cores).

After some investigation, `TRITON_F32_DEFAULT` defaults to "tf32" on NVIDIA GPUs.

TF32 has 10 mantissa bits, which gives relative error $2^{-10}=9.7e-4$, and roundoff error $4.9e-4$. With this, the max errors under the "fp32" (actually tf32) table are also within a few multiples of the roundoff error.

### dtype=fp32 is probably not computing in fp32

I forced Triton to use FP32 with `TRITON_F32_DEFAULT=ieee TRITON_ALWAYS_COMPILE=1`.
It definitely took effect because the script took significantly longer to run.

#### `max_rel`, fp32 with ieee matmuls (true fp32), units of $10^{-4}$

| $T$ | $d_k$ | $d_v$ | fwd out | fwd state | dq | dk | dv | dg | dbeta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 64 | 1.069 | 0.685 | 1.343 | 1.691 | 1.708 | 1.183 | 1.476 |
| 256 | 64 | 128 | 0.903 | 0.839 | 0.912 | 1.730 | 1.571 | 0.890 | 1.350 |
| 256 | 128 | 128 | 0.746 | 0.475 | 0.829 | 1.277 | 1.218 | 0.447 | 1.061 |
| 1024 | 64 | 64 | 1.073 | 0.724 | 1.015 | 1.430 | 1.622 | 1.100 | 1.463 |
| 1024 | 64 | 128 | 1.015 | 0.992 | 1.629 | 1.114 | 1.741 | 0.722 | 2.037 |
| 1024 | 128 | 128 | 0.780 | 0.392 | 0.740 | 1.078 | 1.031 | 0.761 | 1.189 |
| 4096 | 64 | 64 | 1.101 | 0.706 | 1.172 | 1.430 | 1.528 | 0.898 | 1.205 |
| 4096 | 64 | 128 | 1.331 | 0.620 | 1.660 | 1.149 | 1.343 | 1.005 | 1.175 |
| 4096 | 128 | 128 | 0.928 | 0.510 | 0.876 | 1.236 | 1.127 | 0.742 | 0.912 |

The errors have improved, but are still pretty large relative to the roundoff errors of FP32, so there is still probably something else going on that is using something with lower precision.


## Plots

Putting all three precision settings on one plot makes the story easier to see. Each dot is
one $(T, d_k, d_v)$ config, and the dashed lines are the roundoff errors (unit roundoff $u$)
of bf16, TF32, and fp32.

{% include figure.liquid
    path="/assets/img/posts/linear_attention/day1/kernel_check_summary.webp"
    class="z-depth-1"
    num=1
    alt="Max relative error of the fla chunked GDN kernel against a naive recurrent reference, for fp32-ieee, fp32-tf32, and bf16, across seven tensors."
    caption="Max relative error vs the naive recurrent reference, per tensor. bf16 (green) and TF32 (orange) both land within a few multiples of their own roundoff floors, but fp32-ieee (blue) sits about three orders of magnitude above the fp32 floor."
%}

bf16 and TF32 each sit right around their own $u$, which is what you would want. The blue
cluster is the odd one out: it drops by roughly $15\times$ once IEEE matmuls are forced, but
it settles around $10^{-4}$ instead of anywhere near $6\times10^{-8}$, i.e. still ~2000 $u_{\text{fp32}}$
off the floor. So forcing `TRITON_F32_DEFAULT=ieee` fixes the matmuls, but something else in
the chunked path is still accumulating at lower precision.

Now splitting by sequence length and shape:

{% include figure.liquid
    path="/assets/img/posts/linear_attention/day1/kernel_check_detail.webp"
    class="z-depth-1"
    num=2
    alt="Top row: max relative error vs sequence length for fp32-ieee, fp32-tf32, and bf16, one panel each. Bottom row: heatmaps of max relative error in units of each mode's unit roundoff, per tensor and per config."
    caption="Top: worst-of-7-tensors error vs sequence length $T$, with the band spanning the range across tensors. Bottom: the same errors expressed in units of each mode's own unit roundoff $u$."
%}

Some observations:
1. Error doesn't become worse with increasing sequence length $T$
2. Error in gradients are worse than forward pass outputs, which is consistent with accumulated errors

## Future work

There's still a couple of things that I haven't tested:

1. Are kernels deterministic across launches?
2. Is there a direction bias in the error?
3. How much do these errors change with different training shapes/will they hold at the training shapes I care about later?







All 126 checks passed. Values are copied from your log; the table layout was generated by script, so nothing was retyped by hand. Each row is one $(T, d_k, d_v)$ config and each column is one tensor.


### $1-\cos$, fp32, in ulps (integer multiples of $2^{-24}$)

Every fp32 value in this column is an exact integer multiple of $2^{-24} \approx 5.96\times10^{-8}$, so I show the integer. Read this table as "how many units in the last place", not as a continuous measurement.

| $T$ | $d_k$ | $d_v$ | fwd out | fwd state | dq | dk | dv | dg | dbeta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 64 | 1 | 0 | 1 | 4 | 2 | 6 | 4 |
| 256 | 64 | 128 | 2 | 2 | 2 | 3 | 1 | 6 | 4 |
| 256 | 128 | 128 | 1 | 2 | 1 | 1 | 6 | 8 | 4 |
| 1024 | 64 | 64 | 2 | 3 | 1 | 6 | 2 | 7 | 2 |
| 1024 | 64 | 128 | 2 | 1 | 2 | 4 | 3 | 8 | 3 |
| 1024 | 128 | 128 | 2 | 2 | 2 | 3 | 4 | 7 | 3 |
| 4096 | 64 | 64 | 2 | 1 | 2 | 5 | 2 | 8 | 3 |
| 4096 | 64 | 128 | 0 | 0 | 2 | 4 | 3 | 10 | 4 |
| 4096 | 128 | 128 | 1 | 1 | 1 | 2 | 0 | 8 | 4 |

### $1-\cos$, bf16, units of $10^{-6}$

| $T$ | $d_k$ | $d_v$ | fwd out | fwd state | dq | dk | dv | dg | dbeta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 64 | 6.68 | 4.23 | 9.36 | 11.32 | 8.29 | 7.03 | 5.84 |
| 256 | 64 | 128 | 6.80 | 4.17 | 9.36 | 11.27 | 8.17 | 7.87 | 5.72 |
| 256 | 128 | 128 | 6.44 | 4.05 | 9.12 | 10.85 | 7.87 | 6.80 | 5.72 |
| 1024 | 64 | 64 | 6.91 | 4.17 | 9.60 | 11.44 | 8.40 | 7.69 | 5.78 |
| 1024 | 64 | 128 | 7.03 | 4.29 | 9.66 | 11.50 | 8.46 | 7.75 | 6.08 |
| 1024 | 128 | 128 | 6.62 | 4.11 | 9.42 | 11.03 | 8.23 | 8.05 | 5.54 |
| 4096 | 64 | 64 | 7.03 | 4.23 | 9.54 | 11.56 | 8.46 | 7.63 | 5.90 |
| 4096 | 64 | 128 | 6.97 | 4.29 | 9.66 | 11.50 | 8.46 | 7.75 | 6.08 |
| 4096 | 128 | 128 | 6.80 | 4.11 | 9.42 | 11.09 | 8.23 | 7.27 | 5.54 |