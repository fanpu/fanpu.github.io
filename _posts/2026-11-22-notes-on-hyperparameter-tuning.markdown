---
title: "Notes on Hyperparameter Tuning"
published: false
layout: distill
tags: [machine-learning]
cover: great_allegheny_passage_ridges.webp
cover_preview: great_allegheny_passage_ridges.webp
caption: Somewhere along the trail of the Great Allegheny Passage, USA
class: post-template
author: fanpu
giscus_comments: true
description: ""
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
toc: true
---

These are notes from an investigation into hyperparameter tuning, based on the
first assignment of
[Stanford CS 312: Deep Learning Alchemy](https://deep-learning-alchemy.github.io/).
The first experiment sweeps one hyperparameter at a time. Learning rate matters
most and batch size comes second. Weight decay is close to flat over an
$81\times$ range, and warmup only matters when it is too short.

## Setup

Every run trains `d8`, the course's default model. It is a decoder-only
transformer with 8 layers and hidden size 512. The runs are compute-matched.
The model is the same in every run, and each run makes a single pass over the
same 600,000 sequences of 1,024 tokens.

The default recipe is AdamW with peak learning rate $3 \times 10^{-3}$, weight
decay $0.1$ and batch size 64, which gives 9,375 optimizer steps. The learning
rate warms up linearly from zero over the first 1% of steps, then decays
linearly to zero. The default run ends at a validation loss of 2.925.

## Sweeping one hyperparameter at a time

Each sweep changes one hyperparameter and leaves the rest at their defaults.
Learning rate, weight decay and warmup move in factors of 3 around the default.
Batch size moves in factors of 2. Each point is a single run with one seed, and
I have not measured the seed-to-seed noise yet. Gaps of a few thousandths may
not be real.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_sweeps.webp"
    class="z-depth-1"
    zoomable=true
    num=1
    alt="Top row: final validation loss against learning rate, weight decay, warmup fraction and batch size, one panel each. Bottom row: validation loss curves over training for each run."
    caption="Top: final validation loss for each sweep, on a shared vertical axis. The dashed line is a quadratic fit in $\log_{10}$ of the hyperparameter, and $a$ is its curvature in loss per decade squared. Bottom: the validation loss curves for the same runs. Learning rate and batch size have the largest curvature."
%}

| Learning rate | $3.33 \times 10^{-4}$ | $10^{-3}$ | $3 \times 10^{-3}$ | $9 \times 10^{-3}$ | $2.7 \times 10^{-2}$ |
| :-- | --: | --: | --: | --: | --: |
| Final loss | 3.127 | 2.977 | 2.925 | 2.948 | 3.019 |

| Weight decay | 0.0111 | 0.0333 | 0.1 | 0.3 | 0.9 |
| :-- | --: | --: | --: | --: | --: |
| Final loss | 2.948 | 2.940 | 2.925 | 2.918 | 2.954 |

| Warmup fraction | 0.00111 | 0.00333 | 0.01 | 0.03 | 0.09 |
| :-- | --: | --: | --: | --: | --: |
| Warmup steps | 10 | 31 | 93 | 281 | 843 |
| Final loss | 2.992 | 2.942 | 2.925 | 2.926 | 2.922 |

| Batch size | 16 | 32 | 64 | 128 | 256 |
| :-- | --: | --: | --: | --: | --: |
| Final loss | 2.928 | 2.918 | 2.925 | 2.951 | 3.020 |

### Learning rate matters most

The learning rate sweep spans 0.20 in final loss, the widest of the four. The
sweeps do not all cover the same range, so the fitted curvature $a$ is the
fairer comparison. It is 0.162 for learning rate, 0.138 for batch size, 0.034
for warmup and 0.030 for weight decay.

My explanation is that Adam divides each gradient coordinate by a running
estimate of its size. A step can therefore move a weight by at most about the
learning rate, however large the gradient is. The learning rate caps the size
of every update, and the number of updates is fixed, so it sets how much the
weights can change over the run.

The two sides of the sweep fail in different ways. At $3.33 \times 10^{-4}$ the
curve is still far above the others at the end, and it is slowing down as the
learning rate decays. The run ends before the weights have moved far enough. At
$2.7 \times 10^{-2}$ the curve is the worst of the five for most of training,
then drops quickly near the end.

That late recovery may make the sweep asymmetric. A learning rate $3\times$
above the default costs 0.023, and one $3\times$ below costs 0.052. At
$9\times$ the costs are 0.094 and 0.202. One reading is that the schedule
forgives a high learning rate. Linear decay takes a run that starts too high
through every smaller learning rate on its way to zero, so it spends some time
at a good one. A run that starts too low never sees a higher learning rate.
Part of the asymmetry is the grid, though. The quadratic fit puts the best
learning rate near $4.3 \times 10^{-3}$, above the default, so the high side of
the sweep starts closer to the optimum.

### Weight decay is nearly flat

Weight decay moves the final loss by 0.04 across the whole sweep. In a single
pass over the data there is no memorization for it to prevent, so whatever it
does here it does by controlling the size of the weights. I have two candidate
reasons for why that has so little effect.

The first is that small values barely act within 9,375 steps. Decay applies to
the weights of the linear layers. At step $t$, AdamW multiplies each of them by
$1 - \eta_t \lambda$, where $\eta_t$ is the learning rate at that step and
$\lambda$ is the weight decay. The gradient updates push the other way. If they
did not, a weight would end the run scaled by

$$
\prod_t (1 - \eta_t \lambda) \approx \exp\Big(-\lambda \sum_t \eta_t\Big).
$$

The sum $\sum_t \eta_t$ is the area under the learning rate schedule. The
schedule is a triangle with base 9,375 and height $3 \times 10^{-3}$, so its
area is about 14.06.

| Weight decay $\lambda$ | 0.0111 | 0.0333 | 0.1 | 0.3 | 0.9 |
| :-- | --: | --: | --: | --: | --: |
| Shrink factor $\exp(-14.06\lambda)$ | 0.86 | 0.63 | 0.25 | 0.015 | $3 \times 10^{-6}$ |

This factor measures how strong decay is over the run. It is the most that
decay could shrink a weight. At $\lambda = 0.0111$, decay could remove at most
14% of a weight over the entire run. The left half of the sweep compares almost no decay
with a little decay.

The second is that large values fade out with the schedule. The best run,
$\lambda = 0.3$, has a shrink factor of 0.015, so the gradient updates must be
rebuilding the weights as decay removes them. The decay applied at each step is
$\eta_t \lambda$, which falls to zero together with the learning rate. The
$\lambda = 0.9$ curve sits far above the others for most of training and then
closes to within 0.029 of the default by the end.

### Too little warmup hurts, and more costs nothing visible

With 10 warmup steps the final loss is 0.067 worse than the default. Its curve
falls behind early and stays behind for the whole run. My guess is that this
comes from Adam's running estimate of gradient size. The recipe uses
$\beta_2 = 0.95$, so that estimate averages over about $1/(1 - \beta_2) = 20$
steps. A 10-step warmup reaches the full learning rate while the estimate is
still built from a handful of gradients, on a freshly initialized network.

Going from 93 warmup steps to 843 changes the loss by 0.003. I expected longer
warmup to cost something, since it delays the peak learning rate. It does not
reduce the area under the schedule, though. A triangle with base 9,375 and
height $3 \times 10^{-3}$ has the same area wherever its peak sits. The area is
14.06 for all five warmup settings. Longer warmup only moves learning rate from
the start of training to later in it.

### Larger batches hurt at a fixed learning rate

Every run sees the same tokens, so a batch $4\times$ larger takes $4\times$
fewer steps.

| Batch size | 16 | 32 | 64 | 128 | 256 |
| :-- | --: | --: | --: | --: | --: |
| Optimizer steps | 37,500 | 18,750 | 9,375 | 4,687 | 2,343 |
| Warmup steps | 375 | 187 | 93 | 46 | 23 |
| Area under schedule | 56.3 | 28.1 | 14.1 | 7.0 | 3.5 |

My hypothesis is that the lost steps are what hurts. A larger batch gives a less
noisy gradient, but the learning rate still caps each update. With the learning
rate fixed, $4\times$ fewer updates cannot be made up by cleaner ones.

A rough check is to treat the area under the schedule as the only thing that
matters. Batch size 256 has the same area as batch size 64 with the learning
rate cut by $4\times$, to $7.5 \times 10^{-4}$. Interpolating the learning rate
sweep to that value, linearly in log learning rate, predicts 3.016. Batch size
256 scored 3.020. The same check for batch size 128 predicts 2.958 against a
measured 2.951.

The check has three weaknesses.

1. Warmup is a fraction of the run, so batch size 256 also gets only 23 warmup
   steps. The warmup sweep shows that short warmups cost loss. This sweep cannot
   separate the two effects.
2. The check fails in the other direction. It predicts 2.940 and 2.967 for
   batch sizes 32 and 16, and they scored 2.918 and 2.928. Area is a crude
   proxy. Many small steps do better than the same area spent on fewer, larger
   steps.
3. The learning rate was not retuned. A larger batch may only pay off at a
   higher learning rate, and this sweep holds it fixed.
