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
I sweep learning rate, weight decay, warmup and batch size one at a time, and
then in pairs.

## Setup

Every run trains `d8`, the course's default model. It is a decoder-only
transformer with 8 layers and hidden size 512. The runs are compute-matched.
The model is the same in every run, and each run makes a single pass over the
same 600,000 sequences of 1,024 tokens.

The default recipe is AdamW with peak learning rate $3 \times 10^{-3}$, weight
decay $0.1$ and batch size 64, which gives 9,375 optimizer steps. The learning
rate warms up linearly from zero over the first 1% of steps, then decays
linearly to zero. The default run ends at a validation loss of 2.925.

Each point in the plots below is a single run with one seed.

## Sweeping one hyperparameter at a time

Each sweep changes one hyperparameter and leaves the rest at their defaults.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_sweeps.webp"
    class="z-depth-1"
    zoomable=true
    num=1
    alt="Four panels of final validation loss against learning rate, weight decay, warmup fraction and batch size, on a shared vertical axis."
    caption="Final validation loss for each sweep, on a shared vertical axis. Learning rate moves the loss the most."
%}

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_sweep_curves.webp"
    class="z-depth-1"
    zoomable=true
    num=2
    alt="Four panels of validation loss over training tokens, one line per run, for the learning rate, weight decay, warmup and batch size sweeps."
    caption="Validation loss during training for the same runs."
%}

1. **Learning rate is the most important hyperparameter to tune.** Its sweep
   moves the final loss more than any other.
2. **Weight decay is relatively flat.** An $81\times$ range moves the final loss
   by about 0.04.
3. **Too little warmup hurts, and extra warmup costs almost nothing.**
4. **Larger batches hurt.** Every run sees the same tokens, so a larger batch
   takes fewer optimizer steps.

## Changing two hyperparameters together

### Batch size and learning rate

**The best learning rate did not clearly move with batch size between 16 and
256.** The square-root rule scales the learning rate with the square root of
the batch size, and the linear rule scales it in proportion. Neither is
supported. A likely explanation is that the gradient
noise is already small at these batch sizes, but I did not measure it.

The test was confounded by warmup, which the next section covers. With warmup
held at 93 steps, batch size 256 slightly prefers the higher learning rate
(2.966 at $6 \times 10^{-3}$ against 2.977 at $3 \times 10^{-3}$).

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_pair1_bs_lr.webp"
    class="z-depth-1"
    zoomable=true
    num=3
    alt="Final validation loss against learning rate for batch sizes 16, 64 and 256, plotted against the raw learning rate and against the learning rate divided by the square root of batch size over 64."
    caption="Left: final validation loss against learning rate at three batch sizes. Right: the same runs with the learning rate divided by $\sqrt{\text{batch size}/64}$. The square-root rule predicts that the minima line up on the right, and they do not."
%}

### Batch size and warmup

**Warmup is needed in steps, not as a percentage of the run.** Warmup is set as
a percentage of all steps, and a larger batch has fewer steps. At 1% warmup,
batch size 64 gets 93 warmup steps and batch size 256 gets 23. Giving batch
size 256 the same 93 steps recovered 0.043 of its 0.095 penalty.

The usual explanation for why steps matter is that the first updates are
unreliable. Adam's running gradient statistics are still noisy and the loss
surface is sharp early on, and both settle over a number of steps whatever the
batch size. I did not test that explanation.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_pair2_bs_warmup.webp"
    class="z-depth-1"
    zoomable=true
    num=4
    alt="Final validation loss for batch sizes 64 and 256, against warmup as a percentage of all steps on the left and against warmup steps on the right."
    caption="Left: final validation loss against warmup as a percentage of all steps. Right: the same runs against warmup steps. Batch size 256 improves when it gets as many warmup steps as batch size 64."
%}

### Warmup and learning rate

**What matters is how fast the learning rate climbs.** Warmup raises the
learning rate in a straight line from zero to its peak, so the slope of the ramp
is the peak learning rate divided by the number of warmup steps. A higher peak
over the same number of steps is a steeper climb, and the steep climb is what
costs loss. Raising the learning rate $3\times$ needs $3\times$ the warmup
steps.

The default ramp has a slope of about $3 \times 10^{-5}$ per step, and ramps at
or below that slope cost nothing. A slope near $1 \times 10^{-4}$ costs about
0.02, and a slope near $3 \times 10^{-4}$ costs about 0.07. The one exception is
learning rate $1 \times 10^{-3}$ over 10 steps, which has a slope of
$1 \times 10^{-4}$ and paid nothing.

The batch size result above is the same effect. Batch size 256 at 1% warmup has
a slope of $0.003 / 23 = 1.3 \times 10^{-4}$, which is in the costly range.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_pair3_warmup_lr.webp"
    class="z-depth-1"
    zoomable=true
    num=5
    alt="Final validation loss at four learning rates, against warmup steps on the left and against peak learning rate divided by warmup steps on the right."
    caption="Left: final validation loss against warmup steps at four learning rates. Right: the same runs against the ramp slope. Loss rises once the slope passes about $3 \times 10^{-5}$ per step."
%}

### Learning rate and weight decay

**Keep learning rate $\times$ weight decay fixed when you change the learning
rate.** The best product was $9 \times 10^{-4}$ at learning rates
$1 \times 10^{-3}$, $3 \times 10^{-3}$ and $9 \times 10^{-3}$. This was only
shown at batch size 64. Across batch sizes, use the rule in the next section.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_pair4_lr_wd.webp"
    class="z-depth-1"
    zoomable=true
    num=6
    alt="Final validation loss at three learning rates, against weight decay on the left and against learning rate times weight decay on the right."
    caption="Left: final validation loss against weight decay at three learning rates. Right: the same runs against learning rate $\times$ weight decay. Each learning rate has its lowest loss at the same product."
%}

### Batch size and weight decay

**A larger batch wants more weight decay.** Each step shrinks the weights by a
fraction learning rate $\times$ weight decay, so the total shrinkage over a run
scales with learning rate $\times$ weight decay $\times$ steps. Batch size 256
takes $4\times$ fewer steps, so at the same settings it gets $4\times$ less
total decay.

The candidate rule is to keep learning rate $\times$ weight decay $\times$ steps
fixed. At the best batch size 64 run, which used weight decay 0.3, it is
$0.003 \times 0.3 \times 9375 \approx 8.4$. For batch size 256, with a quarter
of the steps, that predicts a best weight decay of about $1.2$.

This fits the runs but is not confirmed. Batch size 256 improved from 3.020 to
2.991 to 2.973 as weight decay went from 0.1 to 0.3 to 0.9, and the loss was
still falling at 0.9. One run above 0.9 would show whether it turns up near 1.2.

The learning rate and weight decay rule above is the special case where the
step count does not change.

{% include figure.liquid
    path="/assets/img/posts/hp_tuning/p1_pair5_bs_wd.webp"
    class="z-depth-1"
    zoomable=true
    num=7
    alt="Final validation loss for batch sizes 64 and 256, against weight decay on the left and against peak learning rate times weight decay times optimizer steps on the right."
    caption="Left: final validation loss against weight decay at two batch sizes. Right: the same runs against learning rate $\times$ weight decay $\times$ steps. Batch size 256 is still improving at weight decay 0.9."
%}

### The pattern across all five

Changing the batch size at a fixed token budget changes the step count. That
silently changes the number of warmup steps and the total weight decay. Most of
the apparent penalty for batch size 256 came from those two side effects.
