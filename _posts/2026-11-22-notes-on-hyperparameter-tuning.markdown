---
title: "Notes on Hyperparameter Tuning"
published: false
layout: distill
tags: [machine-learning]
cover: great_allegheny_passage_ridges.webp
cover_preview: great_allegheny_passage_ridges.webp
caption: Great Allegheny Passage, USA
class: post-template
author: fanpu
giscus_comments: true
description: ""
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
toc: true
---

These are notes from an investigation into hyperparameter tuning. They are based
on [Stanford CS 312: Deep Learning Alchemy](https://deep-learning-alchemy.github.io/).

## The model

The experiments use `d8`, the course's default model. It is a small decoder-only
transformer, trained to predict the next token on a tokenized subset of DCLM.
The course names its models by depth. A model called `dX` has $X$ layers and
hidden size $64X$, so `d8` has 8 layers and hidden size 512.

| Part           | `d8`                                              |
| -------------- | ------------------------------------------------- |
| Layers         | 8                                                 |
| Hidden size    | 512                                               |
| Attention      | 8 causal heads of dimension 64, with RoPE         |
| MLP            | SwiGLU, intermediate size 1792 ($3.5 \times 512$) |
| Normalization  | RMSNorm, applied before each attention and MLP    |
| Vocabulary     | 4096 tokens                                       |
| Context length | 1024 tokens                                       |

Each block has the usual pre-norm residual form.

$$
\begin{aligned}
h &\leftarrow h + \mathrm{Attention}(\mathrm{RMSNorm}(h)) \\
h &\leftarrow h + \mathrm{MLP}(\mathrm{RMSNorm}(h))
\end{aligned}
$$

Three details differ from a textbook transformer. The queries and keys are
normalized with RMSNorm before the attention dot product (QK norm). Every linear
layer is bias-free. The input embedding and the output head are separate
matrices, not tied.
