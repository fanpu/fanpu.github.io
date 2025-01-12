---
layout: summary
title: "Transformers in Speech Processing: A Survey"
giscus_comments: true
bib_id: 2303.11607v1
---

### Three Important Things

#### 1. Limited Gains

Applying transformers to speech processing tasks has shown limited or inconclusive gains compared to
recurrent/convolutional neural network architectures. This is owing to the many limitations
of transformers, such as requiring more training
data due to its lack of inductive bias,
when such speech data is already very limited.

#### 2. Challenges of Transformers in Speech Processing

A major problem is that audio data has longer sequence lengths (i.e. a few
thousand frames in Speech Emotion Recognition (SER)) and are less
information-dense, and hence because self-attention has complexity quadratic to
the number of frames, this results in very expensive runtime.

#### 3. Lack of Standardized Benchmark

Speech processing lacks a standardized benchmark like
[GLUE](https://gluebenchmark.com/) for evaluating spoken dialogue systems. As a
result, there is large diversity in the datasets used to evaluate such systems,
making comparisons and establishing the state-of-the-art difficult.

### Most Glaring Deficiency

It is generally difficult to criticize a survey paper, but one pertinent
objection would be the lack of a summary and discussion of existing datasets and
tasks used for evaluating spoken dialogue systems. This would help to provide
some examples on the type of tasks these systems are being evaluated on in
practice.

### Conclusions for Future Work

There could possibly be a new architecture more suited for speech data, which
has a better inductive bias than transformers and can therefore be trained with much
less data.
