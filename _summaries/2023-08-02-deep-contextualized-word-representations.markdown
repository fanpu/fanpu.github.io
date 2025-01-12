---
layout: summary
title: "Deep contextualized word representations (ELMo)"
giscus_comments: true
bib_id: 1802.05365v2
---

### Three Important Things

#### 1. Context-aware Word Embeddings

Most word embedding techniques only embed single words without regard to their
context, which could affect the performance of downstream tasks.
ELMo (Embeddings from Language Models) instead embeds words as a function of the
entire input sentence using a two-layer bidirectional language model (biLM).

#### 2. Using All biLM Layers For Representation

Previous work only used the last layer of the biLM.
However, it was empirically observed that using all layers
(by a task-specific weighing of each layer that decays geometrically)
achieved better results.

This is because different layers represent different
information and so including all biLM layers helps
in downstream tasks.

#### 3. Better Sample Efficiency

It was observed empirically that using ELMo can improve
sample efficiency significantly.

### Most Glaring Deficiency

Reasons for why ELMO achieves better sample efficiency were not discussed or
hypothesized, unlike why utilizing ELMo obtained better results on many NLP
tasks than state-of-the-art. Postulating some hypotheses to answer this question
for future research directions would have been helpful.

### Conclusions for Future Work

Context-aware representations can improve performance. Even intermediate layers
of a final representation could be useful for downstream tasks to take advantage
of.
