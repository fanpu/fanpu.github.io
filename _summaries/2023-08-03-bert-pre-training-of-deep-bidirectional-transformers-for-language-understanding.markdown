---
layout: summary
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
giscus_comments: true
bib_id: 1810.04805v2
---

### Three Important Things

#### 1. Masked Language Modeling

A challenge of training the deep bidirectional model is that it cannot be done
either left-to-right or right-to-left, since the bidirectional nature of the
Transformer will allow it to see the tokens in advance.

To resolve this, the authors introduced "masked LM" (MLM), where tokens are
masked at random and the task is to predict the masked tokens. This is
the first task used in their pre-training phase, the other being next
sentence prediction (NSP).

#### 2. Next Sentence Prediction

To ensure that the language model understands the relationships between sentences,
it is also pre-trained on next sentence prediction tasks.

#### 3. Robust on Both Fine-tuning and Feature Extraction

Ablation studies shows that BERT performs competitively with both fine-tuning
on downstream tasks, and on using frozen feature extractions with a biLSTM layer
on top.

### Most Glaring Deficiency

In the setup for next sentence prediction, it was unclear to me how
the embeddings for `IsNext` and `NotNext` were chosen and what
the loss function is such that $$C$$ has some useful representation.

This is particularly pertinent because of the importance of $$C$$ for
downstream tasks.

### Conclusions for Future Work

To take advantage of bi-directionality in sequential data, we can apply ideas
similar to masked language modeling even for data of other modalities.
