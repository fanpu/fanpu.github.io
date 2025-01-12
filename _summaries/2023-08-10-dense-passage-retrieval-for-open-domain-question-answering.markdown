---
layout: summary
title: "Dense Passage Retrieval for Open-Domain Question Answering"
giscus_comments: true
bib_id: 2004.04906v3
---

### Three Important Things

#### 1. Dense Passage Retriever

The authors introduce the dense passage retriever (DPR),
which consists of two encoders:

- $$E_P(\cdot)$$ for embedding passages,
- $$E_Q(\cdot)$$ for embedding queries.

In practice, both $$E_P(\cdot)$$ and $$E_Q(\cdot)$$ are the `[CLS]` tokens from separate BERT encoders.
They are then
fine-tuned on question-passage pairs to minimize
the following loss function:

$$
L\left(q_i, p_i^+, p_{i,1}^-, \cdots, p_{i,n}^-\right)
= -\log \frac{e^{\text{sim}(q_i, p_i^+)}}
{e^{\text{sim}(q_i, p_i^+)} + \sum_{j=1}^n e^{\text{sim}(q_i, p_{i,j}^-)}}
$$

where

- $$q_i$$ is the question,
- $$p_i^+$$ is the relevant positive passage,
- $$p_{i,j}^-$$ is one of the $$n$$ irrelevant negative passages,
- $$\text{sim}$$ is the dot product of the query and passage
  encoders, i.e $$\text{sim}(q,p) = E_Q(q)^T E_P(p)$$.

The authors showed that DPR significantly improved
on [BM25](https://dl.acm.org/doi/10.1561/1500000019)
for most benchmarks, considered the state-of-the-art in QA systems at the
time. The only exception was the SQuAD dataset, due to the syntactic similarity
between the crafted questions and the original passage by the labelers.

#### 2. In-batch Negatives

The authors used the trick of [in-batch negatives](https://dl.acm.org/doi/10.5555/2018936.2018965) by using the positive
passages from all other samples in the batch as the negative sample for a
particular question. This resulted in better sample efficiency and was also faster
to train due to memory efficiency.

#### 3. Robustness of Methodology

The authors showed via ablation studies that DPR
generally maintains its performance even when different choices of negative
passages were used for training. They used the following choices of passages:

1. Random: any randomly chosen passage,
2. BM25: the top-scoring BM25 passage that does not contain the answer,
3. Gold: passages that are the answers to other question-passage pairs.

They also showed that on top of dot product distance,
L2 distance also performed well, though cosine
distance was slightly worse.

### Most Glaring Deficiency

It would have been beneficial to also experiment and report the performance
using a DPR retriever that was not fine-tuned on any question-passage pairs at
all. This is because in practice there may be a lack of available data for
specialized QA systems, and therefore it would be very promising if BERT is
already capable of great baseline performance even without fine-tuning.

### Conclusions for Future Work

Dense representations are a promising new direction for information retrieval
for QA systems, as opposed to traditional sparse representations. This is helped
in part by recent work on efficient maximum inner product search (MIPS)
algorithms.

This also further illustrates the versatility of the BERT `[CLS]` token for
various downstream tasks.
