---
layout: summary
title: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
giscus_comments: true
bib_id: 1910.13461v1
---

### Three Important Things

#### 1. Bidirectional and Auto-Regressive Transformers

BART has an architecture comprising a bi-directional encoder (similar to BERT)
and an autoregressive decoder (similar to GPT), which gives it its name.

#### 2. Corrupting Text with Arbitrary Noising Functions

BART generalizes the corruption scheme of masked language modeling (MLM)
introduced in [BERT](https://arxiv.org/abs/1810.04805v2) by introducing the
following transformations for corruption during pre-training:

1. Token Masking - similar to BERT, where random tokens are masked with the `[MASK]` token
2. Token Deletion - tokens are deleted from random positions, and the model must figure out what these positions are
3. Text Infilling - a generalization of token masking, where instead of a single
   token, a span of tokens with length drawn from a Poisson distribution (possibly
   0-length) are deleted and replaced with a single `[MASK]` token. The model must
   predict the number of missing tokens.
4. Sentence Permutation - sentences delineated by full stops are randomly shuffled, and the model has to predict
   the original permutation
5. Document Rotation - the document is rotated uniformly randomly, and the model
   has to predict the original starting token.

These data corruption schemes can be combined via composition during pre-training.

#### 3. Text Infilling Produces the Best Results

When each of these pre-training objectives above is trained in isolation on a
BART base, text infilling results in the strongest performance.

### Most Glaring Deficiency

In general, I thought the paper did not introduce many novel ideas, though the
results from their experiments are still useful for guiding incremental
performance gains in large language models.

It would have also been interesting to see the performance of BART with all of
the transformations applied, or even at least with more variations of the
transformations, since the only compound transformations that were performed
were with text infilling and sentence shuffling, which resulted in the best
performance for a third of the tasks that it was evaluated on. This could
indicate that there may still be room for further gains from combining more of
the proposed pre-training tasks.

### Conclusions for Future Work

Token masking, and more generally token infilling is essential to the
performance of language models on language tasks.
