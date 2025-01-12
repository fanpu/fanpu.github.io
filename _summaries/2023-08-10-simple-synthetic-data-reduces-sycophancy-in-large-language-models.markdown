---
layout: summary
title: "Simple synthetic data reduces sycophancy in large language models"
giscus_comments: true
bib_id: 2308.03958v1
---

### Three Important Things

#### 1. The Problem of Sycophancy

Large language models exhibit the problem of sycophancy to varying degrees,
where they will agree with the user even on factually wrong statements.
An example from the paper is given below.

{% include figure.liquid
    path="/assets/img/summaries/sycophancy.webp"
    width="600px"
    class="z-depth-1"
%}

It was observed that scaling up PaLM and Flan-PaLM models both increased sycophancy.

#### 2. Synthetic Data to Teach Truthfulness is Independent of User's Opinion

Using existing NLP datasets, the authors generated synthetic data as follows:

1. Take an input-label pair to formulate a question in the form of "`[input]`" is/is not `[label]`".
2. Add a user opinion that either agrees or disagrees with the claim. Add additional data about the user
   to increase data diversity.
3. Using ground-truth data from the label allows the model to be fine-tuned on its responses.

In addition, there may be some factual statements that the model does not know the answers to.
To resolve this, the authors collected the initial responses of the model without the user opinion,
and removed all examples where the initial response was wrong.

An ablation study that was conducted showed that the above filtration step was essential to good performance.

#### 3. Requires Large Model Capacity

It was empirically observed that fine-tuning using the synthetic generation
method mentioned above actually results in worse performance on small models,
such as the Flan-PaLM-8B model. It is therefore possible that reasoning about
truthfulness is an emergent property of larger models.

### Most Glaring Deficiency

There are almost-duplicated figures, like Table 1 and Figure 1 in the paper,
which are showing essentially the same examples. Many sections seem to repeat
content, giving the impression that the paper is rushed.

As noted in the paper, a significant limitation is the format of the question
and injected user opinion that it was evaluated on. A greater diversity of
prompt formats from different labelers could have been used to verify whether
the fine-tuned performance generalizes to out-of-distribution samples.

### Conclusions for Future Work

Sycophancy is a real problem in language models that can result in an
echo-chamber effect, and one possible approach to slightly alleviate this would be to
explicitly teach the model that user opinion is independent of factual data,
such as by using additional synthetic data as proposed in the paper.

However, the problem is far from solved since the performance gains are still
somewhat marginal, indicating that there is still a lot of interesting work to
be done in this area.
