---
layout: summary
title: "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
giscus_comments: true
bib_id: 2212.10509v2
---

### Two Important Things

#### 1. Interleaved Retrieval guided by Chain-of-Thought

The paper's insight is to use CoT to guide retrieval, and use the retrieved
contents to then guide CoT again.

This is done as follows:

- Generate one sentence of CoT
- Use CoT sentence to retrieve additional piece of context
- Using new context, repeat the previous steps until answer is provided, or reached max number of steps

The retrieved context is ordered randomly at each step. As the LLM may output
multiple sentences of CoT each time, they just take one newly generated sentence
and drop the rest.

Here's the overall structure of the prompt:

```
Wikipedia Title: <Page Title>
<Paragraph Text>
...
Wikipedia Title: <Page Title>
<Paragraph Text>
Q: <Question>
A: <CoT-Sent-1> ... <CoT-Sent-n>
```

{% include figure.liquid
    path="/assets/img/summaries/ir_cot_workflow.webp"
    width="600px"
    class="z-depth-1"
%}

#### 2. Results

Unsurprisingly outperforms one-step retrievers and when
no retrievers were used.

{% include figure.liquid
    path="/assets/img/summaries/ir_cot_results.webp"
    width="600px"
    class="z-depth-1"
%}

They also found that IRCoT's CoT trace had fewer factual errors, and remains
effective on smaller models (0.2B to 11B).

### Most Glaring Deficiency

Marginal novelty given other techniques like self-ask that came previously.

### Conclusions for Future Work

Could use a LLM to drive querying of RAG datastores guided by CoT to resolve
complex queries.
