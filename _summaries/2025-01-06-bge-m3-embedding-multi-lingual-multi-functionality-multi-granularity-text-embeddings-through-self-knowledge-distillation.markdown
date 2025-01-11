---
layout: summary
title: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
giscus_comments: true
bib_id: 2402.03216v4
hidden: true
---

### Three Important Things

#### 1. Foo

Unsupervised data curation: title-body, title-abstract, instruction-output

Synthetic data: choose paragraphs, use GPT-3.5 to generate questions, added to fine-tuning data

Their embedding model can support all 3 common retrieval functions:
1. Dense retrieval
2. Lexical (sparse) retrieval - this is not in the tf-idf/BM25 sense, but rather passing the text encoder outputs $$H_q[i]$$ through a projection $$W_{lex}$$ & ReLU to get $$w_{q_t} \leftarrow \textrm{ReLU}(W^T_{lex} H_q[i])$$, and then taking the sum of the product of the activations between each term that appears in both the query and passage (using the maximum value if there are duplicates)
3. Multi-vector retrieval - i.e ColBERT style

The final score is a weighted sum of the 3 relevance scores above.

#### 2. Bar

#### 3. Baz

### Most Glaring Deficiency

### Conclusions for Future Work
