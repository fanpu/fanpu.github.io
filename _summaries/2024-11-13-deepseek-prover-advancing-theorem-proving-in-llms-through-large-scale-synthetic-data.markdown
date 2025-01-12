---
layout: summary
title: "DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data"
giscus_comments: true
bib_id: 2405.14333v1
published: false
---

### Three Important Things

#### 1. Training a Theorem Prover

Their goal is to train a theorem-proving LLM using synthetically generated data.

They started from the DeepSeekMath-Base 7B model, and performed SFT on it with
Lean tasks with backtranslated instructions. This was necessary to "warm up" the
model for the task distribution, as initial performance was very poor.

Due to the scarcity of Lean proofs, they had to generate their own synthetic
data. This was achieved by taking existing proofs, and having LLMs attempt
to formalize the statements in Lean, which they call "autoformalization".

As this is not a completely reliable process, they applied two kinds of filters.

The first is essentially a model grader to classify the quality of the formal
statement, and how interesting it would be for the mathematical community.
Poor quality statements are excluded.

The second is to address the fact that many of the generated formalized
statements are false/nonsensical. As such, they discarded statements where they
can prove `False` from, as it indicates inconsistent assumptions in the
statement.

This gave them a new dataset of over 700k formal statements to work with.

#### 2. Challenges in Auto-formalization

#### 3. Baz

### Most Glaring Deficiency

### Conclusions for Future Work
