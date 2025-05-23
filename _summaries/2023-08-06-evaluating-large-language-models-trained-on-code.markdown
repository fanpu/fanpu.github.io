---
layout: summary
title: "Evaluating Large Language Models Trained on Code (Codex)"
giscus_comments: true
bib_id: 2107.03374v2
---

### Three Important Things

#### 1. Codex

Codex is a model fine-tuned from GPT on code publicly available from Github.
Given a docstring, it is capable of generating Python code that implements
what is described in the docstring.

The authors use the pass@k metric to evaluate the functional correctness of the model
on different problems. In the pass@k metric, the model generates $$k$$ different code
samples for a problem, and $$c$$ is the number of outputs that passes all test cases
for that problem. They then use the following estimator:

$$pass@k \coloneqq \mathbb{E}_{\text{Problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right].$$

They showed that Codex performed better than GPT on their own HumanEval dataset. The HumanEval
dataset is their own newly-developed dataset of problems and solutions, to avoid testing
on problems that the model may have already seen in training.

#### 2. Codex-S

To further improve performance, they applied supervised fine-tuning
on Codex to obtain Codex-S. The data used for fine-tuning came from two sources:

1. Problems from competitive programming and interview preparation websites,
   complete with test cases
2. Problems obtained from continuous integration setups in open-source projects.
   This is achieved by profiling the inputs and outputs of functions called during
   the integration tests, and collecting them.

#### 3. Docstring Generation

The authors also consider the reverse problem of generating docstrings from a piece of code.
This helps to improve the explainability of the code generated and helps with AI safety.
This was trained by grading the sample docstrings manually by hand, and training
was therefore limited in scale with only 10 samples per problem. This yielded a pass
rate for docstring generation that was comparable to the pass rate for code generation
by Codex-S.

The authors also experimented with choosing the code sample generated by Codex-S that
maximizes the back-translation probability as evaluated by Codex-D, but this performed
worse than just using the log-probabilities of Codex-S in the original setting.

### Most Glaring Deficiency

Major limitations of Codex include remembering variable binding, and docstrings
with long chains of operations. The former could possibly be rectified with
a static analysis framework with feedback in a RLHF-manner, which helps to resolve many
issues with code that almost compiles. However, approaches to resolve either of them
were neither discussed nor attempted.

### Conclusions for Future Work

Language models can also be successfully adapted to code generation. While the
technology is nascent, it is very promising and there remains much future work
to be done to improve on many of its most glaring limitations.

The many societal and economic considerations mentioned in the paper are also
worth keeping in mind when making use of such technologies, such as the bias
towards using popular libraries suggested by Codex, and its tendency to suggest
weak security parameters and insecure code that exist in its training data.
