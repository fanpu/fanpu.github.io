---
layout: summary
title: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
giscus_comments: true
bib_id: 2310.03714v1
---

### Important Things

#### 1. DSPy

From the authors of [ColBERT](
{% link _summaries/2024-02-22-colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert.markdown %}),
DSPy is a framework for improving a language model (LM) pipeline. It can be thought
of as an improvement on [LangChain](https://www.langchain.com/), where instead of just allowing application developers to compose
different parts of a LM pipeline together, it is also capable of optimizing the prompts within each component.

DSPy stands for "Demonstrate–Search–Predict".

#### 2. Components

There are 3 main components in DSPy.

1. Signature: the signature defines the input/outputs of a module, which represents one or many LLM calls.
   An example would be like `question -> answer`:

   {% include figure_simple.liquid
       path="/assets/img/summaries/dspy_signature.webp"
       width="600px"
       class="z-depth-1"
   %}

2. Modules: modules are the main building blocks that have a signature defined on them.
   Examples of modules are `Predict` (as given above), `ChainOfThought`, `MultiChainComparison`, `ReAct`.

3. Teleprompters: these are what they call the optimizers that will optimize the prompts for a given module.
   In order to use a teleprompter, there must be an evaluation dataset and a metric to measure model
   performance (i.e exact match or F1).

   {% include figure_simple.liquid
       path="/assets/img/summaries/dspy_teleprompter.webp"
       width="600px"
       class="z-depth-1"
   %}

   In the example above, the teleprompter is used to generate few-shot examples that will help the model
   answer the question more effectively.

With this design, modules can be composed together and optimized by teleprompters.

#### 3. Example of a Teleprompter

They included a few examples of pseudocode of teleprompters in the appendix.
Let's look at the one for `BootstrapFewShot`:

{% include figure.liquid
    path="/assets/img/summaries/dspy_teleprompter_example.webp"
    width="600px"
    class="z-depth-1"
%}

The goal of this teleprompter is to come up with some few-shot examples.
This is done as follows:

1. Take examples from the train set, and have a _teacher_ (supposedly a
   DSPy program that doesn't perform too poorly on the task) generate
   answers for the problems. If a teacher doesn't exist, use the student program.
2. If the _teacher_ gets the answer right based on some metrics, this is accepted
   as a new few-shot example.

### Most Glaring Deficiency

I expected the paper to go into the details of the algorithms used for optimizing the
prompts used in the pipeline, but none were provided other than the very simple
few-shot example generation ones. That would have been the most interesting part
of the DSPy pipeline as a reader, which made the paper a somewhat disappointing
read.

### Conclusions for Future Work

Abstract the "how" away from the "what" to build composable programs. In this case,
the functionality of how the LLM modules work based on their prompts was abstracted away
from the goals of what each module is supposed to achieve.
