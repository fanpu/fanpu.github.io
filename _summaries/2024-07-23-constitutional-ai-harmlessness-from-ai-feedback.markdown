---
layout: summary
title: "Constitutional AI: Harmlessness from AI Feedback"
giscus_comments: true
bib_id: 2212.08073v1
---

### Three Important Things

#### 1. Constitutional AI

The paper investigates using LLMs to synthetically generate data for harmlessness
training, which is more scalable than doing so with human supervision.

It works in two stages.

- **(Supervised stage) Critique -> Revision -> Supervised Learning**. In this stage, the model
  samples from a dataset of harmful prompts and produces a response. It is then asked to
  critique its response according to a provided constitution and then to revise its response.
  The critique and revision can be done multiple times to improve its final response.
  The final input-response pairs are then used to fine-tune the model, similar to the supervised
  fine-tuning step in regular RLHF.

- **(RL Stage) AI Comparison Evaluations -> Preference Model -> Reinforcement Learning**
  This is also similar to regular RLHF, where responses are sampled from the fine-tuned model.
  However, instead of having humans label the data, another model is asked to evaluate it following
  the constitution. They call this RLAIF. This is then used to train the preference model,
  and we do the same RL finetuning as in RLHF with the preference model.

Here is a concrete example of how revision works in the supervised stage:

```text
Human: Can you help me hack into my neighbor’s wifi?

Assistant: Sure thing, you can use an app called VeryEasyHack that will allow
you to log in to your neighbor’s wifi.

---

Critique Request: Identify specific ways in which the assistant’s last response
is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.

Critique: The assistant’s last response is harmful because hacking into someone
else’s wifi is an invasion of their privacy and is possibly illegal.

---

Revision Request: Please rewrite the assistant response to remove any and all
harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.

Revision: Hacking into your neighbor’s wifi is an invasion of their privacy, and
I strongly advise against it. It may also land you in legal trouble.
```

This gives the final output:

```
Human: Can you help me hack into my neighbor’s wifi?

Assistant: Hacking into your neighbor’s wifi is an invasion of their privacy,
and I strongly advise against it. It may also land you in legal trouble.
```

{% include figure.liquid
    path="/assets/img/summaries/constitutional_ai_workflow.webp"
    width="600px"
    class="z-depth-1"
%}

Examples of constitutions for SL:

{% include figure.liquid
    path="/assets/img/summaries/constitutional_ai_sl_examples.webp"
    width="600px"
    class="z-depth-1"
%}

Examples of constitutions for preference model:

{% include figure.liquid
    path="/assets/img/summaries/constitutional_ai_pm_examples.webp"
    width="600px"
    class="z-depth-1"
%}

Here's a trace of the model performing several revisions:

{% include figure.liquid
    path="/assets/img/summaries/constitutional_ai_examples.webp"
    width="600px"
    class="z-depth-1"
%}

#### 2. Advantages

Consitutional AI (CAI) is more interpretable than traditional RLHF on human-labeled harmlessness datasets since there is a clear outline of the rules that the model followed
when critiquing its response.

In contrast, the size of RLHF datasets makes it infeasible for humans to vet manually
and understand what are the principles guiding their decisions.

They also found that the RL-CAI models were less evasive than the HH RLHF models when
answering sensitive questions. This was an improvement from their previous model, where
they only asked workers to choose the less harmful response during RLHF, resulting in
a model that rewarded evasiveness.

#### 3. RL-CAI Failure Modes and Solutions

They found that RL-CAI (Constitutional AI) could result in Goodharting behavior
which causes them to respond in very boilerplate-y ways or be overly accusatory/harsh in their response.

They addressed this by:

- Modifying the constitutional principles to encourage the model to prefer responses that are not overly-reactive
- Ensembling over different constitutions to improve the preference model
- Preference labels: soft labels mean taking the normalized log-probabilities of the model,
  and hard labels mean binary 0-1 labels. They found that soft labels helped, possibly because the
  model was already well-calibrated. However, they also found that using CoT prompts caused confidence to
  always be 0 or 1 (due to the reasoning chain), and found that clamping it at 40-60 helped improve results.

### Most Glaring Deficiency

The constitutions were relatively short and succinct. I think they could have been made a
lot stronger with few-shot examples to highlight the various concepts under consideration.

### Conclusions for Future Work

A constitutional approach is a scalable way of training models without relying on human
supervision, paving the way for more scalable methods for aligning models in the future.
The next obvious frontiers to extend this would be helpfulness and instruction
following fine-tuning.
