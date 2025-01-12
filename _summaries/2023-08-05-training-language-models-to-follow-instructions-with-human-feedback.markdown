---
layout: summary
title: "Training language models to follow instructions with human feedback (InstructGPT)"
giscus_comments: true
bib_id: 2203.02155v1
---

### Three Important Things

#### 1. Reinforcement Learning with Human Feedback

InstructGPT uses a technique called reinforcement learning with human feedback (RLHF),
trained in the following manner:

1. A supervised fine-tuning (SFT) model is trained by fine-tuning GPT-3 on
   a dataset of prompts and their desired outputs, which were done by human labelers
2. A reward model (RM) is trained by fine-tuning the SFT model without the final
   unembedding layer as follows: given a prompt and several (4-8) outputs to the
   prompt, a human labeler ranks the outputs, and the reward model learns to
   minimize a loss function that encourages it to output a score for each
   output that agrees with the human rankings
3. A reinforcement learning (RL) model is trained by fine-tuning the SFT model
   using proximal policy optimization (PPO). PPO is a gradient update method
   that also penalizes the Kullback-Leibler (KL) distance between the original and
   updated distributions, hence preventing updates that can cause the policy to
   veer off too much. The RL environment is a single-episode bandit environment,
   where given a prompt, the policy outputs a response, with the reward determined
   by the reward model. This is the InstructGPT model.

#### 2. Alignment

To ensure that the model is aligned with human preferences and does not produce
toxic outputs, the authors use the working definition that alignment means that
the model is helpful, honest, and harmless. The contracted labelers were told
to prioritize truthfulness and harmlessness over helpfulness when coming up with
outputs, and when ranking them.

There was also extensive discussion on the importance of understanding whose
preferences the model is aligning to.

#### 3. Generalization Of "Following Instructions" To Other Settings

The main goal of InstructGPT is to improve on GPT-3 by producing output
that better follows the user's intentions. Many tasks that were
outside of the supervised task used to train the SFT and RM models,
such as explaining what a piece of code does and prompts in
languages other than English, also saw better performance.
This indicates that the model has generalized the notion of
"following instructions".

### Most Glaring Deficiency

For the reward model to be robust, it also has to be exposed to many negative/toxic
outputs that are labeled by humans. Otherwise, it would fail to provide a reliable
training signal for the InstructGPT model when confronted with toxic outputs.

There was no mention of the proportion of such toxic samples used in training
the RM model, or the potential impact that they may have on the
psychology of the human labelers who may be harmed by exposure to these samples.

### Conclusions for Future Work

RLHF is a promising way of improving the alignment of machine learning models to
human preferences by overcoming the problem of the scarcity of supervised
datapoints through the use of a reward model.
However, care and transparency should always be taken towards whose preferences these are,
to avoid harming groups that are under-represented in training data.
