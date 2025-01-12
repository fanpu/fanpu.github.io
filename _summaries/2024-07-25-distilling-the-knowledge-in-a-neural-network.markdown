---
layout: summary
title: "Distilling the Knowledge in a Neural Network"
giscus_comments: true
bib_id: 1503.02531v1
---

### Three Important Things

#### 1. Motivation

The paper opens with a really interesting analogy: insects begin with a larval
form optimized for extracting energy and nutrients from the environment, before
metamorphosing into a completely adult form optimized for traveling and
reproduction.

In a similar vein, for machine learning models we have access to plentiful compute and time during the training, but become constrained by different
considerations like latency during inference.
Given the different requirements and affordances of both phases, why should we
expect to simply use the same model for both of them?

To address this, the paper introduces the idea of distillation, to transfer
the knowledge of a very large/cumbersome model (possibly an ensemble of models)
to a smaller one.

#### 2. Distillation

There are several ways to approach distillation.

1. A natural first approach would be to make the student model learn to predict
   the same class probabilities as the teacher model. These are known as "soft targets" (in contrast, "hard targets" are the ground-truth labels). Soft targets are "an effective way of communicating the regularities discovered by a model
   trained on all of the data to another model".
2. In practice, the correct label usually dominates the others by
   orders of magnitude. But a lot of information about the learned function can be
   gleaned from the ratios of the other labels, which can provide information on
   how one label might be confused for another and allow the model to draw more
   conclusions.

If we just use cross-entropy loss, these vital set of information will be lost
as their relative values are tiny relative to the most confident prediction and
don't contribute much to the loss.

To address this, the authors added a temperature parameter before the softmax,
and increased the temperature to a suitably high level until the targets are "soft" enough. The same temperature is used when training the small model.
The temperature is reset to 1 afterwards at test time.

Note that changing temperature also requires the gradient update of the
cross-entropy loss of the softmax predictions to be scaled appropriately by
$$(1/T^2)$$ (via the chain rule).

The paper shows that the gradient of the cross-entropy loss
is approximately $$\frac{1}{NT^2}(z_i - v_i)$$ using a first-order Taylor
approximation, which holds when temperature is high. By integrating this,
the loss is proportionate to $$\frac{(z_i - v_i)^2}{2}$$, showing that
the loss function is essentially matching the logits.

#### 3. Ensembles of Specialists

One way they applied the idea of matching soft targets with temperature is as follows.

Train time:

1. Train a big generalist model the normal way
2. Use the generalist model to identify clusters of labels by using the covariance matrix of its predictions
3. For each of these clusters, train a specialist model. The specialist model
   only learns to classify the things in its cluster, and all other labels
   are lumped into a "dustbin" class

Test time:

1. Run generalist model on test case sample. Take the top $$n$$ most probable
   classes, call this set $$k$$
2. Take all the specialist classes that can handle anything in $$k$$.
   So each specialist class outputs its own logits. The goal is to optimize for the
   logits $$\mathbf{q}$$ that minimizes over the KL between the logits of the generalist model
   and $$\mathbf{q}$$
   , as well as the logits between each of the specialist classes and $$\mathbf{q}$$, i.e:
   $$K L\left(\mathbf{p}^g, \mathbf{q}\right)+\sum_{m \in A_k} K L\left(\mathbf{p}^m, \mathbf{q}\right)$$
3. Take the most probable class in $$\mathbf{q}$$ as the prediction

### Most Glaring Deficiency

The paper did not (and acknowledged it did not) report on the results of
actually distilling the outputs of the "cumbersome" ensemble of specialist models back into a small model, which felt like one of the main selling points
that they were making.

### Conclusions for Future Work

Distillation is now a pretty popular way to transfer the knowledge of larger/more cumbersome models into smaller ones, allowing for wins in latency and cost
during live inference.
