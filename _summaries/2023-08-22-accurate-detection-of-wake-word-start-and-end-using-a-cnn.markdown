---
layout: summary
title: "Accurate Detection of Wake Word Start and End Using a CNN"
giscus_comments: true
bib_id: 2008.03790v1
---

### Three Important Things

#### 1. Start-end Regression Model

The authors introduce two types of architectures
for wake word (WW) detection. The first is the
start-end regression model, where the input signal
is passed through several stacked convolution and pooling layers,
before forking off into two different outputs, as illustrated
in the diagram below.

{% include figure.liquid
    path="/assets/img/summaries/ww-start-end-regression.webp"
    width="600px"
    class="z-depth-1"
%}

The first set of outputs is the probability that a wake word exists,
and the second set of outputs is the start and end offsets
of the wake word, normalized such that $$[0,1]$$ represents
the window of the input.

The two outputs share most of the same backbone in the network, due to the
belief that the network has learned useful representations that will help with
both downstream tasks.

#### 2. Multi-aligned Output Wake Word Model

The second architecture is the multi-aligned output WW model.
In this architecture, instead of just detecting whether a WW exists,
it detects three things:

1. Start of the WW
2. End of the WW
3. Main detector of the WW (centrally aligned)

{% include figure.liquid
    path="/assets/img/summaries/ww-multi-aligned.webp"
    width="600px"
    class="z-depth-1"
%}

Note that the combination of these three detection outputs
means that it is no longer necessary to output the regression
of where the start and end of the wake word is detected.

This architecture was found to perform the best in WW detection.

#### 3. Pseudo-Ground Truth Labels for Training and Evaluation

Due to the difficulty of annotating WW endpoint labels, the authors used the
then state-of-the-art acoustic model + Hidden Markov Model keyword spotter
(AM+HMM KWS) to label the data as pseudo-ground truth labels.

### Most Glaring Deficiency

The labels used are not ground truth labels but rather pseudo-ground truth
labels, which may affect the reliability of the results obtained.

### Conclusions for Future Work

Instead of just detecting for a particular feature in inputs,
we could decompose it into detecting different parts of the feature.
In the case of sequential data, we could break it up into start,
middle, and end like for the multi-aligned output model in this paper.
This also has the advantage of lower latency for detecting the start
of the feature, which could help to improve the user experience for
latency-sensitive applications.
