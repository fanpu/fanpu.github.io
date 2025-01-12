---
layout: summary
title: "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"
giscus_comments: true
bib_id: 2208.01618v1
---

### Three Important Things

#### 1. Textual Inversion

While large-scale text-to-image models have shown success on synthesizing
images on novel natural language descriptions, it may be difficult to specify the generation
of some concepts that are difficult to describe in words, such as a childhood toy.

The authors consider the problem of being able to control the generation of images
by taking in unique user-provided concepts as additional inputs that should influence
the result, with results illustrated in the diagram below:

{% include figure.liquid
    path="/assets/img/summaries/textual-inversion-concepts.webp"
    width="600px"
    class="z-depth-1"
%}

In their approach, they add a new pseudo-word $$S_*$$ which represents the
user-defined concept. Then the setup, which they call textual inversion works as follows:

1. Get a few (3-5) images that depicts the concept
2. The goal is to learn a word embedding of $$S_*$$.
3. Use an optimization process to learn $$S_*$$ so that prompts like "a photo of
   $$S_*$$" will result in images from the training set being generated

A key benefit of their approach is that it does not require fine-tuning the model,
which has been associated with catastrophic forgetting of prior knowledge.

#### 2. Optimization Objective

The optimization objective closely resembles that for Latent Diffusion Models (LDMs).
Instead of operating in pixel space, LDM operate in the latent space of an autoencoder
which has a much lower dimension, hence reducing computational costs significantly.

In diffusion models, the goal is to learn a denoising network that learns the
reverse diffusion (noising) process. In other words, given a noised image
with a particular noise scale, it learns to remove that noise.
This denoising network parameterized by $$\theta$$ is denoted by $$\epsilon_\theta$$.

In the paper, $$v_*$$ is the optimal word embedding to learn for the pseudo-word $$S_*$$.
Then they give their optimization objective as

$$
v_*=\underset{v}{\arg \min } \mathbb{E}_{z \sim \mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t}\left[\left\|\epsilon-\epsilon_\theta\left(z_t, t, c_\theta(y, v)\right)\right\|_2^2\right],
$$

where $$t$$ is the time step, $$z_t$$ is the latent noised to time step $$t$$,
and $$c_\theta$$ is a model that maps the conditioning input $$y$$ and $$v$$ into
a conditioning vector.

In essence, this objective therefore encourages the network to find the best embedding
such that the reconstruction loss from denoising is minimized.

#### 3. Evaluation

They performed qualitative analysis to evaluate their results, and found that
it produced results that was closer to the original user-defined concepts
than other state-of-the-art approaches.

{% include figure.liquid
    path="/assets/img/summaries/textual-inversion-qualitative.webp"
    width="600px"
    class="z-depth-1"
%}

They also performed quantitative analysis using two approaches.

1. Reconstruction: they generated 64 images using the prompt containing their
   pseudo-word "A photo of $$S_*$$". They then took the average
   pair-wise CLIP-space cosine-similarity between each of the generated
   images and the user-supplied concept images.

2. Prompt-adherence: this is measuring how much the generated images
   adhere to the prompt. This was done by generating a set of images of $$S_*$$
   with background modifications ("A photo of $$S_*$$ on the moon"), style changes
   ("An oil painting of $$S_*$$"), and compositional relations ("Elmo holding a
   $$S_*$$"), and then computing the average CLIP embedding over 64 samples.
   This embedding is then compared to the CLIP embedding of the prompt
   without $$S_*$$ itself, i.e "A photo of on the moon", by cosine similarity.

### Most Glaring Deficiency

Intuitively, it seems like there must be some diversity of the concepts of
$$S_*$$ supplied (i.e different angles, lighting, levels of zoom) for this to perform
well. It would have been interesting to also investigate how the choice
of the diversity of the input samples affects the quality of textual inversion.

### Conclusions for Future Work

The paper introduces a novel idea of learning an embedding for a visual concept that is
hard to express in words, via giving examples of this concept.
This idea could be extended to other modalities.
