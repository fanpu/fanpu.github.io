---
layout: summary
title: "InstructPix2Pix: Learning to Follow Image Editing Instructions"
giscus_comments: true
bib_id: 2211.09800v2
---

### Three Important Things

#### 1. InstructPix2Pix

In this paper, the authors tackle the problem of instruction-based image
editing. The setting is that given an image and an instruction, output an image
that has the changes from the instruction applied, without altering the
original style and contents of the image that was not asked to be changed.

Some of their results:

{% include figure.liquid
    path="/assets/img/summaries/instructpix2pix-results.webp"
    width="600px"
    class="z-depth-1"
%}

They take a supervised learning approach, and use latent diffusion models (which
operate in the latent space of a pretrained variational encoder instead of image
space for faster training) to generate the edited image. How they obtained the
dataset is explained in the next section.
Their training objective closely resembles that of the latent diffusion model,
where $$\mathcal{E}$$ is the VAE encoder, $$\epsilon_\theta$$ is the network
to train which learns to predict the noise added to the latent $$z_t$$
conditioned on the source image conditioning $$c_I$$ and text edit instruction
$$c_T$$. The training objective is hence

$$
\left.L=\mathbb{E}_{\mathcal{E}(x), \mathcal{E}\left(c_I\right), c_T, \epsilon \sim \mathcal{N}(0,1), t}\left[\| \epsilon-\epsilon_\theta\left(z_t, t, \mathcal{E}\left(c_I\right), c_T\right)\right) \|_2^2\right],
$$

where minimizing it means that $$\epsilon_\theta$$ is doing a better job at
predicting the noise.

They bootstrapped training from a pre-trained Stable Diffusion
checkpoint. In order to incorporate the image conditioning $$c_I$$,
they added an additional input channel to the first convolutional layer
to take in the encoded image conditioning $$\mathcal{E}(c_I)$$.
To take in the text editing instructions, they re-used the text conditioning
mechanism initially meant for captions.

#### 2. Generating a Dataset

Because of the difficulty of obtaining a large-scale dataset that has the
desired images, text editing instructions, and the images after the edit, the
authors had to generate their own dataset.

To generate instructions, they fed GPT-3 a caption of the image, and
asked it to generate a plausible editing instruction. They then
asked it to determine what a reasonable output caption for the image
after the edits might be, which results in an input-output caption pair.

To generate the paired images, they used Stable Diffusion and
asked it to generate images for both the input and output caption.
However, just using vanilla Stable Diffusion could result in the two
images having wildly different characteristics and styles, as
the figure below shows.
Instead, they make use of the Prompt-to-Prompt plugin, which
ensures similarity between the two images that are generated.

{% include figure.liquid
    path="/assets/img/summaries/instructpix2pix-prompt2prompt.webp"
    width="600px"
    class="z-depth-1"
%}

The overall data generation pipeline is thus summarized in the figure below:

{% include figure.liquid
    path="/assets/img/summaries/instructpix2pix-data.webp"
    width="600px"
    class="z-depth-1"
%}

#### 3. Classifier-free Guidance for Two Conditionings

Classifier-free diffusion guidance is a method of guiding
the diffusion process towards generating images with higher
likelihood under some implicit classifier.

In the usual context, we only have one conditional $$c$$.
Then $$e_\theta(z_t, c)$$ denotes the predicted noise
conditioned on $$c$$, and $$e_\theta(z_t, \varnothing)$$
is the predicted noise with no conditioning.
Then classifier-free diffusion guidance is where we modify
our noise prediction to be
$$\tilde{e}_\theta(z_t, c)$$, which is extrapolated
away in the direction towards the conditional
and away from the unconditional:

$$
\tilde{e_\theta}\left(z_t, c\right)=e_\theta\left(z_t, \varnothing\right)+s \cdot\left(e_\theta\left(z_t, c\right)-e_\theta\left(z_t, \varnothing\right)\right)
$$

They extend this idea to the case where we have 2 conditionals, $$c_I$$ and $$c_T$$,
with scaling factors $$s_I$$ and $$s_T$$ influencing how strongly the input image
and text instructions respectively should influence the final image:

$$
\begin{aligned}
\tilde{e_\theta}\left(z_t, c_I, c_T\right)= & e_\theta\left(z_t, \varnothing, \varnothing\right) \\
& +s_I \cdot\left(e_\theta\left(z_t, c_I, \varnothing\right)-e_\theta\left(z_t, \varnothing, \varnothing\right)\right) \\
& +s_T \cdot\left(e_\theta\left(z_t, c_I, c_T\right)-e_\theta\left(z_t, c_I, \varnothing\right)\right)
\end{aligned}
$$

Controlling these two knobs hence allows us to balance how faithful
the output is to the original source image versus the edit instructions,
as the following figure shows:

{% include figure.liquid
    path="/assets/img/summaries/instructpix2pix-classifier-free-guidance.webp"
    width="600px"
    class="z-depth-1"
%}

### Most Glaring Deficiency

The biggest limitation is that the performance of their approach is limited by
the performance of the Prompt-to-Prompt technique, since this is where its
training dataset comes from. Therefore, it is not a technique whose limitations
can be easily addressed on its own. In a sense, the main contribution of the
paper is UX-based by giving a new instruction-based interface for image editing,
with the previous alternative being simply using Prompt-to-Prompt in Stable
Diffusion and coming up with the caption yourself.

### Conclusions for Future Work

For machine learning models to be more widely adopted, usability is key. For
instance, LLM mass adoption never really took off until OpenAI's ChatGPT made it
intuitive and easy for people to use.

This paper leverages existing techniques and provides a new instruction-based
text interface for editing images, by creatively finding ways to condition on
both text and image inputs. Such an approach could be inspiration for adapting
other models to have an interface that is more natural for the user.
