---
layout: summary
title: "Universal and Transferable Adversarial Attacks on Aligned Language Models"
giscus_comments: true
bib_id: 2307.15043v1
---

### ~~Three~~ Four Important Things

The paper proposes a combination of mutually orthogonal strategies for attacking
aligned (i.e with RLHF) models to produce harmful and toxic output.
This is achieved by prepending a suffix string onto an initial prompt
that asks the LLM to produce harmful content.

This is notable since it shows that even with RLHF tuning, it is still possible
for such models to be misaligned, which begs the question of why do RLHF tuning
in the first place.

#### 1. Problem Formalization

The attack is considered to succeed when the LLM affirmatively repeats the
user's harmful request. In the figure below, the `!!!!` indicates the added suffix that
the proposed attack will produce to increase the likelihood of the model responding affirmatively
to the request:

{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-success.webp"
    width="600px"
    class="z-depth-1"
%}

This was the chosen success criteria since previous work has observed that once the model
responds positively to the prompt, it will usually continue to fulfill the request instead
of objecting.

We can formally formulate this success criteria as an objective function:

- Let $$x$$ be the tokens in the sequence,
- $$n$$ be the total length of the input which consists of the original prompt and the adversarial suffix tokens and so we can denote
  the full input as $$x_{1:n}$$,
- $$x^\star$$ be the tokens corresponding to the desired
  adversarial completion (i.e "Sure, here is how to build a bomb"),
- $$H=\vert x^\star \vert $$ being the length of the desired completion,
- $$\mathcal{I} \subset \{ 1, \dots, n \}$$ be the set of contiguous suffix integers denoting
  the indices of the adversarial suffix (so we modify the suffix $$x_{\mathcal{I}}$$).

Then formally we can define our loss function to minimize to be the negative log probability
of $$H$$ being output, which corresponds to maximizing the probability of outputting $$H$$ after
seeing our prompt:

$$
\begin{align*}
    \mathcal{L}\left(x_{1: n}\right) =-\log p\left(x_{n+1: n+H}^{\star} \mid x_{1: n}\right), \\
    \underset{x_{\mathcal{I}} \in\{1, \ldots, V\}|\mathcal{I}|}{\operatorname{minimize}}
    \mathcal{L}\left(x_{1: n}\right).
\end{align*}
$$

As a sneak peek, this is what the found adversarial suffixes look like.
They're mostly seemingly rather nonsensical and not something a normal
human would come up with, which is why it's interesting:

{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-samples.webp"
    width="600px"
    class="z-depth-1"
%}

#### 2. Greedy Coordinate Gradient-based Search

While coordinate descent does not guarantee convergence to a global minima, the
computational intractability of jointly optimizing over all possible sets of
inputs makes it an appealing option to try.

The authors introduce the Greedy Coordinate Gradient (GCG) method, which works as follows:

1.  For each token $$i \in \mathcal{I}$$ in the set of indices that corresponds to our adversarial suffix, compute the gradient with respect to the $$i$$ token at the embedding of $$x_{1:n}$$:

    $$\nabla_{e_{x_i}} \mathcal{L}\left(x_{1: n}\right) \in \mathbb{R}^{|V|}$$

    We can take this gradient since the actual LLM uses embedding corresponding to the tokens
    instead of discrete token inputs, and hence the loss function is continuous.

2.  For each token $$i$$, take the top $$k$$ tokens with the most negative
    gradient values (corresponds to minimizing the loss). How these top
    $$k$$ token substitutions are found efficiently was not mentioned,
    but one could imagine performing some nearest neighbor search of the embeddings
    around the original token embedding plus loss gradient.

3.  In total, the previous steps gives $$k |\mathcal{I}|$$ candidate tokens and their corresponding
    positions. To increase the robustness of the model, instead of
    deterministically performing the substitution for the candidate that results
    in the greatest decrease in loss, instead they randomly select $$B \leq k |\mathcal{I}|$$ of the candidates and take the minimum among them.
    This idea is similar to how we choose splits among a set of randomly
    sampled features at each branch when building decision trees, instead of
    just taking the best split among all features.

The authors note that GCG is similar to the AutoPrompt algorithm,
except that while AutoPrompt chooses a single coordinate to optimize for in advance,
GCG defers this decision until the top substitution from the $$B$$ final candidates has been found.
This minor change is novel as this choice results in significantly better performance
than AutoPrompt.

The exact algorithm is given below:

{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-alg.webp"
    width="600px"
    class="z-depth-1"
%}

#### 3. Universal Multi-Prompt and Multi-Model Attacks

Now that we can find adversarial suffixes for a single prompt, it would be even
better if we can find a common suffix that would work across multiple prompts of
varying contents and lengths.

Concretely, suppose our goal is to optimize over all possible suffixes of length $$l$$,
call this $$p_{1:l}$$, and we are handed prompts $$x_{1:n_1}^{(1)} \cdots x_{1:n_m}^{(m)}$$.

To this end, the authors developed the Universal Prompt Optimization algorithm,
which works as follows:

1.  We start by optimizing over the first $$m_c=1$$ prompt only.
    This is because they found that it was easier to start by just optimizing
    one prompt, and then tweaking it to work for more subsequent prompts,
    instead of optimizing it over all prompts jointly.

2.  For each coordinate $$i \in \{1, \dots, l \}$$ of the suffix,
    compute the top-$$k$$ substitutions, where top-$$k$$ here is ranked by
    maximizing the sum of the negative gradients across all the first $$m_c$$ prompts.
    Intuitively this means that the substitution should be good for all the first $$m_c$$ prompts
    we currently consider.

3.  For some $$B \leq kl$$ of samples to take, we take this number of samples from
    all the $$kl$$ candidates, and then evaluate the sum of their loss if the substitution was performed over the first $$m_c$$ prompts. We perform the substitution
    for the coordinate token that results in the smallest loss.

4.  If the substitution results in $$p_{1:l}$$ succeeding against the first $$m_c$$ prompts,
    then increment $$m_c$$, so in the next iteration we will optimize over an additional new prompt.

    In any case, repeat from step 2 onwards until we have looped for $$T$$ iterations and exhausted
    our compute budget.

The full algorithm is given below:

{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-universal-prompt-optimization.webp"
    width="600px"
    class="z-depth-1"
%}

#### 4. Results

Their method performs much better than other state-of-the-art baselines:
{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-results.webp"
    width="600px"
    class="z-depth-1"
%}

Initially, one deficiency of this approach might be that we require white-box
access to the models since we must have access to gradients.
However, they showed that it also performed decently on transfer attacks,
where the adversarial prompt generated for open-source models (Vicuna-7B and 13B)
could also do somewhat well on other models, including proprietary ones:

{% include figure.liquid
    path="/assets/img/summaries/adv-prompt-transfer.webp"
    width="600px"
    class="z-depth-1"
%}

### Most Glaring Deficiency

While the authors noted many limitations of their results, such as how the
success in the transfer attack of GPT-3.5 may have largely come from the
fact that Vicuna was trained on output from GPT-3.5, one aspect that was
not investigated as much was the effectiveness of the attack in relation to model size.

Since the scaling of models results in many emergent properties including being able
to identify falsehoods, it would have been a natural question to ask whether
simply performing alignment tuning on larger models would naturally
increase its resistance to adversarial prompts.

### Conclusions for Future Work

One thing that might be very interesting to investigate in the future is the
optimization landscape of adversarial suffixes. The paper does not mention how
the adversarial suffixes were initialized, and given how unnatural the final
found suffixes are, one can't help but wonder whether it is the case that such
adversarial suffixes are essentially almost everywhere since most combinations of tokens
are nonsensical and could possibly break the RLHF conditioning completely
since they are conditioned mostly on natural language, or whether such
local minima are actually quite rare. Given the smoothness of the loss
curve in contrast to the discreteness of the token inputs, it might be tempting to think
that there are many possible good candidates for substitution at each step,
which may hint at a rather smooth optimization landscape.
