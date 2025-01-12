---
layout: summary
title: "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"
giscus_comments: true
bib_id: stochastic-parrots
---

### Three Important Things

This is mostly a qualitative paper that details the environmental and social
harms of training and deployment of LLMs.

#### 1. Environmental Costs

There is a large carbon footprint for training LLMs -
for instance, training a large Transformer model emits 284t of CO2, compared
to the average per-capita emission of 5t for humans.
The rate of growth of LLMs will only make this progressively worse.

As such, it is important to pay attention to green AI (i.e computationally
efficient hardware and algorithms, promoting efficiency as an evaluation
metric), and to internalize the environmental costs of training LLMs.

In particular, the people in countries most affected by the environmental
impacts of climate change are often those who stand to gain the least from LLMs
due to being economically and socially marginalized.

#### 2. Unfathomable Training Data

The rate of growth of datasets used for training LLMs has also resulted in a
documentation debt, where existing corpuses are undocumented and too large to
now be documented.

Documentation is important because the choice of datasets used for training
popular models currently reflect the views of a vocal minority on the Internet.
For instance, GPT-2 included all outbound links from Reddit as its training corpus,
under the assumption that these are likely links that contain information which
are interesting to humans. However, Reddit users are predominantly men
between 18 and 29 from Western countries, which skews the representations
that the language model learns.

Another problem that LLMs face is with distribution shift - while underlying
social views may be changing, LLMs may continue to perpetuate biases and
stereotypes from the training set it had access to when it was trained.

Finally, while it is important to ensure that language mdoels do not output toxic
content, doing so could counterintuitively further marginalize disadvantaged communities.
Developers of previous language models have made good-faith attempts to
do this using methods including removing potentially toxic content from
their training datasets. A popular way of doing this is using a
filter list, i.e
[List of Dirty Naughty Obscene and Otherwise Bad Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en),
and to remove any content that contains words in the list. However, this was
found to marginalize LGBTQ communities where words like _twink_ appeared
frequently in their discourse, but are often included in such lists.

#### 3. Risks and Harms

The paper lists many of the risks and harms that can result from the deployment of LLMs.

On a psychological front, there is a risk that people ascribe intention behind
the stochastic output of LLMs, believing that they actually understand what they
output and are trying to convey meaning. This could be due to our implicit
programming as we would always believe that the output from other humans carry
some form of intent, but it is not something we should carry over to LLM output.
However, the increasingly blurry distinction between human and LLM-produced outputs
makes this especially challenging and risky.

On a fairness and inclusion front, LLMs will largely absorb the "hegemonic worldview"
from their training corpus, and learn biases that are present, especially from
groups who are already in privileged positions in society.

Furthermore, in spite of attempts at making LLMs more fair and equitable, such
attempts will never be sufficiently comprehensive and complete. Prevailing
views of diversity and inclusion largely draws from legally protected attributes
in the US such as race and gender, but miss out on many other marginalized
communities around the world that don't fit into this mould.

On a reliability front, there are also concerns of LLMs producing wrong output
that lead to real-world consequences, in particular for disadvantage communities.
For instance, they cited the example of how
[a Palestinian man was wrongly arrested](https://www.theguardian.com/technology/2017/oct/24/facebook-palestine-israel-translates-good-morning-attack-them-arrest)
by the Israeli authorities when a post saying "good morning" in a dialect of
Arabic resulted in Facebook translating it into "hurt them" in English, which
prompted the government to take action. A lack of representative training data for
these languages with fewer number of speakers is likely to be a major reason for
the mistranslation.

### Most Glaring Deficiency

I found the paper overly qualitative with a lack of clear evidence or results to
validate their points, and their arguments sometimes appeals to the pathos too
much, which would be fine in normal writing but is probably insufficient for an
academic paper.

### Conclusions for Future Work

While there is not much in the way of technical contribution, this is still
an influential paper in its own right, as it kicked off a lot of discussion
regarding the ethics and morality of LLMs back when the paper was published in
2021 before usage of LLMs truly took off.

Most of what is discussed is common knowledge now, which is perhaps a testament
to the success of the efforts of the authors and all others who work to ensure
that LLMs are put to use for good.

To end off, quoting Birhane and Prabhu, based off a quote from Ruha
Benjamin:

> "Feeding AI systems on the world’s beauty, ugliness, and cruelty, but
> expecting it to reflect only the beauty is a fantasy."
