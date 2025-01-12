---
layout: summary
title: "Large Language Models for Software Engineering: Survey and Open Problems"
giscus_comments: true
bib_id: 2310.03533v4
---

### Three Important Things

As a survey paper, it covered a lot of topics on how LLMs are being applied in
software engineering that I didn't think was worthwhile to repeat for the sake
of it. Instead, I'll highlight the 3 most important sentiments/takeaways I
gained.

#### 1. The Future of Coding

The paper mentioned some pretty surprising statistics:

> In 2021, OpenAI introduced CodeX, a fined-tuned descendant of GPT-3. CodeX is
> used by GitHub’s Copilot, which provides users of Visual Studio Code, Visual
> Studio, Neovim, and JetBrains with code completion. The new version of Copilot,
> GitHub Copilot X2 , is based on GPT-4. In February 2023, GitHub reported that,
> on average, 46% of the developers’ code was written by Copilot [25]. For Java
> only, that number is 62%. Thomas Dohmke, CEO of GitHub, said Copilot will write
> 80% of code “sooner than later” in June 2023 [26]

I knew that coding assistant usage is getting popular, but had no idea it has
already taken off to this point that in the near future most code will be
AI-generated.

This was scary in some ways: the flywheel effect of code language models
eventually being trained on its own outputs is much closer than I anticipated,
and begins to erode the status of code as an art form to a commodity.

There was a great article from James Somers from the New Yorker that
reflects my sentiment really well:
[A Coder Considers the Waning Days of the Craft](https://www.newyorker.com/magazine/2023/11/20/a-coder-considers-the-waning-days-of-the-craft).
A quote from the article that I found particularly quaint:

> Perhaps what pushed Lee Sedol to retire from the game of Go was the sense that the game had been forever cheapened. When I got into programming, it was because computers felt like a form of magic. The machine gave you powers but required you to study its arcane secrets—to learn a spell language. This took a particular cast of mind. I felt selected. I devoted myself to tedium, to careful thinking, and to the accumulation of obscure knowledge. Then, one day, it became possible to achieve many of the same ends without the thinking and without the knowledge. Looked at in a certain light, this can make quite a lot of one’s working life seem like a waste of time.

#### 3. Paradigm Shift: Probabilistic Programming

Using off-the-shelf foundation models to perform many software engineering tasks
in general has a pretty poor accuracy rate, and most of the techniques
considered are ways to improve this to a level that is more usable.

However, the point is that fundamentally using LLMs to solve these problems will
always be a probabilistic process, and therefore the focus is on adopting a
framework that can allow for these failures and yet still be useful most of the
time.

#### 3. Rethinking Hallucinations

LLM hallucinations are traditionally cast in a negative light
as a failure mode that needs to be "fixed" with better techniques
and data.

However, this paper provides an alternative view: LLMs by their
nature are autoregressive machines that samples statistically probable
completions, and the fact that hallucinations happen is indicative
of what it deems a 'plausible future' that we simply disagree with.

This is down to two reasons:

1. Our intention was misunderstood by the LLM, but if a
   LLM misunderstood the context around our code, couldn't this
   misunderstanding also be possible from humans? Maybe our code was
   poorly written and had misleading variable names?
2. The hallucinated functions/variables may be things that the LLM needs, but
   could not find and hence it simply came up with them. Perhaps this is indicative
   that the code is fundamentally missing some useful constructs that should be
   added?

Based on this new perspective, we can view hallucinations as instead a
constructive guiding step towards writing better code, such as to perform
refactoring or creating new APIs that are needed to solve the task at hand.

### Most Glaring Deficiency

Can't really criticize a survey paper much, I think they summarized the field
well.

### Conclusions for Future Work

A lot of exciting work is being done to apply LLMs to software engineering, ranging from code generation to code repair to debugging. There has been
promising early signs of success, and the way we write software in the future
could change dramatically.
