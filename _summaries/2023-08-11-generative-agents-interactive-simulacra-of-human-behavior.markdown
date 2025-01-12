---
layout: summary
title: "Generative Agents: Interactive Simulacra of Human Behavior"
giscus_comments: true
bib_id: 2304.03442v2
---

### ~~Three~~ Four Important Things

#### 1. Smallville
The paper aims to investigate whether LLM-powered (specifically ChatGPT)
generative agents can behave in a believable human-like behavior in a sandboxed
game that they call Smallville.  In this simulation, 25 agents are each given
distinct personalities and backgrounds.  Two of them are given special
directives, which are initially private only to themselves:

1. Isabella Rodriguez is tasked with organizing a Valentine's Day party
2. Sam Moore is tasked with running for mayor

It was observed that the social interaction between the agents resulted in
multiple other agents also learning about the Valentine's Day party (some of
whom eventually attended the party), as well as discussions about their feelings
regarding Sam running for mayor. This provides evidence for the emergence of
social behavior.

An example of instructions given to a particular agent is shown below.

```
John Lin is a pharmacy shopkeeper at the Willow
Market and Pharmacy who loves to help people. He
is always looking for ways to make the process
of getting medication easier for his customers;
John Lin is living with his wife, Mei Lin, who
is a college professor, and son, Eddy Lin, who is
a student studying music theory; John Lin loves
his family very much; John Lin has known the old
couple next-door, Sam Moore and Jennifer Moore,
for a few years; John Lin thinks Sam Moore is a
kind and nice man; John Lin knows his neighbor,
Yuriko Yamamoto, well; John Lin knows of his
neighbors, Tamara Taylor and Carmen Ortiz, but
has not met them before; John Lin and Tom Moreno
are colleagues at The Willows Market and Pharmacy;
John Lin and Tom Moreno are friends and like to
discuss local politics together; John Lin knows
the Moreno family somewhat well — the husband Tom
Moreno and the wife Jane Moreno.
```

#### 2. Memory and Retrieval, 
Memory and retrieval is important to the believability of an agent's interaction, as they
should be able to remember interactions and information from the past. All events pertaining
to an agent is stored to the memory stream of the agent.  However, due to the
limited context window of ChatGPT, and the distracting nature of having too many irrelevant
events, it is infeasible to simply input the entire memory stream as part of the prompt
when determining how the agent should respond.

{% include figure.liquid 
    path="/assets/img/summaries/simulacra-memory-stream.webp"
    width="600px"
    class="z-depth-1"
%}

Instead, the authors use the following approach to retrieve salient memories, summarized in the figure below:

{% include figure.liquid 
    path="/assets/img/summaries/simulacra-retrieval.webp"
    width="600px"
    class="z-depth-1"
%}

1. Each event in the memory stream is assigned a recency score, where more recent events have a higher score (more relevant)
2. A LLM query is performed to determine the importance of an event:
3. The relevance of the event to the current query is computed by the cosine similarity between the embedding of the query, and the embedding of the event

```
On the scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the
following piece of memory.
Memory: buying groceries at The Willows Market
and Pharmacy
Rating: <fill in>
```

The score is then given by the following:

$$\text{score} =
\alpha_{\text{recency}} \cdot \text{recency} + 
\alpha_{\text{importance}} \cdot \text{importance} + 
\alpha_{\text{relevance}} \cdot \text{relevance},
$$

where in the paper all scaling parameters $$\alpha$$ are set to 1.

#### 3. Reflection
For the agents to exhibit higher-level cognitive abilities and to synthesize their experiences,
the authors program agents to go into a reflection state once the importance
scores of the latest events cross a certain threshold (150 in the paper). 

The reflection state results in a LLM call like the following:

```
Statements about Klaus Mueller
1. Klaus Mueller is writing a research paper
2. Klaus Mueller enjoys reading a book
on gentrification
3. Klaus Mueller is conversing with Ayesha Khan
about exercising [...]
What 5 high-level insights can you infer from
the above statements? (example format: insight
(because of 1, 5, 3))
```

The new insight is then added to the memory stream of the agent.

#### 4. Planning and Reacting
The authors also add a planning and reacting component to the agents.  The first
is necessary so that agents don't repeat their actions over the course of a day
(i.e have lunch 3 times a day), and the second is necessary to exhibit
believable social behavior.

Planning can be done by a LLM query, like as follows:

```
Name: Eddy Lin (age: 19)
Innate traits: friendly, outgoing, hospitable
Eddy Lin is a student at Oak Hill College studying
music theory and composition. He loves to explore
different musical styles and is always looking for
ways to expand his knowledge. Eddy Lin is working
on a composition project for his college class. He
is taking classes to learn more about music theory.
Eddy Lin is excited about the new composition he
is working on but he wants to dedicate more hours
in the day to work on it in the coming days
On Tuesday February 12, Eddy 1) woke up and
completed the morning routine at 7:00 am, [. . . ]
6) got ready to sleep around 10 pm.
Today is Wednesday February 13. Here is Eddy’s
plan today in broad strokes: 1)
```

The level of granularity of planning can be broken down by iteratively
performing more queries.

Finally, reacting can also be done by a LLM call, with an example prompt below:

```
[Agent’s Summary Description]
It is February 13, 2023, 4:56 pm.
John Lin’s status: John is back home early from
work.
Observation: John saw Eddy taking a short walk
around his workplace.
Summary of relevant context from John’s memory:
Eddy Lin is John’s Lin’s son. Eddy Lin has been
working on a music composition for his class. Eddy
Lin likes to walk around the garden when he is
thinking about or listening to music.
Should John react to the observation, and if so,
what would be an appropriate reaction?
```

### Most Glaring Deficiency
It would be interesting to study the behavior of agents when
they have conflicting goals, as ensuring that pathological behavior
does not result is a key goal of [cooperative artificial intelligence](http://www.cs.cmu.edu/~conitzer/FOCALAAAI23.pdf).
The sycophantic nature of language models has been well-documented, and therefore 
understanding how conflict resolution would be undertaken will be illuminating.

### Conclusions for Future Work
This paper demonstrates the presence of emergent social behavior of generative
agents based on LLMs, and is a convincing demonstration that this could be a
viable technology for applications such as games and tools for interpersonal
communication.

It also shows how usage of the reasoning capabilities of LLMs can be used to 
solve some problems that we currently don't have good algorithmic approaches
for, such as determining the importance of events during memory retrieval.