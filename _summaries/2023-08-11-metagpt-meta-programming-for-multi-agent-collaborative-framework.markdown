---
layout: summary
title: "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"
giscus_comments: true
bib_id: 2308.00352v3
---

### Three Important Things

#### 1. MetaGPT
MetaGPT is a multi-agent system, where different agents have their own
roles, standard operating procedures (SOP), and goals.

{% include figure.liquid 
    path="/assets/img/summaries/metagpt-overview.webp"
    width="600px"
    class="z-depth-1"
%}

In the paper, the goal of the overall system is to perform software development,
with roles split between: Product Manager, Architect, Project Manager, Engineer, and QA Engineer.

The different roles are responsible for different parts of the system: given a single
one-line prompt (i.e `Make the 2048 sliding tile number puzzle game`),
the product manager will collect requirements and define the scope of the task, 
the architect will decide on the framework, overall architecture, and files of the project,
the engineer will implement the code, and the QA engineer will run and debug the code.

#### 2. Standardized Output Scheme
The authors enforce that the output schema for each action by the agents is standardized.
This helps to create a uniform interface for communication, restricts possible behaviors of the
LLM outputs, and aligns with real-world quality standards of software engineering development.

#### 3. Diverse Roles Help
Ablation studies showed that removing roles resulted in worse performance
across various game development tasks.

{% include figure.liquid 
    path="/assets/img/summaries/metagpt-ablation.webp"
    width="600px"
    class="z-depth-1"
%}

This shows that having agents specialized in different tasks to accomplish a complex task
can result in improvements.

### Most Glaring Deficiency
This still incurs high setup cost, where one must come up with a suitable role
division when trying to create a multi-agent setting with distinct roles to
tackle a problem. As such, it requires a lot of human input and priors, and
there is no guarantee that our standard approach of structuring role divisions
is actually the most effective one.

Could one also come up with a more general framework where even the required roles
can be automatically learned, based on past failure points? For instance,
a MetaGPT software development team could realize a need to be able to talk to a customer
to improve the usability of a product (perhaps due to poor UI/UX scores in an
alternative evaluation framework), and a customer role could be automatically
added into the team.

As an aside, I found the paper unnecessarily verbose and repetitive.

### Conclusions for Future Work
Multi-agent teams with specialized roles can be used to further break down complex tasks,
in a similar spirit as [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903).