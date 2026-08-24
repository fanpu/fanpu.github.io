---
name: exercise-post-style
description: Writing style for Fan Pu's exercise-series blog posts (the "Exercises for Understanding ..." series in _posts/ with {% include theorem.md type="exercise" %} blocks and {% answer %} bodies). Load BEFORE drafting or editing any prose in these posts, including exercise statements, answer bodies, and section intros. Triggers on any request to write, draft, expand, revise, or add exercises/answers to a post in this series.
---

# Exercise-series prose style

Applies to the exercise series only (posts built from `theorem.md` exercise
includes + `{% answer %}` bodies). Not to other posts, not to code, not to chat.

Everything below is measured from the 50 published posts in `_posts/`. Posts
marked `published: false` are not evidence, whatever their content.

Read the positive half first. The prohibitions at the end are a checklist for
revision, not a drafting strategy: prose that merely avoids all of them lands on
generic technical writing, which is the failure mode this file exists to fix.

## Registers

Three registers, mapped to the parts of an exercise post. They are different,
and averaging them produces nobody.

**Answer bodies.** Anchors: `2022-12-28-central-limit-theorem-and-berry-esseen`,
`2025-01-12-bounding-markov-chain-mixing-times-by-spectral-gap`,
`2023-07-14-high-dimensional-analysis-of-m-estimators`. Plural "we", not "I".
Lead-ins that name the quantity. Difficulty calibrated aloud. Plain stops. No
jokes inside a derivation.

**Post and section intros.** Anchors: `2023-01-02-latex-tips`,
`2020-07-30-breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-1`, and Fan
Pu's own intro to the exercise post itself:

> I frequently get questions from friends and internet strangers on good ways to
> get started with understanding how LLMs work or are trained.

First person singular, a hook or a question, warm second person, an admitted
motivation. "Let's" and an exclamation mark belong here.

**Asides and opinions.** Anchors: `2023-06-09-advanced-operating-systems-course-review`,
`2022-08-07-impagliazzos-five-worlds`, `2023-06-16-cmu-steam-tunnels`. "I" plus
lived specifics, hedged, dry jokes allowed.

Warmth lives in the intros and section openers. An answer body stays in the
middle register.

## Core markers, all registers

**"However," is the pivot word.** 67 uses in the technical posts and 21 in the
essays, roughly 4x the next connective and 21x "But". It is nearly always the
hinge of a motivation paragraph: here is the standard thing, however here is the
problem.

**Paragraph shape.** Several long additive sentences, then one short declarative
that closes the thought. This holds in the math posts and the essays alike.

**Keep the intensifiers.** "very" 36, "really" 21, "quite" 7, "pretty" 5. The
copyedited-flat sentence is an LLM default, not his.

**Evaluative adjectives are plain.** "interesting" 29, "nice" 16, "wonderful" 8,
"cool" 8, "delightful" 6, "beautiful" 4, "elegant" 0.

**Parentheticals do real work**: gloss a term, flag scope, slip in a reaction.
"(this is even harder than NP-hard)", "(of Hamming code fame)", "126 layers (!!)".

**Loose is authentic.** The corpus contains doubled words ("that that",
"possible possible", "how how"), agreement slips, 50-word run-ons, and mixed
British/American spelling inside one post. Do not polish toward balance. A
balanced tricolon or a colon-plus-three-clause summary is a tell.

## Answer bodies

**Lead-ins name the quantity, as a purpose clause.** This is the most consistent
structural habit in the math posts:

> To upper bound $$\| \Delta_{\mocal} \|$$, we have
> To relate the L2 distance to L1 distance, we can apply the above inequality to get
> Our goal is to show that $$Z_n$$ converges in distribution to $$Z \sim N(0,1)$$.
> We lower bound the first term as $$\langle \nabla \lcal (\ts), \Delta \rangle \geq - \frac{\lambda_n}{2} \rcal(\Delta)$$:

There are no "Step 1 / Step 2" markers anywhere in the corpus.

**Calibrate difficulty aloud.** "This is not too hard to see by induction", "The
base case is trivial", "for brevity".

**State scope limits plainly** rather than hiding them: "We will not discuss
coupling in this post, but will instead develop how spectral gaps can be used."

**"i.e" with no trailing period** in technical prose (55 of 57 corpus uses). In
narrative prose he writes "For instance," (33) instead.

**"Note that"** flags a condition that matters, followed by the reason and often
a parenthetical counterexample.

**"This ..." as anaphoric opener** naming the previous sentence's consequence:
"This is because", "This means that", "This shows that".

**Admitted confusion is allowed, in first person**: "but I did not manage to
figure out how they managed to produce a $$\lambda_n$$ term".

## Math and numbers

**Displays are clauses, not exhibits.** About 45% are followed by a lowercase
word continuing the sentence: "where the last step follows from the fact that
$$x$$ is a probability distribution", "which is a cone.", "where $$A$$ is the
adjacency matrix of the graph."

**Step justification goes inside the align block**, as a right-aligned `\text{}`
or a `\tag{}`, not in a prose paragraph:

    & = \frac{d}{dt} \mathbb{E} \left[ X^{k-1} e^{tX} \right] & \text{(by IH)}
    & \text{(Cauchy-Schwarz using dual norms)}
    \tag{by Cauchy-Schwarz}

**After a number, interpretation is 0 to 1 sentences**, usually a trailing clause
of the same sentence rather than a new one:

> a 402B model with 16.55T tokens is optimal, which led to their 405B model

**A surprising number gets a parenthetical**, not a sentence of commentary:
"Llama 3 405B: 126 layers (!!)".

**Never bold a result mid-prose.** Zero instances in 3,195 lines of math posts.
Bold is for list labels only (`**(G1)**`, `**(a)**`). Italics mark a term of art
at first definition, once.

## How to end

Across ~70 terminal positions (section end, post-math, post-list, post-figure)
in the technical posts: **zero portable aphorisms.** The distribution is roughly
26% bare stop, 31% straight on to the next concrete fact, 20% forward roadmap,
16% local summary, 7% editorial punchline.

Bare stops: "as desired.", "This concludes the proof.", "hence we have equality."
Forward: "Our goal now is to formulate a robust version of this corollary."

When he does summarize it is **one sentence, never two**, and it stays inside the
object under discussion:

> This shows that if your spectral gap is bounded by a constant, your mixing time is in $$O(\log (n))$$.
> This short analysis tells us that if our graph looks like a line graph then we should expect poor mixing times; whereas if it looks more like a complete graph then we can expect the opposite.

Summarize the object, never the structure of your discussion of it, and never
past one sentence. Ending forward or stopping dead beats summarizing.

## Prohibitions

**No em-dashes.** One `&mdash;` in 50 posts, and no `—` at all. Use a comma, a
colon, a period, or parentheses.

> ✗ `201 GB of activations on an 80 GB card — 2.5 cards' worth, per card.`
> ✓ `That is 201 GB of activations on an 80 GB card, against 2.19 GB of model state.`

**No principle-minting.** Do not name an abstraction that has no name, do not
restate a list one altitude up, no portable maxims, no "Three things to take
away. First... Second...". Corpus counts across 50 posts: "That said" 0,
"Additionally" 0, "Crucially" 0, "Fundamentally" 0, "At its core" 0, "worth
noting" 0, "It is important to note" 0, "In essence" 0, "Simply put" 0, "The key
insight" 0, "In conclusion" 0, "arguably" 0, "delve" 0, "pivotal" 0, "testament"
0, "underscore" 0, sentence-initial "Of course" 0. "Moreover", "Notably",
"Importantly" and "To summarize" appear once each.

> ✗ `The ordering follows the cost of evidence: 1 to 3 come from data already on hand, 4 and 5 need one profiler run, and the rest need cluster telemetry.`
> ✓ `Hypotheses 1 to 3 are answered by data already on hand, and 4 and 5 need a profiler run.`

**Lead-ins must name a destination, not an activity.** Announcing the step is
fine and characteristic, as long as the sentence carries a fact. If deleting it
loses nothing, it was scaffolding.

> ✓ `To bound the logits term, we have` / `Our goal is to show the gather cannot be hidden.`
> ✗ `State the formula first:` / `Now count.` / `Derive it from the pass structure.` / `First, frame the budget.` / `Putting it together`

**No bold-lead paragraph headers.** Don't open paragraphs with `**The budget.**`
or `**ZeRO-3 gather buffers.**`, and don't use an italic lead-in for the same
job. If a chunk needs a label it needs a heading.

**Don't dramatize numbers.** Report the quantity and its consequence. No "a
rounding error", "the entire reason A2 exists", "1.6× your entire checkpoint
budget" as a flourish.

> ✗ `the thing that forced you onto 512 GPUs is now a rounding error`
> ✓ `Model state is 9% of the footprint here, so the remaining decisions are all about activations.`

**Second person is warm, never imperious.** "you shouldn't feel bad for making
them" is his voice; "you must", "worth knowing about", "here's the thing" are
not. Corpus-wide there are 57 "Let's", 232 exclamation marks, and "I" (538) is as
frequent as "we" (570), so the register is not the problem. The scolding is.

## The symptom underneath

Stage directions, bold-lead labels, enumerated wind-ups and minted principles are
one habit: stepping outside the material to comment on it, whether by narrating
the shape of the answer or by extracting a portable lesson from it. It reads as
performing rigor rather than being rigorous, and it is the strongest single tell
of LLM-written prose. The reader can see that a paragraph is a new topic, that a
formula came before its numbers, and that a list has five items.

The test: delete the sentence or label. If nothing is lost, it was scaffolding.

## Default shape of an answer body

Name what is being computed, show the arithmetic, give the number, and fold the
implication into that same sentence where it fits. If it does not fit, one
sentence. Then stop.

## What to keep

The technical content should not be softened: itemized tables with
`{: .table .table-bordered .table-sm }`, explicit arithmetic in `$$...$$`,
per-part `**(a)**` `**(b)**` labels, stated assumptions. Precision and
completeness are the point. Only the connective prose changes.

## Rewrite bank

Sentences Fan Pu has flagged, verbatim. These are ground truth about his ear:
when he flags another, append it here with the rule it broke.

| Flagged | Rule |
|---|---|
| `**(b)** State the formula first:` | activity, not destination |
| `The ordering follows the cost of evidence: 1 to 3 come from data already on hand, 4 and 5 need one profiler run, and the rest need cluster telemetry.` | principle-minting |
| The 13-row per-layer tensor census in A2(c) | over-elaboration; he asked for it cut |
| `**The budget.**`, `*The recompute working set.*` | bold/italic lead-in labels |
| `Three things to take away. First... Second... Third...` | enumerated wind-up |
