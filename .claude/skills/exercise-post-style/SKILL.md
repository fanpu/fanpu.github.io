---
name: exercise-post-style
description: Writing style for Fan Pu's exercise-series blog posts (the "Exercises for Understanding ..." series in _posts/ with {% include theorem.md type="exercise" %} blocks and {% answer %} bodies). Load BEFORE drafting or editing any prose in these posts — exercise statements, answer bodies, section intros. Triggers on any request to write, draft, expand, revise, or add exercises/answers to a post in this series.
---

# Exercise-series prose style

Applies to the exercise series only (posts built from `theorem.md` exercise
includes + `{% answer %}` bodies). Not to other posts, not to code, not to chat.

## Voice reference

Match `_posts/2025-01-21-gaussian-processes.markdown` and
`_posts/2024-08-07-llama-3.1-technical-report-notes.markdown`. Read a few
paragraphs of one before drafting. Calm, first-person, explanatory. The writing
gets out of the way of the material.

## Hard rules

**No em-dashes.** Zero, not "sparingly". Fanpu's posts contain none. Use a
comma, a colon, a period, or parentheses.

> ✗ `201 GB of activations on an 80 GB card — 2.5 cards' worth, per card, and the model state was 2.19 GB.`
> ✓ `That is 201 GB of activations on an 80 GB card, against 2.19 GB of model state.`

**No bold-lead paragraph headers.** Don't open paragraphs with `**The budget.**`
or `**ZeRO-3 gather buffers.**`. If a chunk needs a label it needs a heading;
otherwise let the paragraph start with a sentence.

**No stage directions.** Don't announce the move before making it, and don't
echo the question's instructions back as a step. Putting the formula down *is*
stating the formula first; writing the derivation *is* deriving it.

> ✗ `**(b)** State the formula first:` / `**(d)** Derive it from the pass structure. Let $$F$$ be...`
> ✓ `**(b)** One checkpoint per layer, each the layer's input of shape $$[t, d]$$ in bf16:` / `**(d)** Let $$F$$ be...`

Same for `Now count.`, `First, frame the budget.`, `Putting it together`,
`Start with the activations.` Cut the sentence; the paragraph underneath already
does the thing.

**No enumerated takeaway wind-ups.** Drop "Three things to take away. First...
Second... Third...". State the points as plain sentences, or don't state them at
all if the table above already showed them.

**Don't dramatize numbers.** Report the quantity and its consequence. No "a
rounding error", "worked so hard to shrink", "the entire reason A2 exists",
"1.6× your entire checkpoint budget" as a flourish.

> ✗ `the thing that forced you onto 512 GPUs is now a rounding error`
> ✓ `Model state is 9% of the footprint here, so the remaining decisions are all about activations.`

**No contrastive punch-line closers.** Avoid the "X is not free", "the memory
argument is stronger than the speed argument", "it's not X, it's Y" shape as a
paragraph ender.

**Bold is for table headers and defined terms**, not for emphasising a number
mid-sentence. If a number matters, the sentence should already make that clear.

**No second-person scolding or hype.** No "you must", "worth knowing about",
"here's the thing". Explain, don't coach.

## The symptom underneath

Stage directions, bold-lead paragraph labels, and enumerated takeaway wind-ups
are one habit: narrating the shape of the answer instead of letting the answer
have a shape. It reads as performing rigor rather than being rigorous, and it is
the strongest single tell of LLM-written prose. The reader can see that a
paragraph is a new topic, that a formula came before its numbers, and that a
table has three rows. Announcing it adds words and subtracts trust.

The test: delete the sentence or label. If nothing is lost, it was scaffolding.

## What to keep

The technical content is good and should not be softened: itemized tables with
`{: .table .table-bordered .table-sm }`, explicit arithmetic in `$$...$$`,
per-part `**(a)**` `**(b)**` answer labels, stated assumptions. Precision and
completeness are the point. Only the connective prose changes.

## Default shape of an answer body

State what is being computed, show the arithmetic, give the number, then one or
two sentences on what it implies. Stop there.
