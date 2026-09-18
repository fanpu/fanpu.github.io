---
title: "Research Sprint Day 4 - ???"
layout: distill
tags: [machine-learning, linear-attention]
cover: kearsarge_pass.webp
cover_preview: kearsarge_pass.webp
caption: Kearsarge Pass, Sierra Nevada, California, U.S
class: post-template
author: fanpu
giscus_comments: true
description: ""
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
toc: true
---

## MQAR

MQAR (multi-query associative recall) is a synthetic task from the [Zoology paper](https://arxiv.org/abs/2312.04927) that tests whether a model can bind a key to a value and look it up later. Each sequence starts with a block of key-value pairs, and the rest of the sequence is filler with those keys sprinkled back in as queries. Whenever a key shows up again, the model has to output the value it was paired with. We only compute loss at the query positions and mask out everything else.

The "multi-query" part matters. The earlier version of the task only had a single query at the end of the sequence, and that turned out to be too easy: gated convolutions solved it perfectly, but still did much worse than attention on real text. Once you ask for many lookups at different positions in the same forward pass, the gap with attention shows up again, which is why this is the version worth running.

The reason I like MQAR is that it's pretty close to a direct measurement of state capacity. A linear attention layer has a fixed-size state $S \in \mathbb{R}^{d_k \times d_v}$, so all $D$ key-value bindings in a sequence have to fit in those $d_k d_v$ numbers, regardless of sequence length. Softmax attention doesn't have this limit since its KV cache just grows with $N$. So instead of a single score, it's more informative to plot MQAR accuracy against state size, which gives a recall-memory frontier. The question for any subquadratic architecture is then where its curve sits.

The difficulty of the task is controlled by the number of key-value pairs, the sequence length, and the vocabulary size, which generally have to be scaled together, plus a fourth knob for how far each query is from its key. This means an MQAR accuracy number is only meaningful alongside its configuration, so I'll always report the two together.

Here is one sequence, with keys drawn from $\{a,b,c,d\}$, values from the digits, filler written as `.` and masked positions written as `-`.

```
position   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
input      a  4  b  3  c  6  d  1  .  .  c  .  a  .  d  .
labels     -  -  -  -  -  -  -  -  -  -  6  -  4  -  1  -
```

The model sees `c` at position 10 and has to predict `6`, recovering the binding `c 6` from positions 4 and 5. Positions 12 and 14 are the other two queries, and only those three positions contribute to the loss.


## Toy experiments

We first build some intuition on how the delta rule helps with interference from naive linear attention.

Recall their updates:

#### Additive linear attention:
$$
% Additive linear attention
\begin{align}
S_t &= S_{t-1} + v_t k_t^\top, \qquad S_0 = 0, \\
o_t &= S_t q_t = \sum_{i=1}^{t} v_i \,(k_i^\top q_t).
\end{align}
$$

#### Delta-rule linear attention:

$$
\begin{align}
S_t &= S_{t-1} - \beta_t \left( S_{t-1} k_t - v_t \right) k_t^\top \\
    &= S_{t-1} \left( I - \beta_t k_t k_t^\top \right) + \beta_t v_t k_t^\top, \\
o_t &= S_t q_t, \qquad \beta_t \in (0, 1).
\end{align}
$$

The linear update performs poorly due to progressive interference from subsequently added key-value state.
The delta-rule update makes it such that the most recent key that was added would have minimal retrieval error (but at the cost of interfering with previously added keys). 

A fun widget to play around with:

<iframe id="retrieval-toy" src="{{ '/assets/html/research_sprint/retrieval_toy.html' | relative_url }}" title="Additive vs delta rule retrieval toy" width="100%" height="1500" style="border:0; display:block;" loading="lazy"></iframe>
<script>
  (function () {
    var frame = document.getElementById("retrieval-toy");
    function sendTheme() {
      var t = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
      frame.contentWindow && frame.contentWindow.postMessage({ type: "retrieval-toy-theme", theme: t }, "*");
    }
    window.addEventListener("message", function (e) {
      if (e.source === frame.contentWindow && e.data && e.data.type === "retrieval-toy-height") frame.style.height = e.data.height + "px";
    });
    frame.addEventListener("load", sendTheme);
    new MutationObserver(sendTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  })();
</script>
