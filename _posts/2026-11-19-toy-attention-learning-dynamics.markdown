---
title: "Attention in the y = abx Toy Model"
layout: distill
tags: [machine-learning, learning-dynamics, attention]
cover: kinkakuji.webp
cover_preview: kinkakuji.webp
caption: Kinkaku-ji, Kyoto, Japan
class: post-template
author: fanpu
giscus_comments: true
description: "Learning dynamics of a scalar linear attention layer on in-context regression"
authors:
  - name: Fan Pu Zeng
    url: "https://fanpu.io"
  - name: James Ang Ming Liang
    url: "https://jamesmldoesml.github.io"
toc: true
---

This is joint work with [James Ang Ming Liang](https://jamesmldoesml.github.io),
and a follow-up to his post on
[training dynamics](https://jamesmldoesml.github.io/blogs/learning-dynamics/).
There he studies gradient descent on the toy model $f(x) = abx$, which is two
scalar weights multiplied together. Even that small model has a whole manifold of
solutions, a conserved quantity, and a saddle at the origin.

Here we ask what the analogous toy model for attention looks like.

## Setup

As in the $abx$ model, every token and every weight is a single number. The input is a sequence of $n$ context tokens
$(x_1, y_1), \dots, (x_n, y_n)$, followed by a query $x_q$. The model has to
produce a prediction $\hat{y}$ for the query.

Give the layer a scalar key weight $k$, query weight $q$ and value weight $v$.
Each context token offers key $k x_i$ and value $v y_i$. The query is $q x_q$.
Linear attention averages value times key times query:

$$
\hat{y} = \frac{1}{n} \sum_{i=1}^{n} (v\, y_i)(k\, x_i)(q\, x_q).
$$

There is no softmax here, so the weight on each value is the raw product of
its key with the query.

## In-context regression

Each sequence has its own hidden slope $\theta$. The context shows $n$ examples
with $y_i = \theta x_i$, and the model must output $\theta x_q$ for the query.
The slope changes from sequence to sequence, so the only way to do well is to
read $\theta$ off the context. We draw
$\theta, x_i, x_q \sim \mathcal{N}(0, 1)$ independently.

Plugging $y_i = \theta x_i$ into the model gives

$$
\hat{y} = \frac{1}{n} \sum_{i=1}^{n} (v\, \theta x_i)(k\, x_i)(q\, x_q)
= \underbrace{(v\, k\, q)}_{p}\, \theta\, S\, x_q,
\qquad S = \frac{1}{n} \sum_{i=1}^{n} x_i^2.
$$

Only the product $p = vkq$ matters, and $S$ is the average of $n$ squared
normal numbers, which is close to 1 for long contexts.

### Closed form of the loss

We train with squared error, averaged over sequences:

$$
L(p) = \tfrac12\, \mathbb{E}\big[(\hat{y} - \theta x_q)^2\big].
$$

This has a closed form:

$$
L(p) = \frac{n+2}{2n}\,(p - p^*)^2 + \frac{1}{n+2},
\qquad p^* = \frac{n}{n+2}.
$$

So the best product is $p^* = \frac{n}{n+2}$, and even there the loss cannot go
below $\frac{1}{n+2}$.

The plot below shows this loss as a function of $p$. The line is the closed
form, and the circles are the average loss of the model's predictions on 30,000
sampled sequences. Drag the slider to change the context length.

{% raw %}
<div class="icl-fig" id="icl-fig" markdown="0">
  <style>
    .icl-fig { margin: 1.5rem 0; font-size: 0.9rem; color: var(--global-text-color); --icl-best: #2e8b57; }
    html[data-theme="dark"] .icl-fig { --icl-best: #5fcf95; }
    .icl-fig .icl-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
    .icl-fig input[type="range"] { width: 220px; max-width: 60vw; accent-color: var(--global-theme-color); }
    .icl-fig svg { width: 100%; height: auto; display: block; overflow: visible; }
    .icl-fig .grid { stroke: var(--global-divider-color); stroke-width: 1; }
    .icl-fig .axis { stroke: var(--global-text-color-light); stroke-width: 1; }
    .icl-fig text { fill: var(--global-text-color-light); font-size: 12px; }
    .icl-fig .curve { fill: none; stroke: var(--global-text-color); stroke-width: 2; }
    .icl-fig .pt { fill: var(--global-bg-color); stroke: var(--global-theme-color); stroke-width: 2; }
    .icl-fig .best { fill: var(--icl-best); stroke: var(--global-bg-color); stroke-width: 2; }
    .icl-fig .guide { stroke: var(--icl-best); stroke-width: 1.25; stroke-dasharray: 5 4; }
    .icl-fig .cross { stroke: var(--global-text-color-light); stroke-width: 1; visibility: hidden; }
    .icl-fig .legend { display: flex; flex-wrap: wrap; gap: 1.25rem; margin-top: 0.25rem; color: var(--global-text-color-light); }
    .icl-fig .legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
    .icl-fig .sw-line { width: 22px; height: 0; border-top: 2px solid var(--global-text-color); }
    .icl-fig .sw-pt { width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--global-theme-color); }
    .icl-fig .sw-best { width: 10px; height: 10px; border-radius: 50%; background: var(--icl-best); }
    .icl-fig .readout { margin-top: 0.5rem; font-variant-numeric: tabular-nums; }
    .icl-fig .tip { position: absolute; pointer-events: none; visibility: hidden; padding: 4px 8px; border-radius: 4px;
      background: var(--global-bg-color); border: 1px solid var(--global-divider-color); color: var(--global-text-color);
      font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .icl-fig .plot { position: relative; }
  </style>
  <div class="icl-row">
    <label for="icl-n">Context length <i>n</i></label>
    <input id="icl-n" type="range" min="1" max="64" step="1" value="4">
    <span id="icl-n-val">4</span>
  </div>
  <div class="plot">
    <svg id="icl-svg" viewBox="0 0 640 380" role="img" aria-label="Loss of scalar linear attention as a function of the product p"></svg>
    <div class="tip" id="icl-tip"></div>
  </div>
  <div class="legend">
    <span><i class="sw-line"></i>closed form</span>
    <span><i class="sw-pt"></i>sampled</span>
    <span><i class="sw-best"></i>best <i>p</i> and the loss floor</span>
  </div>
  <div class="readout">
    best product <i>p</i>* = <i>n</i>/(<i>n</i>+2) = <b id="icl-ps"></b>
    &nbsp;&nbsp; loss floor 1/(<i>n</i>+2) = <b id="icl-fl"></b>
  </div>
</div>
<script>
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var svg = document.getElementById("icl-svg"), tip = document.getElementById("icl-tip");
  var slider = document.getElementById("icl-n");
  var W = 640, H = 380, m = { l: 48, r: 14, t: 12, b: 44 };
  var X0 = 0, X1 = 1.6, Y0 = 0, Y1 = 0.6, NSAMP = 30000;
  function sx(x) { return m.l + (x - X0) / (X1 - X0) * (W - m.l - m.r); }
  function sy(y) { return H - m.b - (y - Y0) / (Y1 - Y0) * (H - m.t - m.b); }
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  }
  function rng(seed) {
    var a = seed >>> 0, spare = null;
    function u() { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var r = Math.sqrt(-2 * Math.log(1 - u())), th = 2 * Math.PI * u();
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  }
  function loss(p, n) { return 0.5 * (p * p * (n + 2) / n - 2 * p + 1); }
  var cur = 4;
  function draw() {
    var n = cur = Math.round(+slider.value), ps = n / (n + 2), fl = 1 / (n + 2), i, j;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    for (i = 0; i <= 8; i++) {
      var gx = i * 0.2;
      el("line", { class: "grid", x1: sx(gx), x2: sx(gx), y1: sy(Y0), y2: sy(Y1) });
      el("text", { x: sx(gx), y: sy(Y0) + 16, "text-anchor": "middle" }, gx.toFixed(1));
    }
    for (i = 0; i <= 6; i++) {
      var gy = i * 0.1;
      el("line", { class: "grid", x1: sx(X0), x2: sx(X1), y1: sy(gy), y2: sy(gy) });
      el("text", { x: sx(X0) - 8, y: sy(gy) + 4, "text-anchor": "end" }, gy.toFixed(1));
    }
    el("line", { class: "axis", x1: sx(X0), x2: sx(X1), y1: sy(Y0), y2: sy(Y0) });
    el("line", { class: "axis", x1: sx(X0), x2: sx(X0), y1: sy(Y0), y2: sy(Y1) });
    el("text", { x: sx((X0 + X1) / 2), y: H - 6, "text-anchor": "middle" }, "product p = vkq");
    el("text", { x: 12, y: sy((Y0 + Y1) / 2), "text-anchor": "middle", transform: "rotate(-90 12 " + sy((Y0 + Y1) / 2) + ")" }, "loss");
    el("line", { class: "guide", x1: sx(ps), x2: sx(ps), y1: sy(Y0), y2: sy(Y1) });
    el("line", { class: "guide", x1: sx(X0), x2: sx(X1), y1: sy(fl), y2: sy(fl) });
    var d = "", started = false;
    for (i = 0; i <= 320; i++) {
      var p = X0 + (X1 - X0) * i / 320, L = loss(p, n);
      if (L > Y1) { started = false; continue; }
      d += (started ? "L" : "M") + sx(p).toFixed(1) + " " + sy(L).toFixed(1);
      started = true;
    }
    el("path", { class: "curve", d: d });
    // Monte Carlo: average the squared error of the model's predictions over sampled sequences.
    var g = rng(11 + n), S = new Float64Array(NSAMP), Wt = new Float64Array(NSAMP);
    for (i = 0; i < NSAMP; i++) {
      var q = 0;
      for (j = 0; j < n; j++) { var x = g(); q += x * x; }
      S[i] = q / n;
      var th = g(), xq = g();
      Wt[i] = th * th * xq * xq;
    }
    for (var k = 1; k <= 15; k++) {
      var pp = k / 10, acc = 0;
      for (i = 0; i < NSAMP; i++) { var e = pp * S[i] - 1; acc += Wt[i] * e * e; }
      var est = 0.5 * acc / NSAMP;
      if (est <= Y1) el("circle", { class: "pt", cx: sx(pp), cy: sy(est), r: 4 });
    }
    el("circle", { class: "best", cx: sx(ps), cy: sy(fl), r: 6 });
    cross = el("line", { class: "cross", y1: sy(Y0), y2: sy(Y1) });
    document.getElementById("icl-n-val").textContent = n;
    document.getElementById("icl-ps").textContent = ps.toFixed(3);
    document.getElementById("icl-fl").textContent = fl.toFixed(3);
  }
  var cross = null;
  svg.addEventListener("mousemove", function (ev) {
    var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    var p = X0 + (px - m.l) / (W - m.l - m.r) * (X1 - X0);
    if (p < X0 || p > X1) { hide(); return; }
    cross.setAttribute("x1", sx(p)); cross.setAttribute("x2", sx(p)); cross.style.visibility = "visible";
    tip.textContent = "p = " + p.toFixed(2) + ", loss = " + loss(p, cur).toFixed(3);
    tip.style.visibility = "visible";
    var left = ev.clientX - r.left + 12;
    if (left + tip.offsetWidth > r.width) left = ev.clientX - r.left - tip.offsetWidth - 12;
    tip.style.left = left + "px"; tip.style.top = (ev.clientY - r.top - 30) + "px";
  });
  function hide() { if (cross) cross.style.visibility = "hidden"; tip.style.visibility = "hidden"; }
  svg.addEventListener("mouseleave", hide);
  slider.addEventListener("input", draw);
  draw();
})();
</script>
{% endraw %}

With a short context the best product is well below 1 and the loss cannot
reach zero. As $n$ grows, $p^*$ moves toward 1 and the floor goes to zero.

{% answer Derivation %}

**Step 1. Write the error for one sequence.** The prediction is
$\hat{y} = p\,\theta S x_q$ and the target is $\theta x_q$. Factor out what they
share:

$$
\hat{y} - \theta x_q = \theta x_q\,(pS - 1).
$$

So the loss on one sequence is

$$
\tfrac12(\hat{y} - \theta x_q)^2 = \tfrac12\,\theta^2\,x_q^2\,(pS-1)^2.
$$

**Step 2. Average, using independence.** $\theta$, $x_q$ and the $x_i$ are all
drawn independently, and $S$ depends only on the $x_i$. The expectation of a
product of independent things is the product of expectations:

$$
L(p) = \tfrac12\;\mathbb{E}[\theta^2]\;\mathbb{E}[x_q^2]\;\mathbb{E}\big[(pS-1)^2\big].
$$

Both $\theta$ and $x_q$ are standard normal, so
$\mathbb{E}[\theta^2] = \mathbb{E}[x_q^2] = 1$ and they drop out:

$$
L(p) = \tfrac12\,\mathbb{E}\big[(pS-1)^2\big]
= \tfrac12\Big(p^2\,\mathbb{E}[S^2] - 2p\,\mathbb{E}[S] + 1\Big).
$$

So we only need two numbers, $\mathbb{E}[S]$ and $\mathbb{E}[S^2]$.

**Step 3. Compute those two numbers.** Recall $S = \frac1n\sum_i x_i^2$.

For the mean, each $\mathbb{E}[x_i^2] = 1$, so $\mathbb{E}[S] = 1$.

For the second moment, use
$\mathbb{E}[S^2] = \mathrm{Var}(S) + (\mathbb{E}[S])^2 = \mathrm{Var}(S) + 1$.
Since $S$ is an average of $n$ independent terms,

$$
\mathrm{Var}(S) = \frac{1}{n^2}\cdot n\cdot \mathrm{Var}(x_i^2) = \frac{\mathrm{Var}(x_i^2)}{n}.
$$

A standard normal has fourth moment $\mathbb{E}[x^4] = 3$, so

$$
\mathrm{Var}(x_i^2) = \mathbb{E}[x_i^4] - \big(\mathbb{E}[x_i^2]\big)^2 = 3 - 1 = 2.
$$

Therefore $\mathrm{Var}(S) = \frac2n$ and

$$
\mathbb{E}[S^2] = 1 + \frac2n = \frac{n+2}{n}.
$$

(The fourth moment comes from integration by parts on the Gaussian density,
which gives $\mathbb{E}[x^{k+1}] = k\,\mathbb{E}[x^{k-1}]$. With $k = 3$ that is
$\mathbb{E}[x^4] = 3\,\mathbb{E}[x^2] = 3$.)

**Step 4. Substitute.**

$$
L(p) = \tfrac12\Big(p^2\,\tfrac{n+2}{n} - 2p + 1\Big).
$$

**The minimum and the floor.** Set the derivative to zero:

$$
\frac{dL}{dp} = p\,\tfrac{n+2}{n} - 1 = 0 \quad\Rightarrow\quad p^* = \frac{n}{n+2}.
$$

Plug back in:

$$
L(p^*) = \tfrac12\Big(\tfrac{n}{n+2} - \tfrac{2n}{n+2} + 1\Big)
= \tfrac12\cdot\tfrac{2}{n+2} = \frac{1}{n+2}.
$$

Completing the square gives the form above:

$$
L(p) = \frac{n+2}{2n}\,(p - p^*)^2 + \frac{1}{n+2}.
$$
{% endanswer %}

Compare this with $\tfrac12(ab - 1)^2$ from the $abx$ model. It is the same
loss in the product $p$, with the target moved from 1 to $p^*$, the curvature
rescaled by $\tfrac{n+2}{n}$, and a constant floor added.

The derivation used three things about the data. Independence let $\theta$ and
$x_q$ drop out, $\mathbb{E}[x^2] = 1$ put the ideal $p$ near 1, and
$\mathbb{E}[x^4] = 3$ set how far $p^*$ shrinks below 1 and how high the floor
is. For a different input distribution with unit variance, only the fourth
moment changes.

### Gradient flow on the three weights

Since the loss depends on the weights only through $p = vkq$, the chain rule
gives $\frac{\partial L}{\partial v} = L'(p)\,kq$, where

$$
L'(p) = Ap - 1, \qquad A = \frac{n+2}{n}.
$$

The same holds for $k$ and $q$ with the roles swapped, so gradient flow is

$$
\dot v = (1 - Ap)\,kq, \qquad \dot k = (1 - Ap)\,vq, \qquad \dot q = (1 - Ap)\,vk.
$$

The factor $(1 - Ap)$ is common to all three equations. It is close to 1 when
the weights are small and reaches 0 at $p = p^*$. The equations differ in the
second factor, which for each weight is the product of the other two.

### Observations

We can read a few things off these equations.

1. A weight only moves when the other two are both nonzero. If all three start
   small, each of them has a tiny gradient, and the loss sits on a plateau
   until they have grown enough to pull each other along.

2. The smallest weights grow fastest. If $v$ is large and $k$, $q$ are small,
   then $\dot v$ is proportional to $kq$ and is tiny, while $\dot k$ and
   $\dot q$ each pick up a factor of $v$. So $v$ plays the role of a learning
   rate for $k$ and $q$.

3. The differences $v^2 - k^2$ and $k^2 - q^2$ stay constant during training.
   To see this, multiply each equation by its own weight:

   $$
   v\,\dot v = k\,\dot k = q\,\dot q = (1 - Ap)\,vkq.
   $$

   Since $v\dot v = \tfrac12 \frac{d}{dt} v^2$, this says that $v^2$, $k^2$
   and $q^2$ all grow at the same rate. One consequence is that weights that
   start equal stay equal, and with $v = k = q = w$ the dynamics reduce to
   $\dot w = (1 - Ap)\,w^2$.

4. How long the plateau lasts depends on how many of the weights start small.
   Suppose $k$ and $q$ start at size $\varepsilon$ and $v \approx 1$. Early on
   $\dot k \approx q$ and $\dot q \approx k$, so they grow exponentially and
   leave the plateau after a time of about $\ln(1/\varepsilon)$. If instead
   all three start at $\varepsilon$, then $\dot w \approx w^2$, which has
   solution $w(t) = 1/(1/\varepsilon - t)$, and the plateau lasts about
   $1/\varepsilon$. If we merge the key and query into one weight $w$, so that
   $p = vw$, we are back to the $abx$ model with $\dot v = (1 - Ap)\,w$ and
   $\dot w = (1 - Ap)\,v$, and the plateau is again logarithmic in
   $1/\varepsilon$.

5. The only thing we used about attention is that its output depends on a
   product of scalar weights. Any model like that has the same equations and
   the same conserved quantities. Deep linear networks have a matrix version:
   $W_{i+1}^\top W_{i+1} - W_i W_i^\top$ is conserved for adjacent layers, a
   property called balancedness.
