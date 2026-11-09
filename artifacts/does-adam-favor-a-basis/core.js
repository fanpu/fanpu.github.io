// Delta-rule memory: forward, exact backward, GD/Adam training, paired-run experiment.
// Plain JS, no dependencies.

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// W: array of V columns, each Float64Array(dk)
function normalizeKeys(W) {
  return W.map(w => { const n = Math.hypot(...w); return w.map(x => x / n); });
}
function gram(K) {
  const V = K.length, G = [];
  for (let a = 0; a < V; a++) { G.push([]); for (let c = 0; c < V; c++) G[a].push(dot(K[a], K[c])); }
  return G;
}
function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }

// Episode: k distinct keys, values uniform in Vb, write phase then random-order query phase.
function sampleEpisode(rng, V, Vb, k) {
  const idx = [...Array(V).keys()];
  for (let i = V - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const keys = idx.slice(0, k);
  const vals = keys.map(() => Math.floor(rng() * Vb));
  const perm = [...Array(k).keys()];
  for (let i = k - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const events = [];
  for (let i = 0; i < k; i++) events.push({ a: keys[i], b: vals[i], scored: false });
  for (const i of perm) events.push({ a: keys[i], b: vals[i], scored: true });
  return events;
}

// Forward + backward for one episode. Returns loss, correct count, and accumulates dW (array of columns).
// S is Vb x dk stored as array of Vb rows (Float64Array(dk)).
function episodeGrad(W, K, events, beta, gscale, Vb, dW, wantGrad) {
  const dk = K[0].length;
  let S = Array.from({ length: Vb }, () => new Float64Array(dk));
  const recs = [];
  let loss = 0, correct = 0, nscored = 0;
  for (const ev of events) if (ev.scored) nscored++;
  for (const ev of events) {
    const ka = K[ev.a];
    const r = S.map(row => dot(row, ka));
    let dr = null;
    if (ev.scored) {
      const logits = r.map(x => gscale * x);
      const m = Math.max(...logits);
      const ex = logits.map(x => Math.exp(x - m));
      const Z = ex.reduce((s, x) => s + x, 0);
      const p = ex.map(x => x / Z);
      loss += -Math.log(p[ev.b]) / nscored;
      let am = 0; for (let c = 1; c < Vb; c++) if (logits[c] > logits[am]) am = c;
      if (am === ev.b) correct += 1 / nscored;
      dr = p.map((pc, c) => gscale * (pc - (c === ev.b ? 1 : 0)) / nscored);
    }
    const Sprev = S;
    const delta = r.map((x, c) => (c === ev.b ? 1 : 0) - x);
    S = Sprev.map((row, c) => row.map((x, j) => x + beta * delta[c] * ka[j]));
    if (wantGrad) recs.push({ a: ev.a, Sprev, delta, dr });
  }
  if (wantGrad) {
    let dS = Array.from({ length: Vb }, () => new Float64Array(dk));
    const dK = K.map(() => new Float64Array(dk));
    for (let t = recs.length - 1; t >= 0; t--) {
      const { a, Sprev, delta, dr } = recs[t];
      const ka = K[a];
      // undo write: S_new = Sprev + beta * delta * ka^T, delta = e_b - Sprev ka
      const dSk = dS.map(row => dot(row, ka));            // Vb
      for (let j = 0; j < dk; j++) {
        let t1 = 0, t2 = 0;
        for (let c = 0; c < Vb; c++) { t1 += dS[c][j] * delta[c]; t2 += Sprev[c][j] * dSk[c]; }
        dK[a][j] += beta * (t1 - t2);
      }
      dS = dS.map((row, c) => row.map((x, j) => x - beta * dSk[c] * ka[j]));
      // read at this state (if scored)
      if (dr) {
        for (let c = 0; c < Vb; c++) for (let j = 0; j < dk; j++) {
          dS[c][j] += dr[c] * ka[j];
          dK[a][j] += Sprev[c][j] * dr[c];
        }
      }
    }
    // through normalization k = w/|w|
    for (let a = 0; a < W.length; a++) {
      const w = W[a], k = K[a], n = Math.hypot(...w);
      const kd = dot(k, dK[a]);
      for (let j = 0; j < dk; j++) dW[a][j] += (dK[a][j] - k[j] * kd) / n;
    }
  }
  return { loss, acc: correct };
}

function batchGrad(W, episodes, cfg) {
  const K = normalizeKeys(W);
  const dW = W.map(() => new Float64Array(W[0].length));
  let loss = 0, acc = 0;
  for (const ep of episodes) {
    const r = episodeGrad(W, K, ep, cfg.beta, cfg.g, cfg.Vb, dW, true);
    loss += r.loss / episodes.length; acc += r.acc / episodes.length;
  }
  for (const col of dW) for (let j = 0; j < col.length; j++) col[j] /= episodes.length;
  return { loss, acc, dW };
}

function evaluate(W, episodes, cfg) {
  const K = normalizeKeys(W);
  let loss = 0, acc = 0;
  for (const ep of episodes) {
    const r = episodeGrad(W, K, ep, cfg.beta, cfg.g, cfg.Vb, null, false);
    loss += r.loss / episodes.length; acc += r.acc / episodes.length;
  }
  return { loss, acc };
}

function randomOrthogonal(rng, d) {
  // Gram-Schmidt on a Gaussian matrix, columns = basis vectors. Q[i][j].
  const cols = [];
  for (let c = 0; c < d; c++) {
    let v = Float64Array.from({ length: d }, () => gauss(rng));
    for (const u of cols) { const p = dot(u, v); for (let i = 0; i < d; i++) v[i] -= p * u[i]; }
    const n = Math.hypot(...v); v = v.map(x => x / n);
    cols.push(v);
  }
  const Q = Array.from({ length: d }, (_, i) => Float64Array.from({ length: d }, (_, j) => cols[j][i]));
  return Q;
}
function applyQ(Q, W) { // Q W, column-wise
  return W.map(w => Float64Array.from(Q, row => dot(row, w)));
}
function axisConcentration(K) {
  let s = 0;
  for (const k of K) for (const x of k) s += x * x * x * x;
  return s / K.length;
}
function frobDiff(A, B) {
  let s = 0;
  for (let a = 0; a < A.length; a++) for (let j = 0; j < A[a].length; j++) { const d = A[a][j] - B[a][j]; s += d * d; }
  return Math.sqrt(s);
}
function frobDiffGram(GA, GB) {
  let s = 0;
  for (let a = 0; a < GA.length; a++) for (let c = 0; c < GA.length; c++) { const d = GA[a][c] - GB[a][c]; s += d * d; }
  return Math.sqrt(s);
}
function meanColNorm(W) { return W.reduce((s, w) => s + Math.hypot(...w), 0) / W.length; }

function makeOptimizer(kind, lr, shape) {
  if (kind === 'gd') return { step(W, dW) { for (let a = 0; a < W.length; a++) for (let j = 0; j < W[a].length; j++) W[a][j] -= lr * dW[a][j]; } };
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const m = shape.map(() => new Float64Array(shape[0].length));
  const v = shape.map(() => new Float64Array(shape[0].length));
  let t = 0;
  return { step(W, dW) {
    t++;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    for (let a = 0; a < W.length; a++) for (let j = 0; j < W[a].length; j++) {
      const g = dW[a][j];
      m[a][j] = b1 * m[a][j] + (1 - b1) * g;
      v[a][j] = b2 * v[a][j] + (1 - b2) * g * g;
      W[a][j] -= lr * (m[a][j] / c1) / (Math.sqrt(v[a][j] / c2) + eps);
    }
  } };
}

// Paired experiment: runs {gd, gdQ, adam, adamQ} on identical episode streams.
// Returns a generator-like stepper so the page can animate.
function makeExperiment(cfg) {
  const { dk, V, Vb, k, steps, batch, lrGD, lrAdam, seed, beta, g } = cfg;
  const rng0 = mulberry32(seed);
  const W0 = Array.from({ length: V }, () => Float64Array.from({ length: dk }, () => gauss(rng0)));
  const Q = randomOrthogonal(mulberry32(seed + 1), dk);
  const W0Q = applyQ(Q, W0);
  const runs = {
    gd:    { W: W0.map(w => Float64Array.from(w)),  opt: makeOptimizer('gd', lrGD, W0) },
    gdQ:   { W: W0Q.map(w => Float64Array.from(w)), opt: makeOptimizer('gd', lrGD, W0) },
    adam:  { W: W0.map(w => Float64Array.from(w)),  opt: makeOptimizer('adam', lrAdam, W0) },
    adamQ: { W: W0Q.map(w => Float64Array.from(w)), opt: makeOptimizer('adam', lrAdam, W0) },
  };
  const evalRng = mulberry32(seed + 7);
  const evalEps = Array.from({ length: 200 }, () => sampleEpisode(evalRng, V, Vb, k));
  const epRng = mulberry32(seed + 3);
  const hist = { t: [], divGD: [], divAdam: [], gramGD: [], gramAdam: [], aGD: [], aAdam: [], aGDQ: [], aAdamQ: [], lossGD: [], lossAdam: [] };
  let t = 0;
  function record() {
    const Ks = {}; for (const n in runs) Ks[n] = normalizeKeys(runs[n].W);
    hist.t.push(t);
    hist.divGD.push(frobDiff(runs.gdQ.W, applyQ(Q, runs.gd.W)) / Math.sqrt(runs.gd.W.reduce((s, w) => s + dot(w, w), 0)));
    hist.divAdam.push(frobDiff(runs.adamQ.W, applyQ(Q, runs.adam.W)) / Math.sqrt(runs.adam.W.reduce((s, w) => s + dot(w, w), 0)));
    hist.gramGD.push(frobDiffGram(gram(Ks.gd), gram(Ks.gdQ)));
    hist.gramAdam.push(frobDiffGram(gram(Ks.adam), gram(Ks.adamQ)));
    hist.aGD.push(axisConcentration(Ks.gd)); hist.aGDQ.push(axisConcentration(Ks.gdQ));
    hist.aAdam.push(axisConcentration(Ks.adam)); hist.aAdamQ.push(axisConcentration(Ks.adamQ));
  }
  record();
  return {
    cfg, Q, runs, hist, evalEps,
    get t() { return t; },
    done() { return t >= steps; },
    step(n) {
      for (let i = 0; i < n && t < steps; i++) {
        const eps = Array.from({ length: batch }, () => sampleEpisode(epRng, V, Vb, k));
        for (const name in runs) {
          const r = runs[name];
          const { loss, dW } = batchGrad(r.W, eps, cfg);
          if (name === 'gd') hist.lossGD.push(loss);
          if (name === 'adam') hist.lossAdam.push(loss);
          r.opt.step(r.W, dW);
        }
        t++;
        if (t % Math.max(1, Math.floor(steps / 100)) === 0 || t === steps) record();
      }
    },
    summary() {
      const out = {};
      for (const name in runs) {
        const K = normalizeKeys(runs[name].W);
        const ev = evaluate(runs[name].W, evalEps, cfg);
        // random-rotation baseline for this K
        const brng = mulberry32(seed + 11);
        const vals = [];
        for (let i = 0; i < 64; i++) vals.push(axisConcentration(applyQ(randomOrthogonal(brng, dk), K)));
        const mean = vals.reduce((s, x) => s + x, 0) / 64;
        const sd = Math.sqrt(vals.reduce((s, x) => s + (x - mean) ** 2, 0) / 64);
        out[name] = { acc: ev.acc, loss: ev.loss, A: axisConcentration(K), Abase: mean, Asd: sd, norm: meanColNorm(runs[name].W), K };
      }
      return out;
    }
  };
}

if (typeof module !== 'undefined') module.exports = { mulberry32, gauss, normalizeKeys, gram, sampleEpisode, episodeGrad, batchGrad, evaluate, randomOrthogonal, applyQ, axisConcentration, frobDiff, frobDiffGram, makeExperiment, makeOptimizer, dot };
