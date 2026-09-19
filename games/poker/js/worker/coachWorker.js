import { analyze, simulate, makeRng } from "../core/index.js";

// The coach, off the main thread: Monte Carlo equity and EV sampling must never cost the table a frame.
// Each request carries its own seed, so an answer does not depend on when it happened to be computed.
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "ping") return self.postMessage({ type: "pong" });
  try {
    const rng = makeRng(m.seed);
    const answer = m.type === "equity" ? simulate(m.hero, m.board, m.ranges, m.iters, rng) : analyze(m.state, m.reads, rng, m.opts);
    self.postMessage({ type: "result", id: m.id, analysis: answer });
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, message: String(err?.message || err) });
  }
};
