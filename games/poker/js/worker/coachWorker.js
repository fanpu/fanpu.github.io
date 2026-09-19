import { analyze, makeRng } from "../core/index.js";

// The coach, off the main thread: Monte Carlo equity and EV sampling must never cost the table a frame.
// Each request carries its own seed, so an answer does not depend on when it happened to be computed.
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "ping") return self.postMessage({ type: "pong" });
  try {
    self.postMessage({ type: "result", id: m.id, analysis: analyze(m.state, m.reads, makeRng(m.seed), m.opts) });
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, message: String(err?.message || err) });
  }
};
