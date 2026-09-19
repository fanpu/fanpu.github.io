import { analyze, makeRng } from "../core/index.js";

// Promise API over the coach worker. If the worker cannot be started, stops answering or reports an error,
// the client quietly does the work on the main thread instead (with smaller samples), so the trainer always works.
const defaultFactory = () => new Worker(new URL("./coachWorker.js", import.meta.url), { type: "module" });
const MAIN_THREAD_ITERS = 3000;

export function createCoach({ workerFactory = defaultFactory, pingMs = 1500 } = {}) {
  const pending = new Map(); // id -> { resolve, reject, request }
  let nextId = 1,
    worker = null,
    mode = "worker",
    pingTimer = null;

  function onMain(id) {
    const job = pending.get(id);
    if (!job) return;
    setTimeout(() => {
      if (!pending.has(id)) return; // cancelled meanwhile
      pending.delete(id);
      const { state, reads, seed, opts } = job.request;
      try {
        job.resolve(analyze(state, reads, makeRng(seed), { ...opts, iters: Math.min(opts.iters ?? MAIN_THREAD_ITERS, MAIN_THREAD_ITERS) }));
      } catch (err) {
        job.reject(err);
      }
    }, 0);
  }
  function fallBack() {
    if (mode === "main") return;
    mode = "main";
    clearTimeout(pingTimer);
    try {
      worker?.terminate();
    } catch {
      /* already gone */
    }
    worker = null;
    for (const id of [...pending.keys()]) onMain(id); // nothing asked for is lost
  }

  try {
    worker = workerFactory();
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "pong") return clearTimeout(pingTimer);
      if (m.type === "error") return fallBack();
      const job = pending.get(m.id);
      if (!job) return; // a late answer to something cancelled
      pending.delete(m.id);
      job.resolve(m.analysis);
    };
    worker.onerror = fallBack;
    pingTimer = setTimeout(fallBack, pingMs);
    worker.postMessage({ type: "ping" });
  } catch {
    fallBack();
  }

  return {
    get mode() {
      return mode;
    },
    // state and reads are plain JSON (they are structured-cloned to the worker). opts: { light, iters }.
    analyze(state, reads, { seed, ...opts }) {
      const id = nextId++;
      const request = { state: structuredClone(state), reads: structuredClone(reads), seed, opts };
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject, request });
        if (mode === "main") onMain(id);
        else worker.postMessage({ type: "analyze", id, ...request });
      });
    },
    // The hand has moved on: nobody wants these answers any more.
    cancelAll() {
      for (const job of pending.values()) job.reject({ cancelled: true });
      pending.clear();
    },
    dispose() {
      this.cancelAll();
      clearTimeout(pingTimer);
      worker?.terminate();
      worker = null;
    },
  };
}
