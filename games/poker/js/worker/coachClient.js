import { analyze, simulate, makeRng } from "../core/index.js";

// Promise API over the coach worker. A module worker takes a second or two to load and answer its first ping
// (the deal animation covers that), so the client is patient before deciding the worker is not coming.
// Promise API over the coach worker. If the worker cannot be started, stops answering or reports an error,
// the client quietly does the work on the main thread instead (with smaller samples), so the trainer always works.
const defaultFactory = () => new Worker(new URL("./coachWorker.js", import.meta.url), { type: "module" });
const MAIN_THREAD_ITERS = 3000;

export function createCoach({ workerFactory = defaultFactory, pingMs = 8000 } = {}) {
  const pending = new Map(); // id -> { resolve, reject, request }
  let nextId = 1,
    worker = null,
    mode = "worker",
    pingTimer = null,
    reason = ""; // why the client fell back to the main thread, if it did

  // The same work, here. Used when there is no worker (or not yet).
  function compute(request) {
    const rng = makeRng(request.seed);
    if (request.type === "equity") return simulate(request.hero, request.board, request.ranges, Math.min(request.iters, MAIN_THREAD_ITERS), rng);
    return analyze(request.state, request.reads, rng, {
      ...request.opts,
      iters: Math.min(request.opts.iters ?? MAIN_THREAD_ITERS, MAIN_THREAD_ITERS),
      quality: 1,
    });
  }
  function onMain(id) {
    const job = pending.get(id);
    if (!job) return;
    setTimeout(() => {
      if (!pending.has(id)) return; // cancelled meanwhile
      pending.delete(id);
      try {
        job.resolve(compute(job.request));
      } catch (err) {
        job.reject(err);
      }
    }, 0);
  }
  function ask(request) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, request });
      if (mode === "main") onMain(id);
      else worker.postMessage({ id, ...request });
    });
  }

  // Work on the main thread from now on. A worker that is merely slow to load (a busy or modest device) is kept:
  // the moment it answers, the client goes back to using it. A worker that errored is gone for good.
  function fallBack(why, { keepWorker = false } = {}) {
    if (mode === "main") return;
    mode = "main";
    reason = String(why);
    clock.fellBackAt = Math.round(performance.now());
    clearTimeout(pingTimer);
    if (!keepWorker) {
      try {
        worker?.terminate();
      } catch {
        /* already gone */
      }
      worker = null;
    }
    for (const id of [...pending.keys()]) onMain(id); // nothing asked for is lost
  }

  const clock = { startedAt: Math.round(performance.now()), pongAt: null, fellBackAt: null }; // diagnostics
  try {
    worker = workerFactory();
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "pong") {
        clock.pongAt = Math.round(performance.now());
        clearTimeout(pingTimer);
        if (mode === "main" && worker) (mode = "worker"), (reason = ""); // it was only slow: use it from here on
        return;
      }
      if (m.type === "error") return fallBack("worker reported: " + m.message);
      const job = pending.get(m.id);
      if (!job) return; // a late answer to something cancelled
      pending.delete(m.id);
      job.resolve(m.analysis);
    };
    worker.onerror = (e) => fallBack("worker error: " + (e?.message || "failed to load"));
    pingTimer = setTimeout(() => fallBack("no answer to ping within " + pingMs + " ms", { keepWorker: true }), pingMs);
    worker.postMessage({ type: "ping" });
  } catch (e) {
    fallBack("could not start: " + (e?.message || e));
  }

  return {
    get mode() {
      return mode;
    },
    get fallbackReason() {
      return reason;
    },
    clock,
    // state and reads are plain JSON (they are structured-cloned to the worker). opts: { light, iters }.
    analyze: (state, reads, { seed, ...opts }) => ask({ type: "analyze", state: structuredClone(state), reads: structuredClone(reads), seed, opts }),
    // The hero's equity against one { pct, filters } range per opponent: the trail shown in the review.
    equity: (hero, board, ranges, { seed, iters = 4000 }) =>
      ask({ type: "equity", hero: structuredClone(hero), board: structuredClone(board), ranges: structuredClone(ranges), seed, iters }),
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
